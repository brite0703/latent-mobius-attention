"""Supplied-cache scoring and complete-replicate summaries; no dataset reader.

The future campaign controller must verify the common selection and authorize
actual test-cache creation before using these kernels on scientific test data.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch

import execution
from models import ARMS, PairResidual
import selection

PROCEDURES = ("baseline",) + ARMS
PRIMARY = {"cubic": "mse", "ligand_contact": "rmse"}
UNAVAILABLE = ("missing_parent", "no_valid_residual", "failed_evaluation")


class EvaluationFailure(RuntimeError):
    """Numerical prediction/metric failure, never a reason to reselect."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or not len(truth) or prediction.shape != truth.shape:
        raise ValueError("Aligned nonempty scalar prediction and target vectors are required")
    if not np.isfinite(truth).all():
        raise ValueError("Targets contain invalid values")
    if not np.isfinite(prediction).all():
        raise EvaluationFailure("The selected predictor produced nonfinite values")
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = prediction-truth
            mse = float(np.mean(difference*difference, dtype=np.float64))
            mae = float(np.mean(np.abs(difference), dtype=np.float64))
    except FloatingPointError as error:
        raise EvaluationFailure("Selected-predictor metric arithmetic overflowed") from error
    if not (math.isfinite(mse) and math.isfinite(mae)):
        raise EvaluationFailure("Selected-predictor metrics are nonfinite")
    return dict(mse=mse, rmse=math.sqrt(mse), mae=mae)


def identity(domain, seed, procedure):
    if domain not in selection.DOMAINS or seed not in selection.DOMAINS[domain] or procedure not in PROCEDURES:
        raise ValueError("An outcome must belong to a prespecified domain, replicate and procedure")
    return dict(domain=domain, seed=seed, procedure=procedure)


def cache_identity(cache, domain, seed, procedure):
    row = identity(domain, seed, procedure)
    cache.verify()
    if cache.split != "test" or cache.z.shape[-1] != (12 if domain == "cubic" else 8):
        raise ValueError("Final scoring requires its own domain's supplied test cache")
    if domain == "cubic" and (cache.target_mean, cache.target_sd) != (0., 1.):
        raise ValueError("The original cubic parent uses identity target units")
    row.update(parent_checkpoint_sha256=cache.parent_checkpoint_sha256, cache_digest=cache.digest,
               record_count=len(cache.ids), target_mean=cache.target_mean, target_sd=cache.target_sd,
               record_ids_sha256=hashlib.sha256(json.dumps(cache.ids, separators=(",", ":")).encode()).hexdigest(),
               targets_sha256=hashlib.sha256(cache.truth.contiguous().numpy().tobytes()).hexdigest())
    return row


def unavailable(domain, seed, procedure, status, reason, parent_sha=None):
    row = identity(domain, seed, procedure)
    if status not in ("missing_parent", "no_valid_residual") or not isinstance(reason, str) or not reason.strip():
        raise ValueError("A missing outcome requires its declared category and explicit reason")
    if status == "no_valid_residual" and (procedure == "baseline" or not isinstance(parent_sha, str) or len(parent_sha) != 64):
        raise ValueError("Only residual procedures with a known parent can lack valid rates")
    if status == "missing_parent" and parent_sha is not None:
        raise ValueError("Do not attach a substituted checkpoint to a missing parent")
    row.update(status=status, reason=reason, parent_checkpoint_sha256=parent_sha, metrics=None,
               cache_digest=None, record_count=None, target_mean=None, target_sd=None,
               record_ids_sha256=None, targets_sha256=None,
               candidate_id=None, selected_epoch=None, selection_outcome=None, residual_checkpoint=None)
    return row


def baseline(cache, domain, seed):
    row = cache_identity(cache, domain, seed, "baseline")
    row.update(candidate_id=None, selected_epoch=None, selection_outcome="unchanged_parent", residual_checkpoint=None)
    parent_units = cache.f0.detach().clone()
    original_units = parent_units.double()*cache.target_sd+cache.target_mean
    try:
        measured = metrics(cache.truth.numpy(), original_units.numpy())
    except EvaluationFailure as error:
        cache.verify()
        row.update(status="failed_evaluation", metrics=None, reason=str(error))
        return row, None
    cache.verify()
    row.update(status="evaluated", metrics=measured, reason=None)
    vectors = dict(ids=list(cache.ids), truth=cache.truth.detach().clone(),
                   prediction_parent_units=parent_units, prediction_original_units=original_units)
    return row, vectors


def residual(cache, spec, choice, checkpoint_record, parent_sha, *, batch_size=256):
    selection.validate_spec(spec)
    if parent_sha != cache.parent_checkpoint_sha256:
        raise ValueError("The residual and cached baseline have different parents")
    for key in ("domain", "seed", "arm"):
        if choice[key] != spec[key]:
            raise ValueError("The supplied choice does not belong to this procedure")
    pair = sorted((s for s in selection.candidate_plan() if (s["domain"], s["seed"], s["arm"]) == (
        spec["domain"], spec["seed"], spec["arm"])), key=lambda s: s["rate_index"])
    if choice["candidate_ids"] != [s["id"] for s in pair] or len(choice["candidate_statuses"]) != 2:
        raise ValueError("The selection does not enumerate both declared rates")
    if choice["selected_id"] != spec["id"] or choice["candidate_statuses"][spec["rate_index"]] != "valid":
        raise ValueError("Only the selected valid rate can be evaluated")
    epoch = choice["selected_epoch"]
    if type(epoch) is not int or not 0 <= epoch <= 100 or choice["outcome"] != ("selected_epoch_zero" if epoch == 0 else "selected_trained"):
        raise ValueError("Invalid selected-epoch identity")
    if not 1 <= batch_size <= 256:
        raise ValueError("Unexpected evaluation batch size")
    if sha(checkpoint_record["path"]) != checkpoint_record["sha256"]:
        raise ValueError("The supplied selected checkpoint changed")
    row = cache_identity(cache, spec["domain"], spec["seed"], spec["arm"])
    row.update(candidate_id=spec["id"], selected_epoch=epoch, selection_outcome=choice["outcome"],
               residual_checkpoint=deepcopy(checkpoint_record))
    model = PairResidual(spec["arm"], cache.z.shape[-1], seed=spec["seed"]).to(dtype=cache.z.dtype).eval()
    declared_pairs = model.pairs.clone()
    state = torch.load(checkpoint_record["path"], weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.pairs, declared_pairs):
        raise ValueError("The checkpoint changes the declared latent-pair enumeration")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("A selected checkpoint contains invalid parameters")
    before = execution.tensor_copy(model.state_dict())
    failure = None
    try:
        parent_units, original_units = execution.predict(model, cache, batch_size)
        measured = metrics(cache.truth.numpy(), original_units.numpy())
    except (execution.NumericalFailure, EvaluationFailure) as error:
        failure = str(error)
    if not all(torch.equal(value, before[key]) for key, value in model.state_dict().items()):
        raise ValueError("Evaluation changed the selected checkpoint's state")
    cache.verify()
    if sha(checkpoint_record["path"]) != checkpoint_record["sha256"]:
        raise ValueError("The selected checkpoint file changed during evaluation")
    if failure is not None:
        row.update(status="failed_evaluation", metrics=None, reason=failure)
        return row, None
    if epoch == 0 and not torch.equal(parent_units, cache.f0):
        raise ValueError("A checkpoint labelled epoch zero changes the baseline prediction")
    row.update(status="evaluated", metrics=measured, reason=None)
    vectors = dict(ids=list(cache.ids), truth=cache.truth.detach().clone(),
                   prediction_parent_units=parent_units, prediction_original_units=original_units)
    return row, vectors


def complete_summary(values, expected):
    if type(expected) is not int or expected < 2:
        raise ValueError("A replicate summary requires a fixed count of at least two")
    observed = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("A reported replicate value is nonfinite")
    result = dict(expected_replicates=expected, available_replicates=len(observed), complete=len(observed) == expected,
                  mean=None, sample_sd=None, nominal_t_reference_interval95=None, minimum=None, maximum=None)
    if len(values) != expected:
        raise ValueError("Every planned replicate needs an explicit available or unavailable value")
    if not result["complete"]:
        return result
    mean = math.fsum(value/expected for value in observed)
    sd = math.hypot(*(value-mean for value in observed))/math.sqrt(expected-1)
    half = float(student_t.ppf(.975, expected-1))*sd/math.sqrt(expected)
    endpoints = [mean-half, mean+half]
    if not all(math.isfinite(value) for value in [mean, sd]+endpoints):
        raise ValueError("Summary arithmetic exceeded finite precision; preserve the per-replicate record")
    result.update(mean=mean, sample_sd=sd, nominal_t_reference_interval95=endpoints,
                  minimum=min(observed), maximum=max(observed))
    return result


def summarize(outcomes):
    expected = {(domain, seed, procedure) for domain, seeds in selection.DOMAINS.items()
                for seed in seeds for procedure in PROCEDURES}
    by_key = {(r["domain"], r["seed"], r["procedure"]): r for r in outcomes}
    if len(outcomes) != 60 or len(by_key) != 60 or set(by_key) != expected:
        raise ValueError("All 60 prespecified outcome rows are required exactly once")
    for row in outcomes:
        identity(row["domain"], row["seed"], row["procedure"])
        if row["status"] not in ("evaluated",)+UNAVAILABLE:
            raise ValueError("An unscored or unknown outcome is not a terminal evaluation")
        if row["status"] != "missing_parent":
            digest = row["parent_checkpoint_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("A nonmissing parent needs its actual checkpoint identity")
        elif row["parent_checkpoint_sha256"] is not None:
            raise ValueError("A missing parent cannot carry a replacement checkpoint identity")
        if row["status"] == "evaluated":
            measured = row["metrics"]
            if set(measured) != {"mse", "rmse", "mae"} or any(not math.isfinite(v) or v < 0 for v in measured.values()):
                raise ValueError("Invalid reported metrics")
            if not math.isclose(measured["rmse"], math.sqrt(measured["mse"]), rel_tol=1e-12, abs_tol=1e-14) or measured["mae"] > measured["rmse"]+1e-12:
                raise ValueError("Inconsistent squared, rooted or absolute error")
            if type(row["record_count"]) is not int or row["record_count"] < 1 or not row["cache_digest"]:
                raise ValueError("Evaluated outcomes need aligned cache provenance")
        elif row["metrics"] is not None or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("An unavailable outcome cannot carry selected metrics or omit its reason")
        if row["status"] == "no_valid_residual" and row["procedure"] == "baseline":
            raise ValueError("An unchanged baseline is not a residual selection")
    for domain, seeds in selection.DOMAINS.items():
        for seed in seeds:
            group = [by_key[domain, seed, procedure] for procedure in PROCEDURES]
            missing = [r["status"] == "missing_parent" for r in group]
            if any(missing) and not all(missing):
                raise ValueError("A missing parent affects all four procedures")
            if not all(missing):
                if len({r["parent_checkpoint_sha256"] for r in group}) != 1:
                    raise ValueError("Paired procedures have different frozen parents")
                cached = [r for r in group if r["status"] in ("evaluated", "failed_evaluation")]
                identities = {(r["cache_digest"], r["record_count"], r["target_mean"], r["target_sd"], r["record_ids_sha256"], r["targets_sha256"]) for r in cached}
                if len(identities) > 1:
                    raise ValueError("Paired procedures have different record sets, targets, caches or units")
    molecular = [by_key["ligand_contact", seed, "baseline"] for seed in selection.DOMAINS["ligand_contact"]
                 if by_key["ligand_contact", seed, "baseline"]["status"] != "missing_parent"]
    molecular_data = {(r["record_count"], r["record_ids_sha256"], r["targets_sha256"], r["target_mean"], r["target_sd"]) for r in molecular}
    if len(molecular_data) > 1:
        raise ValueError("The five molecular seeds must evaluate one fixed cohort and target transformation")
    procedures, contrasts = [], []
    for domain, seeds in selection.DOMAINS.items():
        metric = PRIMARY[domain]
        for procedure in PROCEDURES:
            rows = [by_key[domain, seed, procedure] for seed in seeds]
            values = [r["metrics"][metric] if r["status"] == "evaluated" else None for r in rows]
            procedures.append(dict(domain=domain, procedure=procedure, metric=metric,
                                   replicates=[dict(seed=seed, value=value, status=r["status"]) for seed, value, r in zip(seeds, values, rows)],
                                   **complete_summary(values, len(seeds))))
        for comparator in ("baseline", "additive", "pair_mlp"):
            rows, values = [], []
            for seed in seeds:
                left, right = by_key[domain, seed, "product"], by_key[domain, seed, comparator]
                value = left["metrics"][metric]-right["metrics"][metric] if left["status"] == right["status"] == "evaluated" else None
                rows.append(dict(seed=seed, difference=value, product_status=left["status"], comparator_status=right["status"]))
                values.append(value)
            available = [value for value in values if value is not None]
            contrasts.append(dict(domain=domain, contrast="product_minus_"+comparator, metric=metric,
                                  replicates=rows, lower=sum(value < 0 for value in available), tied=sum(value == 0 for value in available),
                                  higher=sum(value > 0 for value in available), sign_count_scope="available paired values; see completeness before interpretation",
                                  **complete_summary(values, len(seeds))))
    return dict(outcome_count=60, procedure_summaries=procedures, primary_contrasts=contrasts,
                outcomes=[deepcopy(by_key[key]) for key in sorted(expected)],
                interval_scope="Nominal Student-t reference intervals over fixed seed/partition repetitions; descriptive, without post-selection or population-coverage claims",
                primary_missing_policy="No aggregate mean/SD/interval unless all required paired replicates are available")
