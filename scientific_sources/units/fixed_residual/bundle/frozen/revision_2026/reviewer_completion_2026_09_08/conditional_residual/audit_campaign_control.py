"""Discarded audit of the full residual control path; no scientific data."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

import campaign_control as control
import campaign_lifecycle as life
import execution
import scoring
import selection

HERE = Path(__file__).resolve().parent
PRODUCER_CALLS = 0


def require_failure(call, expected=(ValueError, FileNotFoundError)):
    try:
        call()
    except expected as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("The intentionally invalid operation was accepted")


def cache_blob(cache):
    return dict(split=cache.split, ids=list(cache.ids), f0=cache.f0, z=cache.z, truth=cache.truth,
                target_mean=cache.target_mean, target_sd=cache.target_sd,
                parent_checkpoint_sha256=cache.parent_checkpoint_sha256, content_digest=cache.digest,
                scope="manufactured discarded campaign inputs")


def store_cache(path, cache):
    life.atomic_tensor(path, cache_blob(cache), immutable=True)
    return life.artifact(path) | dict(content_digest=cache.digest)


def parent_fixture(root, domain, seed, *, missing=False):
    directory = root/"original_fixture_studies"/f"{domain}_{seed}"
    candidate_ids = [f"{domain}_lma1_seed{seed}_lr{i}" for i in range(2)]
    candidate_sources, checkpoint_sources = [], []
    for index, identifier in enumerate(candidate_ids):
        checkpoint_path = directory/"checkpoints"/(identifier+".pt")
        life.atomic_tensor(checkpoint_path, dict(scope="manufactured parent marker, not a trained network", domain=domain, seed=seed, rate_index=index), immutable=True)
        checkpoint_sources.append(life.artifact(checkpoint_path))
        spec = dict(id=identifier, head="lma1", seed=seed, lr=selection.RATES[index], lr_index=index)
        spec["task" if domain == "cubic" else "setting"] = domain
        status, value = ("failed", None) if missing else ("valid", float(2-index))
        if domain == "cubic":
            record = dict(spec=spec, status=status, best_validation_loss=value, best_epoch=None if missing else 5,
                          checkpoint=str(checkpoint_path.relative_to(directory)), checkpoint_sha256=life.sha(checkpoint_path))
        else:
            record = spec | dict(status=status, best_validation_rmse=value, best_epoch=None if missing else 5,
                                 artifacts=[dict(path=str(checkpoint_path.relative_to(directory)), sha256=life.sha(checkpoint_path))])
        path = directory/"candidates"/(identifier+".json")
        life.atomic_json(path, record, immutable=True)
        candidate_sources.append(life.artifact(path))
    displayed_ids = candidate_ids[::-1] if domain == "cubic" and seed in (102,105,106,108,109) else candidate_ids
    choice = dict(head="lma1", seed=seed, candidate_ids=displayed_ids,
                  selected_id=None if missing else candidate_ids[1])
    choice["task" if domain == "cubic" else "setting"] = domain
    choice["validation_loss" if domain == "cubic" else "validation_rmse"] = None if missing else 1.
    if domain == "cubic":
        choice["checkpoint_sha256"] = None if missing else checkpoint_sources[1]["sha256"]
    lock_path = directory/"selection_lock.json"
    life.atomic_json(lock_path, dict(scope="manufactured original parent selection", selections=[choice],
        candidate_records=[dict(path=str(Path(s["path"]).relative_to(directory)), sha256=s["sha256"]) for s in candidate_sources]), immutable=True)
    row = dict(domain=domain, seed=seed, parent_selection=life.artifact(lock_path), parent_candidates=candidate_sources,
               selected_id=choice["selected_id"], status="missing_parent" if missing else "available",
               reason="Both manufactured original parent rates failed" if missing else None,
               parent_checkpoint=None if missing else checkpoint_sources[1], caches={}, native_cache_audit=None)
    if missing:
        return row
    d = 12 if domain == "cubic" else 8
    mean, sd = (0., 1.) if domain == "cubic" else (6.5, 1.2)
    generator = torch.Generator().manual_seed(106000+seed)
    z = torch.randn(1, 8, d, generator=generator)
    for split, count in (("train", 27), ("val", 17)):
        prefix = f"fixture:{domain}:{seed}:" if domain == "cubic" else f"fixture:{domain}:"
        ids = [prefix+f"{split}:{i}" for i in range(count)]
        target = mean if domain == "cubic" and seed == 100 and split == "val" else mean+sd
        cache = execution.make_cache(split, ids, torch.zeros(count), z.expand(count,-1,-1).clone(),
                                     torch.full((count,), target, dtype=torch.float64), mean, sd, checkpoint_sources[1]["sha256"])
        row["caches"][split] = store_cache(directory/"caches"/(split+".pt"), cache)
    audit_path = directory/"native_cache_audit.json"
    life.atomic_json(audit_path, dict(passed=True, scope="manufactured cache contract, not a real native predictor audit",
        domain=domain, seed=seed, parent_checkpoint_sha256=checkpoint_sources[1]["sha256"],
        cache_content_digests={s:r["content_digest"] for s,r in row["caches"].items()},
        sources=[life.artifact(Path(__file__))]+list(row["caches"].values())), immutable=True)
    row["native_cache_audit"] = life.artifact(audit_path)
    return row


def fixture_producer(root, access_record, bundle_record):
    global PRODUCER_CALLS
    PRODUCER_CALLS += 1
    bundle = life.read(bundle_record["path"])
    rows, sources = [], [life.artifact(Path(__file__))]
    for parent in bundle["parents"]:
        domain, seed = parent["domain"], parent["seed"]
        row = dict(domain=domain, seed=seed, status=parent["status"])
        if parent["status"] == "missing_parent":
            row.update(cache=None, reason=parent["reason"])
        else:
            train = life.load_cache(parent["caches"]["train"], "train", parent["parent_checkpoint"]["sha256"])
            generator = torch.Generator().manual_seed(107000+seed)
            n, d = 19, train.z.shape[-1]
            ids = [f"fixture:{domain}:"+(f"{seed}:" if domain == "cubic" else "")+f"test:{i}" for i in range(n)]
            truth = train.target_mean+train.target_sd*(.5+torch.arange(n, dtype=torch.float64)/50)
            cache = execution.make_cache("test", ids, torch.zeros(n), torch.randn(n,8,d,generator=generator), truth,
                                         train.target_mean, train.target_sd, train.parent_checkpoint_sha256)
            path = root/"test_cache_fixtures"/f"{domain}_{seed}.pt"
            record = store_cache(path, cache)
            row.update(cache=record, parent_checkpoint=parent["parent_checkpoint"], native_prediction_and_bucket_equal=True,
                       equality_scope="Manufactured immutable tensor identity, not real parent evaluation")
            sources.append(record)
        rows.append(row)
    path = root/"test_cache_manifest.json"
    life.atomic_json(path, dict(schema=1, scope=control.DISCARDED, test_access=access_record, parent_bundle=bundle_record,
        native_cache_audit_passed=True, producer_sources=bundle["producer_sources"], sources=sources, parents=rows), immutable=True)
    return path


def mutable_trial(path, mutate, call, expected=(ValueError, FileNotFoundError)):
    before = path.read_bytes()
    try:
        value = life.read(path)
        mutate(value)
        life.atomic_json(path, value)
        return require_failure(call, expected)
    finally:
        path.write_bytes(before)


def main():
    global PRODUCER_CALLS
    receipt_path = HERE/"campaign_control_cpu_audit_v2.json"
    root = HERE/"discarded_campaign_audit_20260909_v2"
    if root.exists() or receipt_path.exists():
        raise FileExistsError("Preserve prior campaign-control work")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    root.mkdir()
    parents = [parent_fixture(root, domain, seed, missing=(domain == "ligand_contact" and seed == 46))
               for domain, seed in control.REPLICATES]
    bundle_path = root/"parent_bundle.json"
    life.atomic_json(bundle_path, dict(schema=1, scope=control.DISCARDED,
        producer_sources=[life.artifact(Path(__file__))], parents=parents), immutable=True)
    campaign = root/"campaign"
    checks = []
    blocked = require_failure(lambda: control.create_context(HERE/"retained_study", bundle_path, scope=control.RETAINED))
    assert not (HERE/"retained_study/study_context.json").exists()
    checks.append(dict(name="retained activation rejects missing decision and prerequisite records", rejection=blocked))
    control.create_context(campaign, bundle_path)
    assert control.verify_context(campaign)["scope"] == control.DISCARDED
    reversed_order = [r for r in parents if r["domain"] == "cubic" and life.read(r["parent_selection"]["path"])["selections"][0]["candidate_ids"][0].endswith("lr1")]
    assert len(reversed_order) == 5 and all(control.original_parent_choice(r) == r["selected_id"] for r in reversed_order)
    checks.append(dict(name="historical parent candidate display order does not alter validation selection", reverse_order_parents=5))
    early = require_failure(lambda: control.prepare_test_caches(campaign, fixture_producer))
    assert PRODUCER_CALLS == 0 and not (campaign/"test_access.json").exists()
    checks.append(dict(name="test producer is not invoked before all candidates and common selection", rejection=early))

    current = next(s for s in selection.candidate_plan() if (s["domain"],s["seed"],s["arm"],s["rate_index"]) == ("cubic",100,"product",0))
    assert life.fit_candidate(campaign, current, pause_after_epochs=1) is None
    original_epoch = execution.train_epoch
    def with_failure(runtime, train, val):
        spec = runtime["spec"]
        if (spec["domain"],spec["seed"],spec["arm"]) == ("cubic",101,"pair_mlp"):
            raise execution.NumericalFailure("Injected discarded two-rate residual failure")
        return original_epoch(runtime, train, val)
    execution.train_epoch = with_failure
    try:
        records = control.fit_all(campaign)
    finally:
        execution.train_epoch = original_epoch
    counts = {status:sum(r["status"] == status for r in records) for status in ("valid","failed_numerical","missing_parent")}
    assert len(records) == 90 and counts == dict(valid=82, failed_numerical=2, missing_parent=6)
    checks.append(dict(name="complete 90-candidate accounting with numerical failure and missing parent", counts=counts))
    reference = root/"reference"
    control.create_context(reference, bundle_path)
    control.bind_all(reference)
    life.fit_candidate(reference, current)
    def restored(location):
        _, train, val = life.read_binding(location, current)
        runtime, _ = life.load_snapshot(life.candidate_folder(location,current), train, val)
        return execution.export_runtime(runtime)
    assert life.same(restored(campaign), restored(reference))
    assert len(life.attempt_sources(life.candidate_folder(campaign,current))) == 2
    checks.append(dict(name="campaign context preserves exact complete-epoch continuation", exact_all_runtime_state=True))
    protected = [p for p in (campaign/"candidates").rglob("*") if p.is_file() and p.name != "active_process.lock"]
    before = {str(p):life.sha(p) for p in protected}
    again = control.fit_all(campaign)
    assert again == records and before == {str(p):life.sha(p) for p in protected}
    checks.append(dict(name="completed campaign restart makes no candidate update or new attempt", preserved_files=len(before)))
    late = require_failure(lambda: control.prepare_test_caches(campaign, fixture_producer))
    assert PRODUCER_CALLS == 0
    checks.append(dict(name="all terminal candidates alone do not open the test producer", rejection=late))
    locked = control.lock_selection(campaign)
    assert len(locked["choices"]) == 45 and locked["test_scoring_allowed"] is False
    choices = locked["choices"]
    assert sum(r["outcome"] == "missing_parent" for r in choices) == 3
    assert sum(r["outcome"] == "no_valid_residual" for r in choices) == 1
    assert sum(r["outcome"] == "selected_epoch_zero" for r in choices) >= 3
    permit = control.test_access(campaign)
    assert permit["scientific_test_access"] is False
    checks.append(dict(name="common 45-choice lock preserves epoch zero and missing choices", epoch_zero_choices=sum(r["outcome"] == "selected_epoch_zero" for r in choices)))
    frozen = require_failure(lambda: control.prepare_test_caches(campaign, len))
    assert PRODUCER_CALLS == 0
    checks.append(dict(name="unfrozen test-cache callable is rejected before invocation", rejection=frozen))
    manifest_path = control.prepare_test_caches(campaign, fixture_producer)
    assert PRODUCER_CALLS == 1
    result = control.score_all(campaign, manifest_path)
    summary = result["summary"]
    assert result["complete"] and result["scientific_evaluation"] is False
    outcomes = [life.read(s["path"])["outcome"] for s in result["outcome_files"]]
    outcome_counts = {status:sum(r["status"] == status for r in outcomes) for status in ("evaluated","missing_parent","no_valid_residual")}
    assert len(outcomes) == 60 and outcome_counts == dict(evaluated=55, missing_parent=4, no_valid_residual=1)
    max_delta = 0.
    for source in result["outcome_files"]:
        entry = life.read(source["path"])
        if entry["vectors"] is None:
            continue
        vectors = torch.load(entry["vectors"]["path"], weights_only=True, map_location="cpu")
        truth, prediction = vectors["truth"].tolist(), vectors["prediction_original_units"].tolist()
        errors = [p-y for p,y in zip(prediction,truth)]
        mse = math.fsum(x*x for x in errors)/len(errors)
        mae = math.fsum(abs(x) for x in errors)/len(errors)
        independent = dict(mse=mse, rmse=math.sqrt(mse), mae=mae)
        max_delta = max(max_delta,*(abs(independent[k]-entry["outcome"]["metrics"][k]) for k in independent))
    assert max_delta < 1e-12
    checks.append(dict(name="all 60 original-unit outcomes reconcile with independent scalar metrics", counts=outcome_counts, maximum_metric_difference=max_delta))
    saved = {s["path"]:life.sha(s["path"]) for s in result["outcome_files"]}
    saved[str(campaign/"evaluation.json")] = life.sha(campaign/"evaluation.json")
    assert control.score_all(campaign, manifest_path) == result
    assert saved == {p:life.sha(p) for p in saved} and before == {str(p):life.sha(p) for p in protected}
    checks.append(dict(name="evaluation restart reproduces selected predictions and preserves all files", unchanged_files=len(saved)))

    context_path = campaign/"study_context.json"
    rejected = mutable_trial(context_path, lambda x:x["sources"].pop(), lambda:control.verify_context(campaign))
    checks.append(dict(name="complete source inventory is required", rejection=rejected))
    selection_path = campaign/"selection_lock.json"
    def wrong_choice(x):
        selected = next(r for r in x["choices"] if r["selected_id"] is not None)
        selected["selected_id"] = next(i for i in selected["candidate_ids"] if i != selected["selected_id"])
    rejected = mutable_trial(selection_path, wrong_choice, lambda:control.test_access(campaign))
    checks.append(dict(name="selection cannot change after the common gate", rejection=rejected))
    parent = next(r for r in parents if r["domain"] == "cubic" and r["seed"] == 102)
    original_lock = Path(parent["parent_selection"]["path"])
    def parent_wrong_rate(x):
        x["selections"][0]["selected_id"] = next(i for i in x["selections"][0]["candidate_ids"] if i.endswith("lr0"))
        x["selections"][0]["validation_loss"] = 2.
    rejected = mutable_trial(original_lock, parent_wrong_rate, lambda:control.test_access(campaign))
    checks.append(dict(name="changed original parent selection blocks test access", rejection=rejected))
    original = original_lock.read_bytes()
    try:
        bad = life.read(original_lock); parent_wrong_rate(bad); life.atomic_json(original_lock,bad)
        forged = deepcopy(parent); forged["parent_selection"] = life.artifact(original_lock)
        rejected = require_failure(lambda:control.original_parent_choice(forged))
    finally:
        original_lock.write_bytes(original)
    checks.append(dict(name="independent parent validation rule rejects a refreshed hash with a worse rate", rejection=rejected))

    victim = life.candidate_folder(campaign,current)/"result.json"
    assert victim.resolve().is_relative_to(root.resolve())
    original = victim.read_bytes()
    try:
        victim.unlink()
        rejected = require_failure(lambda:control.test_access(campaign))
    finally:
        victim.write_bytes(original)
    checks.append(dict(name="a missing terminal candidate invalidates an existing access record", rejection=rejected))

    manifest = life.read(manifest_path)
    bad = deepcopy(manifest)
    row = next(r for r in bad["parents"] if r["domain"] == "cubic" and r["seed"] == 102)
    blob = torch.load(row["cache"]["path"], weights_only=True, map_location="cpu")
    train = life.load_cache(parent["caches"]["train"], "train", parent["parent_checkpoint"]["sha256"])
    ids = list(blob["ids"]); ids[0] = train.ids[0]
    altered = execution.make_cache("test", ids, blob["f0"], blob["z"], blob["truth"], blob["target_mean"], blob["target_sd"], blob["parent_checkpoint_sha256"])
    row["cache"] = store_cache(root/"invalid_test_overlap.pt", altered)
    path = root/"invalid_test_overlap.json"; life.atomic_json(path,bad,immutable=True)
    rejected = require_failure(lambda:control.validate_test_manifest(campaign,path))
    checks.append(dict(name="test ID overlap is rejected with internally consistent new file hashes", rejection=rejected))
    bad = deepcopy(manifest); bad["test_access"] = life.artifact(bundle_path)
    path = root/"invalid_test_access.json"; life.atomic_json(path,bad,immutable=True)
    rejected = require_failure(lambda:control.validate_test_manifest(campaign,path))
    checks.append(dict(name="test cache must cite the actual common access record", rejection=rejected))

    entry_path = next(Path(s["path"]) for s in result["outcome_files"] if life.read(s["path"])["outcome"]["selection_outcome"] == "selected_trained")
    entry = life.read(entry_path)
    vector_path = Path(entry["vectors"]["path"])
    entry_bytes, vector_bytes = entry_path.read_bytes(), vector_path.read_bytes()
    try:
        vectors = torch.load(vector_path,weights_only=True,map_location="cpu")
        vectors["prediction_parent_units"] += 17.
        vectors["prediction_original_units"] = vectors["prediction_parent_units"].double()*entry["outcome"]["target_sd"]+entry["outcome"]["target_mean"]
        life.atomic_tensor(vector_path,vectors)
        entry["vectors"] = life.artifact(vector_path)
        entry["outcome"]["metrics"] = scoring.metrics(vectors["truth"].numpy(),vectors["prediction_original_units"].numpy())
        life.atomic_json(entry_path,entry)
        rejected = require_failure(lambda:control.score_all(campaign,manifest_path))
    finally:
        entry_path.write_bytes(entry_bytes); vector_path.write_bytes(vector_bytes)
    checks.append(dict(name="refreshed vector hash and recomputed metrics cannot replace the selected predictor", rejection=rejected))

    decision = dict(decision="activate_conditional_residual", reason="Manufactured activation-contract audit only", recorded_utc=life.utc(),
                    outcome_informed=True, tests_reused=True, candidate_plan=selection.candidate_plan())
    evidence = {}
    for role in control.PREFLIGHT_ROLES:
        path = root/"manufactured_preflight"/(role+".json")
        body = dict(scope="manufactured metadata, not an actual prerequisite completion", sources=control.source_inventory(life.read(bundle_path)))
        body["complete" if role.endswith("profiles") else "passed"] = True
        life.atomic_json(path,body,immutable=True); evidence[role] = life.artifact(path)
    preflight = dict(passed=True, parent_bundle=life.artifact(bundle_path), evidence=evidence, sources=control.source_inventory(life.read(bundle_path)))
    control.validate_activation(decision,preflight,bundle_path)
    bad = deepcopy(preflight); bad["evidence"].pop("matched_final_audit")
    rejected = require_failure(lambda:control.validate_activation(decision,bad,bundle_path))
    checks.append(dict(name="activation metadata requires every preceding-study and residual audit role", rejection=rejected,
                       positive_path_scope="Manufactured metadata contract only; no retained context created"))
    bad_decision = deepcopy(decision); bad_decision["tests_reused"] = False
    rejected = require_failure(lambda:control.validate_activation(bad_decision,preflight,bundle_path))
    checks.append(dict(name="activation cannot relabel reused evaluation as untouched confirmation", rejection=rejected))

    assert PRODUCER_CALLS == 1 and not torch.cuda.is_initialized()
    assert before == {str(p):life.sha(p) for p in protected}
    assert saved == {p:life.sha(p) for p in saved}
    assert len(summary["primary_contrasts"]) == 6
    assert [r["complete"] for r in summary["primary_contrasts"] if r["domain"] == "cubic"] == [True, True, False]
    assert all(not r["complete"] and r["mean"] is None for r in summary["primary_contrasts"] if r["domain"] == "ligand_contact")
    receipt = dict(passed=True, completed_utc=life.utc(), scope="Discarded end-to-end campaign control on manufactured data",
        groups=len(checks), checks=checks, candidates=90, selections=45, outcome_rows=60, contrasts=6,
        candidate_counts=counts, outcome_counts=outcome_counts, maximum_independent_metric_difference=max_delta,
        test_producer_calls=PRODUCER_CALLS, retained_campaign_created=False, retained_residual_fits=0,
        new_scientific_test_predictions=False, cuda_initialized=False,
        sources=control.source_inventory(life.read(bundle_path))+[life.artifact(Path(__file__))],
        artifacts=[life.artifact(campaign/name) for name in ("study_context.json","selection_lock.json","test_access.json","test_cache_manifest.json","evaluation.json")],
        remaining="Actual parent bundle/native cache adapters, native complete-predictor cost adapter, actual preceding-study completion and any retained activation/evaluation")
    life.atomic_json(receipt_path,receipt,immutable=True)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("checks","sources","artifacts")}),flush=True)


if __name__ == "__main__":
    main()
