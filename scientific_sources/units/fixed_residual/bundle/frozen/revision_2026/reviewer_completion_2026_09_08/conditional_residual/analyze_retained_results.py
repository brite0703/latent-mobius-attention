"""Independent reconciliation of the fixed residual study's retained records.

This reporting program does not import the training, selection, scoring or cost
kernels, fit models, generate predictions, or modify campaign artifacts. The
selection mode reads fitting/validation records only. Complete reporting waits
for the entire prediction and cost pipeline to finish.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
STUDY = HERE / "retained_study"
OUT = HERE / "retained_analysis"
DOMAINS = {"cubic": tuple(range(100, 110)), "ligand_contact": tuple(range(42, 47))}
ARMS = ("product", "additive", "pair_mlp")
PROCEDURES = ("baseline",) + ARMS
METRIC = {"cubic": "mse", "ligand_contact": "rmse"}
T_CRITICAL = {5: 2.7764451051977987, 10: 2.2621571628540993}
SEEN = {}
DELTAS = defaultdict(list)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bytes_checked(record):
    path = Path(record["path"]).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == record["sha256"], "Changed artifact: " + str(path))
    require(str(path) not in SEEN or SEEN[str(path)] == digest, "Conflicting source identity")
    SEEN[str(path)] = digest
    return data


def source(path):
    path = Path(path).resolve()
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    bytes_checked(record)
    return record


def checked_json(record):
    return json.loads(bytes_checked(record).decode("utf-8-sig"))


def checked_tensor(record):
    return torch.load(io.BytesIO(bytes_checked(record)), weights_only=True, map_location="cpu")


def near(left, right, label):
    require(math.isfinite(left) and math.isfinite(right), "Nonfinite " + label)
    delta = abs(left - right)
    require(delta <= 1e-12 + 1e-11 * abs(right), "Arithmetic mismatch: " + label)
    DELTAS[label].append(delta)


def metrics(truth, prediction):
    require(len(truth) == len(prediction) > 0, "Unaligned scalar vectors")
    differences = [float(p) - float(y) for y, p in zip(truth, prediction)]
    require(all(math.isfinite(v) for v in differences), "Nonfinite scalar difference")
    mse = math.fsum(v * v for v in differences) / len(differences)
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": math.fsum(map(abs, differences)) / len(differences)}


def summary(values, expected):
    require(len(values) == expected, "Missing planned replicate slot")
    available = [x for x in values if x is not None]
    result = dict(expected_replicates=expected, available_replicates=len(available), complete=len(available) == expected,
                  mean=None, sample_sd=None, nominal_t_reference_interval95=None, minimum=None, maximum=None)
    if len(available) == expected:
        mean, sd = statistics.mean(available), statistics.stdev(available)
        half = T_CRITICAL[expected] * sd / math.sqrt(expected)
        result.update(mean=mean, sample_sd=sd, nominal_t_reference_interval95=[mean-half, mean+half],
                      minimum=min(available), maximum=max(available))
    return result


def compare_summary(calculated, retained):
    for name, value in calculated.items():
        if value is None or isinstance(value, (int, bool)):
            require(retained[name] == value, "Summary coverage differs: " + name)
        elif isinstance(value, list):
            for left, right in zip(value, retained[name]):
                near(left, right, "summary_" + name)
        else:
            near(value, retained[name], "summary_" + name)


def table(path, rows):
    require(bool(rows), "Refuse an empty reported table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()})
    return path, stream.getvalue()


def reconcile_selection(context):
    locked = checked_json(source(STUDY / "selection_lock.json"))
    require(locked["study_context"] == source(STUDY / "study_context.json"), "Selection/context mismatch")
    require(locked["candidate_count"] == 90 and locked["selection_count"] == 45, "Changed finite allocation")
    expected = {(domain, seed, arm, rate) for domain, seeds in DOMAINS.items()
                for seed in seeds for arm in ARMS for rate in range(2)}
    planned = {(r["domain"], r["seed"], r["arm"], r["rate_index"]) for r in context["candidate_plan"]}
    require(len(context["candidate_plan"]) == 90 and planned == expected, "Changed declared candidate plan")
    require(len(locked["candidates"]) == 90, "Incomplete candidate inventory")
    records, candidate_rows, attempts = {}, [], []
    for artifact in locked["candidates"]:
        row = checked_json(artifact)
        spec = row["spec"]
        key = (spec["domain"], spec["seed"], spec["arm"], spec["rate_index"])
        require(key in expected and key not in records, "Duplicate or unexpected candidate")
        require(spec["rate"] == (.0003, .001)[spec["rate_index"]], "Changed learning rate")
        records[key] = row
        flat = dict(spec, status=row["status"], best_epoch=row.get("best_epoch"),
                    validation_mse=row.get("best_validation_mse"), epoch_zero_validation_mse=row.get("epoch_zero_validation_mse"),
                    completed_epochs=row.get("completed_epochs"), optimizer_updates=row.get("optimizer_updates"),
                    termination=row.get("termination"))
        candidate_rows.append(flat)
        require(row["status"] in ("valid", "failed_numerical", "missing_parent"), "Nonterminal candidate")
        if row["status"] == "missing_parent":
            continue
        binding = checked_json(row["binding"])
        require(binding["spec"] == spec, "Candidate binding differs")
        runtime = checked_tensor(row["continuation_state"])
        for artifact in row["attempts"]:
            attempt = checked_json(artifact)
            attempts.append(dict(candidate_id=spec["id"], **attempt))
        if row["status"] != "valid":
            continue
        require(runtime["spec"] == spec, "Continuation belongs to another candidate")
        earliest = min(runtime["history"], key=lambda item: (item["validation_mse"], item["epoch"]))
        require((earliest["validation_mse"], earliest["epoch"]) == (row["best_validation_mse"], row["best_epoch"]),
                "Selected epoch differs from earliest validation minimum")
        require(runtime["history"][0]["epoch"] == 0 and row["best_validation_mse"] <= row["epoch_zero_validation_mse"],
                "Epoch-zero option not retained")
        require([x["epoch"] for x in runtime["epoch_rows"]] == list(range(1, row["completed_epochs"]+1)), "Incomplete epoch history")
        expected_updates = row["completed_epochs"] * (8 if spec["domain"] == "cubic" else 5)
        require(row["optimizer_updates"] == runtime["updates"] == expected_updates, "Incorrect final-batch update count")
        selected_state = checked_tensor(row["artifacts"][0])
        require(selected_state.keys() == runtime["best_state"].keys() and
                all(torch.equal(value, runtime["best_state"][key]) for key, value in selected_state.items()),
                "Retained checkpoint differs from selected state")
        vector = checked_tensor(row["artifacts"][1])
        val = checked_tensor(binding["caches"]["val"])
        require(vector["ids"] == val["ids"] and torch.equal(vector["truth"], val["truth"]), "Changed validation records/targets")
        restored = vector["prediction_parent_units"].double() * val["target_sd"] + val["target_mean"]
        require(torch.equal(restored, vector["prediction_original_units"]), "Validation inverse transform differs")
        measured = metrics(vector["truth"].tolist(), vector["prediction_original_units"].tolist())
        near(measured["mse"], row["best_validation_mse"], "validation_mse")
    require(set(records) == expected, "Incomplete candidate set")
    choices, choice_rows = {}, []
    for item in locked["choices"]:
        key = item["domain"], item["seed"], item["arm"]
        require(key not in choices, "Duplicate choice")
        pair = [records[key + (index,)] for index in range(2)]
        valid = [row for row in pair if row["status"] == "valid"]
        chosen = min(valid, key=lambda row: (row["best_validation_mse"], row["spec"]["rate_index"])) if valid else None
        require(item["candidate_ids"] == [row["spec"]["id"] for row in pair] and
                item["candidate_statuses"] == [row["status"] for row in pair], "Choice omits a prescribed rate")
        require(item["selected_id"] == (chosen["spec"]["id"] if chosen else None), "Choice differs from validation-only minimum")
        if chosen:
            require(item["selected_epoch"] == chosen["best_epoch"], "Selected epoch differs")
        choices[key] = item
        choice_rows.append(dict(item, selected_rate=chosen["spec"]["rate"] if chosen else None,
                                selected_validation_mse=chosen["best_validation_mse"] if chosen else None))
    require(set(choices) == {(d, s, a) for d, seeds in DOMAINS.items() for s in seeds for a in ARMS}, "Incomplete choice set")
    access = checked_json(source(STUDY / "test_access.json"))
    require(access["selection_lock"] == source(STUDY / "selection_lock.json"), "Test access cites another selection")
    require(access["selections"] == locked["choices"], "Test access changes selected models")
    latest_finish = max(datetime.fromisoformat(row["finished_utc"]) for row in records.values())
    require(latest_finish <= datetime.fromisoformat(locked["locked_utc"]) <= datetime.fromisoformat(access["granted_utc"]),
            "Recorded fitting/selection/access order differs")
    complete_attempts = [row for row in attempts if isinstance(row.get("elapsed_seconds"), (int, float))]
    report = dict(candidate_count=90, candidate_statuses=dict(Counter(row["status"] for row in records.values())),
                  selection_count=45, selection_outcomes=dict(Counter(row["outcome"] for row in choices.values())),
                  selected_epoch_counts=dict(Counter(str(row["selected_epoch"]) for row in choices.values())),
                  selected_rate_counts=dict(Counter(str(row["selected_rate"]) for row in choice_rows)),
                  recorded_last_fit_utc=latest_finish.isoformat(), locked_utc=locked["locked_utc"], test_access_utc=access["granted_utc"],
                  recorded_order_passed=True, attempted_fit_records=len(attempts), complete_attempt_durations=len(complete_attempts),
                  summed_recorded_attempt_seconds=math.fsum(row["elapsed_seconds"] for row in complete_attempts),
                  timing_scope="Recorded attempt durations include validation and file I/O; this is not complete pipeline elapsed time or isolated optimizer time.",
                  scope="Independent record, state, validation-vector arithmetic and selection reconciliation; no new inference or training.")
    return report, choice_rows, candidate_rows, choices


def reconcile_evaluation(choices):
    evaluation = checked_json(source(STUDY / "evaluation.json"))
    require(evaluation["complete"] is True and evaluation["scientific_evaluation"] is True, "Incomplete scientific evaluation")
    manifest = checked_json(evaluation["dependencies"]["test_manifest"])
    cache_rows = {(row["domain"], row["seed"]): row for row in manifest["parents"]}
    caches = {key: checked_tensor(row["cache"]) for key, row in cache_rows.items() if row["status"] == "available"}
    expected = {(d, s, p) for d, seeds in DOMAINS.items() for s in seeds for p in PROCEDURES}
    outcomes, metric_rows = {}, []
    for artifact in evaluation["outcome_files"]:
        saved = checked_json(artifact)
        row = saved["outcome"]
        key = row["domain"], row["seed"], row["procedure"]
        require(key in expected and key not in outcomes, "Duplicate or unexpected test outcome")
        require(saved["dependencies"] == evaluation["dependencies"], "Changed evaluation dependencies")
        outcomes[key] = row
        flat = {name: row[name] for name in ("domain", "seed", "procedure", "status", "selected_epoch", "selection_outcome", "candidate_id", "record_count")}
        if row["status"] != "evaluated":
            require(saved["vectors"] is None and row["metrics"] is None and row["reason"], "Unavailable outcome carries predictions")
            flat.update(mse=None, rmse=None, mae=None, reason=row["reason"])
            metric_rows.append(flat)
            continue
        cache = caches[key[:2]]
        vector = checked_tensor(saved["vectors"])
        require(vector["ids"] == cache["ids"] and torch.equal(vector["truth"], cache["truth"]), "Test record order or targets differ")
        require(len(vector["ids"]) == row["record_count"] == (1024 if key[0] == "cubic" else 366), "Changed test denominator")
        require(row["parent_checkpoint_sha256"] == cache["parent_checkpoint_sha256"] and
                (row["target_mean"], row["target_sd"]) == (cache["target_mean"], cache["target_sd"]), "Parent or units differ")
        restored = vector["prediction_parent_units"].double() * cache["target_sd"] + cache["target_mean"]
        require(torch.equal(restored, vector["prediction_original_units"]), "Test inverse transform differs")
        if key[2] == "baseline" or row["selected_epoch"] == 0:
            require(torch.equal(vector["prediction_parent_units"], cache["f0"]), "Baseline/epoch-zero prediction differs")
        if key[2] != "baseline":
            choice = choices[key]
            require((row["candidate_id"], row["selected_epoch"]) == (choice["selected_id"], choice["selected_epoch"]), "Test uses another choice")
        measured = metrics(vector["truth"].tolist(), vector["prediction_original_units"].tolist())
        for name, value in measured.items():
            near(value, row["metrics"][name], "test_" + name)
        flat.update(measured, reason=None)
        metric_rows.append(flat)
    require(set(outcomes) == expected and len(evaluation["summary"]["outcomes"]) == 60, "Incomplete test outcome set")
    for row in evaluation["summary"]["outcomes"]:
        require(row == outcomes[row["domain"], row["seed"], row["procedure"]], "Summary changes a per-seed outcome")
    procedures, contrasts, paired = [], [], []
    retained_p = {(r["domain"], r["procedure"]): r for r in evaluation["summary"]["procedure_summaries"]}
    retained_c = {(r["domain"], r["contrast"]): r for r in evaluation["summary"]["primary_contrasts"]}
    require(len(retained_p) == 8 and len(retained_c) == 6, "Changed analysis family")
    for domain, seeds in DOMAINS.items():
        metric = METRIC[domain]
        for procedure in PROCEDURES:
            values = [outcomes[domain, s, procedure]["metrics"][metric] if outcomes[domain, s, procedure]["status"] == "evaluated" else None for s in seeds]
            calculated = summary(values, len(seeds))
            compare_summary(calculated, retained_p[domain, procedure])
            procedures.append(dict(domain=domain, procedure=procedure, metric=metric, **calculated))
        for comparator in ("baseline", "additive", "pair_mlp"):
            values = []
            for seed in seeds:
                left, right = outcomes[domain, seed, "product"], outcomes[domain, seed, comparator]
                value = left["metrics"][metric] - right["metrics"][metric] if left["status"] == right["status"] == "evaluated" else None
                values.append(value)
                paired.append(dict(domain=domain, seed=seed, comparator=comparator, metric=metric, difference=value))
            calculated = summary(values, len(seeds))
            retained = retained_c[domain, "product_minus_" + comparator]
            compare_summary(calculated, retained)
            signs = dict(lower=sum(v < 0 for v in values if v is not None), tied=sum(v == 0 for v in values if v is not None),
                         higher=sum(v > 0 for v in values if v is not None))
            require(all(retained[k] == v for k, v in signs.items()), "Incorrect paired sign counts")
            require([r["difference"] for r in retained["replicates"]] == values, "Changed paired replicate values")
            contrasts.append(dict(domain=domain, comparator=comparator, metric=metric, **calculated, **signs))
    return dict(outcome_count=60, outcome_statuses=dict(Counter(r["status"] for r in outcomes.values())),
                procedure_summaries=procedures, paired_contrasts=contrasts,
                interval_scope=evaluation["summary"]["interval_scope"]), metric_rows, paired


def linear_quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1) * probability
    left = int(math.floor(position))
    right = min(left+1, len(ordered)-1)
    return ordered[left] + (ordered[right]-ordered[left]) * (position-left)


def reconcile_costs():
    profiles = checked_json(source(STUDY / "profiles.json"))
    require(profiles["complete"] is True and profiles["prescribed_outcomes"] == 435, "Incomplete cost stage")
    expected = set()
    for domain, seeds in DOMAINS.items():
        for seed in seeds:
            for procedure in ("unchanged",) + ARMS:
                for batch in (1, 32, 256):
                    expected.add((domain, seed, procedure, "complete", "complete_inference", batch))
            for arm in ARMS:
                for batch in (1, 32, 256):
                    expected.add((domain, seed, arm, "cached", "inference", batch))
                for batch in ((256,) if domain == "cubic" else (256, 72)):
                    for mode in ("fresh_optimizer_step", "steady_optimizer_step"):
                        expected.add((domain, seed, arm, "cached", mode, batch))
    rows, observed = [], set()
    for artifact in profiles["artifacts"]:
        record = checked_json(artifact)
        allocation = record["allocation"]
        work = allocation["workload"]
        key = allocation["domain"], allocation["seed"], allocation["procedure"], allocation["family"], work["mode"], work["batch_size"]
        require(key in expected and key not in observed, "Unexpected or repeated cost outcome")
        observed.add(key)
        require(record["dependencies"] == profiles["dependencies"], "Cost dependencies differ")
        flat = dict(zip(("domain", "seed", "procedure", "family", "mode", "batch_size"), key), status=record["status"], reason=record["reason"])
        if record["status"] != "profiled":
            require(record["result"] is None and record["reason"], "Unavailable cost carries a timing")
            rows.append(flat)
            continue
        result = record["result"]
        require(result["workload"] == work and result["warmups"] == 5, "Changed timing workload or warm-ups")
        times = result["repetitions_ms"]
        require(len(times) == 30 and all(math.isfinite(v) and v >= 0 for v in times), "Invalid timing repetitions")
        stats = dict(median_ms=statistics.median(times), p10_ms=linear_quantile(times, .1), p90_ms=linear_quantile(times, .9),
                     minimum_ms=min(times), maximum_ms=max(times), mean_ms=statistics.mean(times), sample_sd_ms=statistics.stdev(times))
        require(result["summary"]["repetitions"] == 30, "Changed timing repetition count")
        for name, value in stats.items():
            near(value, result["summary"][name], "cost_" + name)
        residual_parameters = result["parameters"]["residual"] if key[3] == "complete" else result["parameter_count"]
        d = 12 if key[0] == "cubic" else 8
        expected_parameters = 0 if key[2] == "unchanged" else (8*d*d+12*d+1 if key[2] != "pair_mlp" else 8*d*d+11*d+1)
        require(residual_parameters == expected_parameters, "Residual parameter count differs")
        flat.update(stats, residual_parameters=residual_parameters,
                    parent_parameters=result["parameters"]["parent"] if key[3] == "complete" else None,
                    repetitions_ms=times, tensor_storage=result["tensor_storage"], measurement_scope=result["measurement_scope"])
        rows.append(flat)
    require(observed == expected and len(profiles["artifacts"]) == 435, "Incomplete cost allocation")
    require(sum(r["family"] == "complete" for r in rows) == 180 and sum(r["family"] == "cached" for r in rows) == 255, "Changed cost family counts")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"], row["procedure"], row["family"], row["mode"], row["batch_size"]].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        complete = all(row["status"] == "profiled" for row in group)
        require(len(group) == len(DOMAINS[key[0]]), "A timing summary omits a seed")
        values = [row["median_ms"] for row in group if row["status"] == "profiled"]
        summaries.append(dict(zip(("domain", "procedure", "family", "mode", "batch_size"), key),
                              expected_replicates=len(group), profiled=len(values), complete=complete,
                              mean_seed_median_ms=statistics.mean(values) if complete else None,
                              sample_sd_seed_median_ms=statistics.stdev(values) if complete else None,
                              minimum_seed_median_ms=min(values) if complete else None,
                              maximum_seed_median_ms=max(values) if complete else None))
    return dict(outcome_count=435, complete_predictor_outcomes=180, cached_residual_outcomes=255,
                status_counts=dict(Counter(r["status"] for r in rows)), workload_summaries=summaries,
                aggregation_scope="Mean and sample SD across per-seed medians of 30 timed calls; calls are not independent training replicates.",
                final_batch_scope="The molecular batch-72 workload uses the fitting-cache prefix of length 72, matching the final training batch size; it is not the shuffled final batch's record set.",
                memory_scope="Named tensor storage only; no peak-memory claim."), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selection", "complete"))
    args = parser.parse_args()
    status = read(HERE / "retained_pipeline_status.json")
    if args.mode == "complete":
        require(status["status"] == "complete" and set(status["stages_finished"]) == {"train", "evaluate", "profile"},
                "Complete reporting must wait for all prediction and cost stages")
    else:
        require(status["stage"] != "profile" or status["status"] == "complete", "Defer record analysis during retained timing")
        require((STUDY / "selection_lock.json").exists(), "No common selection yet")
    global torch
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Report analysis must remain on CPU")
    context = checked_json(source(STUDY / "study_context.json"))
    require(context["scope"] == "retained_residual_campaign", "Wrong study scope")
    for record in context["sources"]:
        bytes_checked(record)
    activation = checked_json(context["activation_decision"])
    require(activation["outcome_informed"] is True and activation["tests_reused"] is True, "Exploratory provenance changed")
    selection_report, choice_rows, candidate_rows, choices = reconcile_selection(context)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode, complete_scientific_report=args.mode == "complete",
                  selection=selection_report, scientific_scope=activation["scientific_scope"], outcome_informed=True, tests_reused=True,
                  numerical_comparison_tolerance="For record arithmetic only: 1e-12 + 1e-11 times the absolute retained value; this changes no replay tolerance.",
                  source_program=source(__file__))
    files = [table(OUT / "candidate_inventory.csv", candidate_rows), table(OUT / "validation_choices.csv", choice_rows)]
    if args.mode == "complete":
        evaluation, metric_rows, paired = reconcile_evaluation(choices)
        cost_report, cost_rows = reconcile_costs()
        report.update(evaluation=evaluation, costs=cost_report)
        files.extend([table(OUT / "test_metrics.csv", metric_rows), table(OUT / "paired_differences.csv", paired),
                      table(OUT / "procedure_summaries.csv", evaluation["procedure_summaries"]),
                      table(OUT / "paired_contrasts.csv", evaluation["paired_contrasts"]),
                      table(OUT / "cost_outcomes.csv", cost_rows), table(OUT / "cost_summaries.csv", cost_report["workload_summaries"])])
    for path, digest in SEEN.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "Source changed during analysis: " + path)
    report["arithmetic_checks"] = {key: dict(comparisons=len(values), max_absolute_difference=max(values)) for key, values in DELTAS.items()}
    report["verified_sources"] = [dict(path=path, sha256=digest) for path, digest in sorted(SEEN.items())]
    report["read_sources_unchanged"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    for path, body in files:
        if path.exists():
            require(path.read_text(encoding="utf-8") == body, "Refuse to replace a differing report table: " + str(path))
        else:
            path.write_text(body, encoding="utf-8", newline="")
    report["tables"] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path, _ in files]
    destination = OUT / (args.mode + "_reconciliation.json")
    require(not destination.exists(), "A completed analysis receipt already exists; inspect it instead of overwriting")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(mode=args.mode, report=str(destination), selection=selection_report,
                          evaluation=report.get("evaluation"), cost_status_counts=report.get("costs", {}).get("status_counts")), allow_nan=False))


if __name__ == "__main__":
    main()
