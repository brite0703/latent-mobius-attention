"""Complete-choice scoring, independent reconciliation and declared cost workloads.

Test tensors are loaded by command entry points only after all 110 terminal
candidates and the 55-choice validation lock have been verified.
"""
import argparse
import copy
import csv
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import sequence_campaign as campaign
import sequence_execution as execution
import sequence_models as models
import sequence_selection as selection
from sequence_data import SequenceStore

HERE = Path(__file__).resolve().parent
METRICS = ("rmse", "mae", "pearson")


def arrays(ids, truth, prediction):
    ids = np.asarray(ids, dtype=str)
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if ids.ndim != 1 or truth.shape != ids.shape or prediction.shape != ids.shape:
        raise ValueError("Unaligned prediction arrays")
    if len(ids) != len(set(ids)) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Duplicate identifiers or nonfinite prediction/target")
    return ids, truth, prediction


def independent_metric(truth, prediction):
    """NumPy reductions, independent of the execution engine's fsum metrics."""
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid independent metric inputs")
    if not len(y):
        return dict(n=0, rmse=None, mae=None, pearson=None)
    residual = p-y
    yc, pc = y-y.mean(), p-p.mean()
    denominator = float(np.linalg.norm(yc)*np.linalg.norm(pc))
    result = dict(n=len(y), rmse=float(np.sqrt(np.mean(residual**2))),
                  mae=float(np.mean(np.abs(residual))),
                  pearson=float(np.dot(yc, pc)/denominator) if denominator else None)
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite independent metric")
    return result


def metric_delta(left, right):
    if left["n"] != right["n"]:
        raise ValueError("Metric sample counts differ")
    differences = []
    for key in METRICS:
        a, b = left[key], right[key]
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("Undefined metric conventions differ")
        else:
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Independent metric mismatch: "+key)
            differences.append(abs(a-b))
    return max(differences, default=0.)


def read_predictions(path, blob=None):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"ids", "truth", "prediction"}:
            raise ValueError("Unexpected prediction file fields")
        result = arrays(saved["ids"], saved["truth"], saved["prediction"])
    if blob is not None:
        if not np.array_equal(result[0], np.asarray(blob["ids"], dtype=str)) or not np.array_equal(result[1], blob["y"].numpy()):
            raise ValueError("Prediction IDs or targets differ from the declared split")
    return result


def validate_groups(metadata, test_ids):
    full = set(test_ids)
    if len(full) != len(test_ids) or len(metadata["groups"]) != 9:
        raise ValueError("The fixed test cohort and nine groups are required")
    groups = {name: set(ids) for name, ids in metadata["groups"].items()}
    if groups.get("full") != full:
        raise ValueError("The full test group changed")
    for name, ids in metadata["groups"].items():
        if len(ids) != len(groups[name]) or not groups[name] <= full or len(ids) != metadata["subset_sizes"][name]:
            raise ValueError("Invalid fixed subset: "+name)
    names = ("canonical_ligand", "complete_stored_string", "any_supplied_chain", "both_ligand_and_any_chain")
    for name in names:
        absent = groups[name+"_absent_from_train_and_val"]
        other = "either_ligand_or_any_supplied_chain" if name == names[-1] else name
        if groups[other+"_overlap_with_train_or_val"] != full-absent:
            raise ValueError("Subset complement changed")
    if groups[names[-1]+"_absent_from_train_and_val"] != groups[names[0]+"_absent_from_train_and_val"] & groups[names[2]+"_absent_from_train_and_val"]:
        raise ValueError("Joint absence is not the intersection")
    if not groups[names[2]+"_absent_from_train_and_val"] <= groups[names[1]+"_absent_from_train_and_val"]:
        raise ValueError("Any-chain absence must imply literal-string absence")
    return groups


def group_metrics(ids, truth, prediction, groups):
    ids, truth, prediction = arrays(ids, truth, prediction)
    result = {}
    for name, members in groups.items():
        mask = np.asarray([key in members for key in ids], dtype=bool)
        result[name] = execution.metric(truth[mask], prediction[mask]) if mask.any() else independent_metric([], [])
    return result


def statistics_row(values, prescribed=5):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values) or len(values) > prescribed:
        raise ValueError("Invalid replicate metric")
    return dict(prescribed=prescribed, n=len(values), missing=prescribed-len(values),
        mean=math.fsum(values)/len(values) if values else None,
        sample_sd=statistics.stdev(values) if len(values) > 1 else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None)


def summarize(rows, locked, groups):
    expected = {(row["configuration"], row["seed"]): row for row in locked["selections"]}
    actual = {(row["configuration"], row["seed"]): row for row in rows}
    if len(rows) != len(actual) or len(expected) != 55 or set(actual) != set(expected):
        raise ValueError("Exactly all 55 selected outcomes, including failures, are required")
    for key, row in actual.items():
        choice = expected[key]
        if any(row.get(name) != choice[name] for name in ("status", "selected_id", "configuration", "seed")):
            raise ValueError("A scored outcome differs from the validation choice")
        if row["status"] == "valid":
            if set(row["subsets"]) != set(groups):
                raise ValueError("All nine subset results are required")
            for name, members in groups.items():
                value = row["subsets"][name]
                if value["n"] != len(members):
                    raise ValueError("Subset metric count mismatch")
                for metric in METRICS:
                    number = value[metric]
                    if number is not None and not math.isfinite(number):
                        raise ValueError("Nonfinite subset summary")
                    if len(members) and metric != "pearson" and (number is None or number < 0):
                        raise ValueError("A nonempty subset needs a nonnegative error")
        elif row.get("subsets") is not None:
            raise ValueError("An all-candidates-failed outcome cannot contain test metrics")
    aggregated, paired, seed_differences = [], [], []
    for setting, head in models.configurations():
        configuration = selection.key(setting, head)
        outcomes = [actual[(configuration, seed)] for seed in selection.SEEDS]
        for subset in groups:
            valid = [row for row in outcomes if row["status"] == "valid"]
            aggregated.append(dict(configuration=configuration, setting=setting, head=head, subset=subset,
                test_records=len(groups[subset]), prescribed_seeds=5, valid_seeds=len(valid), failed_seeds=5-len(valid),
                metrics={key: statistics_row([row["subsets"][subset][key] for row in valid]) for key in METRICS}))
    for contrast in locked["contrasts"]:
        for subset in groups:
            deltas = []
            for seed in selection.SEEDS:
                components = [actual[(term["configuration"], seed)] for term in contrast["terms"]]
                complete = all(row["status"] == "valid" and row["subsets"][subset]["rmse"] is not None for row in components)
                delta = math.fsum(term["weight"]*row["subsets"][subset]["rmse"]
                    for term, row in zip(contrast["terms"], components)) if complete else None
                seed_differences.append(dict(contrast=contrast["id"], subset=subset, seed=seed,
                    status="complete" if complete else "unavailable", difference=delta))
                if complete:
                    deltas.append(delta)
            paired.append(dict(contrast=contrast["id"], subset=subset, terms=contrast["terms"],
                test_records=len(groups[subset]), difference=statistics_row(deltas),
                negative=sum(value < 0 for value in deltas), zero=sum(value == 0 for value in deltas),
                positive=sum(value > 0 for value in deltas)))
    return dict(procedures=aggregated, paired_contrasts=paired, paired_seed_differences=seed_differences,
        primary_information_setting=locked["primary_information_setting"], primary_contrast=locked["primary_contrast"],
        nominated_procedure=locked["nominated_procedure"], procedure_validation=locked["procedure_validation"],
        qualification="Five optimization seeds on one fixed, previously reused split. Means and sample SDs are descriptive; missing seeds remain explicit. Equality subsets are not independent datasets, causal leakage estimates or sequence-similarity separation. Nomination remains the locked validation decision.")


def selection_context():
    implementation = campaign.verify_lock()
    locked = campaign.read(HERE/"sequence_selection_lock.json")
    lock_sha = campaign.sha(HERE/"sequence_implementation_lock.json")
    if locked["implementation_lock_sha256"] != lock_sha:
        raise ValueError("Selection source lock mismatch")
    expected = {row["id"]: row for row in implementation["candidates"]}
    records = []
    for artifact in locked["candidate_records"]:
        path = HERE/artifact["path"]
        if campaign.sha(path) != artifact["sha256"]:
            raise ValueError("A terminal candidate record changed")
        record = campaign.read(path)
        campaign.verify_candidate(record, expected[record["id"]], lock_sha)
        records.append(record)
    selected = selection.select(records)
    if any(locked[key] != value for key, value in selected.items()):
        raise ValueError("Stored validation choices do not match all 110 terminal candidates")
    return locked, {row["id"]: row for row in records}


def score_selected(choice, record, validation, test, store, groups, *, device="cuda", retain=True):
    if choice["status"] != "valid" or choice["selected_id"] != record["id"] or record["status"] != "valid":
        raise ValueError("Only the valid locked predictor can be scored")
    model = models.build_model(choice["seed"], choice["setting"], choice["head"]).to(device)
    checkpoint = HERE/"sequence_checkpoints"/(record["id"]+".pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    mean, sd = record["target_mean"], record["target_population_sd"]
    saved_ids, saved_y, saved_prediction = read_predictions(HERE/"sequence_validation_predictions"/(record["id"]+".npz"), validation)
    prediction = execution.predict(model, validation, store, mean, sd)
    if not np.array_equal(saved_prediction, prediction):
        raise ValueError("Selected checkpoint does not reproduce stored validation predictions")
    checked = independent_metric(saved_y, saved_prediction)
    val_metric = execution.metric(saved_y, saved_prediction)
    delta = metric_delta(checked, val_metric)
    if val_metric["rmse"] != choice["validation_rmse"] or val_metric["rmse"] != record["best_validation_rmse"]:
        raise ValueError("The selected validation score differs from its stored predictions")
    test_prediction = execution.predict(model, test, store, mean, sd)
    ids, truth, test_prediction = arrays(test["ids"], test["y"].numpy(), test_prediction)
    path = HERE/"sequence_test_predictions"/(record["id"]+".npz")
    if retain:
        campaign.retain_predictions(path, ids, truth, test_prediction)
    stored = read_predictions(path, test)
    if not np.array_equal(stored[2], test_prediction):
        raise ValueError("Stored test predictions differ from the freshly loaded checkpoint")
    subsets = group_metrics(ids, truth, test_prediction, groups)
    for name, members in groups.items():
        mask = np.asarray([key in members for key in ids])
        delta = max(delta, metric_delta(subsets[name], independent_metric(truth[mask], test_prediction[mask])))
    row = choice | dict(validation=val_metric, subsets=subsets,
        parameters=model.parameter_counts(), checkpoint_sha256=campaign.sha(checkpoint),
        test_prediction_file=str(path.relative_to(HERE)), test_prediction_sha256=campaign.sha(path),
        maximum_independent_metric_delta=delta, validation_reload_maximum_delta=0., test_reload_maximum_delta=0.)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def gpu_start():
    from sequence_gpu_audit import assert_inputs
    assert_inputs()
    execution.configure("cuda")


def evaluate():
    with campaign.exclusive_process(HERE/"sequence_run_control"):
        locked, records = selection_context()
        gpu_start()
        store = SequenceStore()
        validation, test = campaign.load_split("val", store), campaign.load_split("test", store)
        metadata = campaign.read(HERE/"sequence_metadata.json")
        groups = validate_groups(metadata, test["ids"])
        rows = []
        for choice in locked["selections"]:
            row = (score_selected(choice, records[choice["selected_id"]], validation, test, store, groups)
                if choice["status"] == "valid" else choice | dict(subsets=None))
            rows.append(row)
            print(campaign.utc(), "scored", len(rows), "/55", choice["configuration"], choice["seed"], flush=True)
        result = dict(completed_utc=campaign.utc(), selection_lock_sha256=campaign.sha(HERE/"sequence_selection_lock.json"),
            metadata_sha256=campaign.sha(HERE/"sequence_metadata.json"), rows=rows,
            all_choices_locked_before_test_loading=True)
        summarize(rows, locked, groups)
        path = HERE/"sequence_evaluation.json"
        if path.exists():
            previous = campaign.read(path)
            if {k:v for k,v in previous.items() if k != "completed_utc"} != {k:v for k,v in result.items() if k != "completed_utc"}:
                raise ValueError("Preserve the previous complete evaluation")
        else:
            campaign.write_json(path, result, immutable=True)


def audit():
    with campaign.exclusive_process(HERE/"sequence_run_control"):
        locked, records = selection_context()
        evaluation = campaign.read(HERE/"sequence_evaluation.json")
        if evaluation["selection_lock_sha256"] != campaign.sha(HERE/"sequence_selection_lock.json") or evaluation["metadata_sha256"] != campaign.sha(HERE/"sequence_metadata.json"):
            raise ValueError("Evaluation lock or metadata changed")
        gpu_start()
        store = SequenceStore()
        validation, test = campaign.load_split("val", store), campaign.load_split("test", store)
        groups = validate_groups(campaign.read(HERE/"sequence_metadata.json"), test["ids"])
        maximum = 0.
        for record in records.values():
            if record["status"] != "valid":
                continue
            _, truth, prediction = read_predictions(HERE/"sequence_validation_predictions"/(record["id"]+".npz"), validation)
            expected = independent_metric(truth, prediction)
            maximum = max(maximum, abs(expected["rmse"]-record["best_validation_rmse"]))
            if not math.isclose(expected["rmse"], record["best_validation_rmse"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Candidate validation score is inconsistent")
            checks = [(row["validation_rmse"], row["epoch"]) for row in record["history"] if "validation_rmse" in row]
            if min(checks) != (record["best_validation_rmse"], record["best_epoch"]):
                raise ValueError("Selected candidate checkpoint is not the earliest validation minimum")
        actual = {(row["configuration"], row["seed"]): row for row in evaluation["rows"]}
        for choice in locked["selections"]:
            if choice["status"] == "valid":
                checked = score_selected(choice, records[choice["selected_id"]], validation, test, store, groups, retain=False)
                if checked != actual[(choice["configuration"], choice["seed"])]:
                    raise ValueError("A separately reloaded selected predictor changed")
                maximum = max(maximum, checked["maximum_independent_metric_delta"])
        summary = summarize(evaluation["rows"], locked, groups)
        valid = sum(row["status"] == "valid" for row in locked["selections"])
        result = dict(passed=True, completed_utc=campaign.utc(), candidates=110, selections=55,
            valid_candidates=sum(row["status"] == "valid" for row in records.values()), valid_selected=valid,
            failed_selected=55-valid, subsets=9, summary_rows=len(summary["procedures"]), contrasts=len(summary["paired_contrasts"]),
            selected_checkpoints_independently_reloaded=True, maximum_checkpoint_prediction_delta=0.,
            maximum_independent_metric_delta=maximum,
            evaluation_sha256=campaign.sha(HERE/"sequence_evaluation.json"), selection_lock_sha256=campaign.sha(HERE/"sequence_selection_lock.json"),
            qualification="Implementation, source, selection, identity, checkpoint and metric reconciliation; not evidence of statistical significance, convergence or biological mechanism.")
        campaign.write_json(HERE/"sequence_final_audit.json", result)
        print(result, flush=True)


def format_stat(value):
    if value["mean"] is None:
        return "unavailable"
    return f"{value['mean']:.6f}"+(f" ± {value['sample_sd']:.6f}" if value["sample_sd"] is not None else " (one valid seed)")


def analyze():
    locked, _ = selection_context()
    audit_record = campaign.read(HERE/"sequence_final_audit.json")
    if not audit_record["passed"] or audit_record["evaluation_sha256"] != campaign.sha(HERE/"sequence_evaluation.json"):
        raise ValueError("Current complete evaluation audit required")
    evaluation = campaign.read(HERE/"sequence_evaluation.json")
    metadata = campaign.read(HERE/"sequence_metadata.json")
    groups = validate_groups(metadata, metadata["groups"]["full"])
    output = summarize(evaluation["rows"], locked, groups)
    output.update(completed_utc=campaign.utc(), evaluation_sha256=campaign.sha(HERE/"sequence_evaluation.json"),
                  final_audit_sha256=campaign.sha(HERE/"sequence_final_audit.json"))
    campaign.write_json(HERE/"sequence_summary.json", output)
    lines = ["# Matched ligand and stored-sequence results", "",
        "All 110 prescribed candidate outcomes and 55 validation choices are retained. Procedure nomination used validation alone: "+str(locked["nominated_procedure"])+". The primary information setting remains ligand plus sequence.", "",
        "| Procedure | Valid / prescribed | Full-test RMSE | Full-test MAE | Pearson correlation |",
        "|---|---:|---:|---:|---:|"]
    for row in output["procedures"]:
        if row["subset"] == "full":
            lines.append(f"| {row['configuration']} | {row['valid_seeds']}/5 | "+" | ".join(format_stat(row["metrics"][key]) for key in METRICS)+" |")
    lines += ["", "| Declared full-test contrast | RMSE difference ± SD | Negative / valid pairs |", "|---|---:|---:|"]
    for row in output["paired_contrasts"]:
        if row["subset"] == "full":
            lines.append(f"| {row['contrast']} | {format_stat(row['difference'])} | {row['negative']}/{row['difference']['n']} |")
    lines += ["", output["qualification"], "",
        "A joint-input improvement changes both the information and model capacity. It does not alone establish receptor mechanism, geometric contact learning or a higher-order pooling advantage. The Gaussian and Boolean theorems do not apply directly to these nonlinear fitted predictors.", ""]
    (HERE/"sequence_report.md").write_text("\n".join(lines), encoding="utf-8")
    with (HERE/"sequence_paired_seed_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["contrast", "subset", "seed", "status", "difference"])
        writer.writeheader()
        writer.writerows(output["paired_seed_differences"])
    with (HERE/"sequence_all_seed_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["configuration", "seed", "selected_id", "status", "subset", "n", *METRICS])
        writer.writeheader()
        for row in evaluation["rows"]:
            for subset in groups:
                metrics = row["subsets"][subset] if row["status"] == "valid" else dict(n=len(groups[subset]), rmse=None, mae=None, pearson=None)
                writer.writerow({key: row[key] for key in ("configuration", "seed", "selected_id", "status")}|dict(subset=subset)|metrics)
    print(dict(procedure_subsets=len(output["procedures"]), contrasts=len(output["paired_contrasts"]),
               paired_seed_rows=len(output["paired_seed_differences"])), flush=True)


def profile_workload(runtime, blob, store, mean, sd, batch, mode, repeats=5):
    if mode not in ("forward", "training_step") or not 1 <= batch <= len(blob["ids"]) or repeats < 1:
        raise ValueError("Invalid declared cost workload")
    model, optimizer = runtime["model"], runtime["optimizer"]
    initial_model = execution.to_cpu(model.state_dict())
    initial_optimizer = execution.to_cpu(optimizer.state_dict())
    was_training = model.training
    use_cuda = next(model.parameters()).is_cuda
    small = execution.indexed_blob({key: blob[key][:batch] for key in ("X", "mask", "adj", "y", "ids")})
    ids = small["ids"]
    def operation():
        if mode == "forward":
            execution.predict(model, small, store, mean, sd)
        else:
            execution.effective_step(model, optimizer, blob, store, ids, mean, sd)
    def synchronize():
        if use_cuda:
            torch.cuda.synchronize()
    try:
        operation()
        synchronize()
        warm_optimizer = execution.to_cpu(optimizer.state_dict())
        times, baselines, peaks = [], [], []
        for _ in range(repeats):
            model.load_state_dict(initial_model, strict=True)
            optimizer.load_state_dict(copy.deepcopy(warm_optimizer))
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
                baselines.append(torch.cuda.memory_allocated())
            started = time.perf_counter()
            operation()
            synchronize()
            times.append(time.perf_counter()-started)
            if use_cuda:
                peaks.append(torch.cuda.max_memory_allocated())
    finally:
        model.load_state_dict(initial_model, strict=True)
        optimizer.load_state_dict(initial_optimizer)
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
    if not all(torch.equal(model.state_dict()[key].cpu(), value) for key, value in initial_model.items()):
        raise ValueError("Profiling failed to restore selected model state")
    return dict(batch=batch, mode=mode, repetitions=repeats, seconds=times, median_seconds=statistics.median(times),
        baseline_allocated_bytes=baselines, peak_allocated_bytes=peaks,
        checkpoint_restored=True, parameters=model.parameter_counts(), record_ids=ids,
        scope=f"One warm-up followed by {repeats} measurements, each restored to the selected model state with warmed optimizer state. Host collation, transfers and whole-model operations are included; state restoration and initial data loading are excluded. Gradients and optimizer updates occur only in training_step. GPU allocation is not total device memory.")


def profile():
    with campaign.exclusive_process(HERE/"sequence_run_control"):
        locked, records = selection_context()
        final = campaign.read(HERE/"sequence_final_audit.json")
        if not final["passed"] or final["evaluation_sha256"] != campaign.sha(HERE/"sequence_evaluation.json"):
            raise ValueError("Complete current prediction audit required before cost profiling")
        gpu_start()
        store = SequenceStore()
        fitting = campaign.load_split("train", store)
        mean, sd = execution.scalers(fitting)
        rows = []
        for setting, head in models.configurations():
            options = [row for row in locked["selections"] if (row["setting"], row["head"], row["status"]) == (setting, head, "valid")]
            if not options:
                for mode, batch in (("forward", 1), ("forward", 64), ("training_step", 1), ("training_step", 256)):
                    rows.append(dict(setting=setting, head=head, mode=mode, batch=batch, status="no_valid_checkpoint"))
                continue
            choice = min(options, key=lambda row: row["seed"])
            record = records[choice["selected_id"]]
            runtime = execution.make_runtime(choice["seed"], setting, head, choice["learning_rate"], device="cuda")
            checkpoint = HERE/"sequence_checkpoints"/(record["id"]+".pt")
            before = campaign.sha(checkpoint)
            runtime["model"].load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            for mode, batch in (("forward", 1), ("forward", 64), ("training_step", 1), ("training_step", 256)):
                measured = profile_workload(runtime, fitting, store, mean, sd, batch, mode)
                rows.append(dict(setting=setting, head=head, selected_id=record["id"], status="profiled", **measured))
            if before != campaign.sha(checkpoint):
                raise ValueError("Profiling changed the retained checkpoint file")
            del runtime
            torch.cuda.empty_cache()
        result = dict(complete=True, completed_utc=campaign.utc(), rows=rows, workloads=44,
            selection_lock_sha256=campaign.sha(HERE/"sequence_selection_lock.json"),
            device=torch.cuda.get_device_name(), torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
            checkpoint_choice="Lowest prescribed seed with a valid validation-selected predictor, independently of its test score.")
        campaign.write_json(HERE/"sequence_profiles.json", result)
        print(dict(complete=True, workloads=len(rows)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("evaluate", "audit", "analyze", "profile"))
    args = parser.parse_args()
    globals()[args.stage]()
