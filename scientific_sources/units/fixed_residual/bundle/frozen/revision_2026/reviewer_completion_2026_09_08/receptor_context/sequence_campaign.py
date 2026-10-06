"""Retained sequence candidates with complete-epoch continuation and fixed selection.

Nothing runs at import. The implementation lock is unavailable until discarded
GPU feasibility and the separate campaign audit have passed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import sequence_execution as execution
import sequence_selection as selection
from sequence_data import SequenceStore
import sequence_models as models

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]/"data/lp_pdbbind/tensors_reconstructed"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        raise FileExistsError("Preserve completed artifact: "+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_process(folder):
    """Windows releases this file lock if the fitting process is interrupted."""
    import msvcrt
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"active_process.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def verify_sources(record):
    for source in record["sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("A source changed: "+source["path"])


def verify_lock():
    lock = read(HERE/"sequence_implementation_lock.json")
    verify_sources(lock)
    if lock["candidates"] != selection.candidate_plan():
        raise ValueError("Candidate plan changed after lock")
    return lock


def load_split(split, store):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown original split")
    blob = execution.indexed_blob(torch.load(DATA/f"pdbbind_{split}.pt", map_location="cpu", weights_only=True))
    expected = {key for key, row in store.by_id.items() if row["split"] == split}
    if set(blob["ids"]) != expected or len(blob["ids"]) != len(expected):
        raise ValueError("Tensor and sequence split identities disagree")
    return blob


def retain_state(path, state):
    if path.exists():
        prior = torch.load(path, map_location="cpu", weights_only=True)
        if set(prior) != set(state) or not all(torch.equal(prior[key], state[key]) for key in prior):
            raise ValueError("Existing checkpoint differs from recovered best state")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    torch.save(state, temporary)
    temporary.replace(path)


def retain_predictions(path, ids, truth, prediction):
    arrays = dict(ids=np.asarray(ids, dtype=str), truth=np.asarray(truth, dtype=np.float64),
                  prediction=np.asarray(prediction, dtype=np.float64))
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or not all(np.array_equal(prior[key], value) for key, value in arrays.items()):
                raise ValueError("Existing predictions differ from the recovered predictor")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def verify_candidate(row, plan, lock_sha):
    if any(row.get(key) != value for key, value in plan.items()) or row["implementation_lock_sha256"] != lock_sha:
        raise ValueError("Candidate identity or source lock mismatch")
    if row["status"] not in ("valid", "failed"):
        raise ValueError("An incomplete candidate is not a terminal result")
    for artifact in row["artifacts"]+row["attempt_records"]:
        if sha(HERE/artifact["path"]) != artifact["sha256"]:
            raise ValueError("Preserved candidate artifact changed")
    if sha(HERE/row["continuation_file"]) != row["continuation_sha256"]:
        raise ValueError("The terminal continuation snapshot changed")


def fit_candidate(plan, train, validation, store, mean, sd, lock_sha, *, device="cuda", epoch_limit=execution.EPOCHS):
    identifier = plan["id"]
    result_path = HERE/"sequence_candidates"/(identifier+".json")
    if result_path.exists():
        row = read(result_path)
        if row["epoch_limit"] != epoch_limit:
            raise ValueError("Completed candidate uses a different epoch horizon")
        verify_candidate(row, plan, lock_sha)
        return row
    continuation = HERE/"sequence_continuation"/(identifier+".pt")
    attempt_folder = HERE/"sequence_attempts"/identifier
    attempt_folder.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempt_folder.glob("attempt_*.json"))
    runtime = execution.make_runtime(plan["seed"], plan["setting"], plan["head"], plan["lr"], device=device, epochs=epoch_limit)
    if continuation.exists():
        boundary = execution.restore_snapshot(continuation, runtime, mean, sd)
    else:
        if previous:
            raise ValueError("An interrupted candidate has no recoverable initial/epoch snapshot")
        execution.save_snapshot(continuation, runtime, mean, sd)
        boundary = "initialized state before the first epoch"
    attempt_path = attempt_folder/f"attempt_{len(previous)+1:03d}.json"
    attempt = dict(started_utc=utc(), process_id=os.getpid(), candidate_id=identifier,
        implementation_lock_sha256=lock_sha, start_boundary=boundary,
        resumed_from_epoch=runtime["epochs_completed"], continuation_sha256_at_start=sha(continuation),
        previous_attempts=[dict(path=str(p.relative_to(HERE)), sha256=sha(p),
            finished="finished_utc" in read(p)) for p in previous],
        discarded_partial_work="Any work beyond the saved complete-epoch boundary is replayed; it is not recorded as preserved optimization progress.")
    write_json(attempt_path, attempt, immutable=True)
    started = time.perf_counter()
    status, error = "valid", None
    artifacts = []
    try:
        while runtime["epochs_completed"] < epoch_limit and runtime["bad_checks"] < execution.PATIENCE:
            epoch = execution.finish_epoch(runtime, train, validation, store, mean, sd)
            execution.save_snapshot(continuation, runtime, mean, sd)
            write_json(HERE/"sequence_active_candidate.json", dict(updated_utc=utc(), candidate_id=identifier,
                completed_epoch=epoch["epoch"], best_epoch=runtime["best_epoch"],
                best_validation_rmse=runtime["best_validation_rmse"], attempt=len(previous)+1))
        if runtime["best_state"] is None:
            raise FloatingPointError("No valid validation checkpoint")
        runtime["model"].load_state_dict(runtime["best_state"], strict=True)
        prediction = execution.predict(runtime["model"], validation, store, mean, sd)
        checked = execution.metric(validation["y"].numpy(), prediction)["rmse"]
        if checked != runtime["best_validation_rmse"]:
            raise RuntimeError("Recovered best-state validation prediction differs from its recorded score")
        checkpoint = HERE/"sequence_checkpoints"/(identifier+".pt")
        validation_file = HERE/"sequence_validation_predictions"/(identifier+".npz")
        retain_state(checkpoint, runtime["best_state"])
        retain_predictions(validation_file, validation["ids"], validation["y"].numpy(), prediction)
        artifacts = [dict(path=str(p.relative_to(HERE)), sha256=sha(p)) for p in (checkpoint, validation_file)]
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exception:
        status, error = "failed", dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
    except Exception:
        attempt.update(finished_utc=utc(), elapsed_seconds=time.perf_counter()-started,
            status="execution_error", traceback=traceback.format_exc())
        write_json(attempt_path, attempt)
        raise
    attempt.update(finished_utc=utc(), elapsed_seconds=time.perf_counter()-started,
        status=status, completed_epoch=runtime["epochs_completed"], error=error,
        continuation_sha256_at_finish=sha(continuation))
    write_json(attempt_path, attempt)
    finite_score = runtime["best_validation_rmse"] if math.isfinite(runtime["best_validation_rmse"]) else None
    row = plan | dict(status=status, finished_utc=utc(), implementation_lock_sha256=lock_sha,
        best_validation_rmse=finite_score, best_epoch=runtime["best_epoch"],
        epoch_limit=epoch_limit, epochs_completed=runtime["epochs_completed"], optimizer_steps=runtime["optimizer_steps"],
        stopping_reason="numerical_or_resource_failure" if status == "failed" else
            "validation_patience" if runtime["bad_checks"] >= execution.PATIENCE else "epoch_cap",
        target_mean=mean, target_population_sd=sd, parameters=runtime["model"].parameter_counts(),
        completed_epoch_seconds=math.fsum(h["completed_epoch_seconds"] for h in runtime["history"]),
        elapsed_current_attempt_seconds=attempt["elapsed_seconds"],
        history=runtime["history"], error=error, artifacts=artifacts,
        attempt_records=[dict(path=str(p.relative_to(HERE)), sha256=sha(p)) for p in sorted(attempt_folder.glob("attempt_*.json"))],
        continuation_file=str(continuation.relative_to(HERE)), continuation_sha256=sha(continuation),
        timing_scope="Completed epoch time includes training and scheduled validation, excludes snapshot/file I/O and any discarded interrupted tail. Attempt records retain observed elapsed time separately.")
    write_json(result_path, row, immutable=True)
    del runtime
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def lock():
    if (HERE/"sequence_implementation_lock.json").exists():
        raise FileExistsError("The retained source lock is preserved")
    # Hashing test tensors records provenance; no test tensor is loaded here.
    sources = set(HERE/name for name in ("sequence_campaign.py", "sequence_selection.py", "sequence_execution.py",
        "sequence_models.py", "sequence_data.py", "sequence_encoder.py", "sequence_protocol.md",
        "sequence_implementation_supplement.md", "sequence_execution_plan.json", "sequence_scoring.py"))
    for name in ("sequence_metadata.json", "sequence_model_cpu_audit.json", "sequence_engine_cpu_audit.json",
                 "sequence_selection_cpu_audit.json", "sequence_gpu_audit.json", "sequence_campaign_cpu_audit.json",
                 "sequence_scoring_cpu_audit.json"):
        audit = read(HERE/name)
        if not audit["passed"]:
            raise ValueError("Required pre-fit audit did not pass: "+name)
        verify_sources(audit)
        sources.add(HERE/name)
        sources.update(Path(item["path"]) for item in audit["sources"])
    for split in ("train", "val", "test"):
        sources.add(DATA/f"pdbbind_{split}.pt")
    if list((HERE/"sequence_candidates").glob("*.json")) or list((HERE/"sequence_attempts").glob("*/*")):
        raise ValueError("Retained fitting artifacts predate the requested implementation lock")
    feasibility = read(HERE/"sequence_gpu_audit.json")
    projected = 100*math.ceil(7384/256)*10*sum(row["ordinary_256_median_seconds"] for row in feasibility["rows"])
    payload = dict(locked_utc=utc(), candidate_count=110, selection_count=55,
        candidates=selection.candidate_plan(), epochs=100, effective_batch=256, patience_checks=8,
        target_scaling="Fitting-only float64 mean and population SD; float32 standardized fitting; CPU float64 inverse scaling and scoring.",
        continuation="Initialized or complete-epoch snapshots; no fresh-seed replacement of numerical/resource failures.",
        projected_training_update_hours=projected/3600,
        projection_scope="100 epochs for all110 candidates, using each configuration's discarded first-256-record median step. This excludes validation, checkpoint I/O and differences in sequence-length composition; it is a planning estimate, not a guarantee or equal-compute limit.",
        sources=[dict(path=str(p), sha256=sha(p)) for p in sorted(sources, key=str)])
    write_json(HERE/"sequence_implementation_lock.json", payload, immutable=True)
    print(json.dumps({k:v for k,v in payload.items() if k not in ("sources", "candidates")}), flush=True)


def train():
    lock_record = verify_lock()
    from sequence_gpu_audit import assert_inputs
    with exclusive_process(HERE/"sequence_run_control"):
        assert_inputs()
        execution.configure("cuda")
        store = SequenceStore()
        fitting, validation = load_split("train", store), load_split("val", store)
        mean, sd = execution.scalers(fitting)
        expected = read(HERE/"sequence_gpu_audit.json")
        if (mean, sd) != (expected["target_mean"], expected["target_population_sd"]):
            raise ValueError("Fitting scaling changed after GPU audit")
        lock_sha = sha(HERE/"sequence_implementation_lock.json")
        finished = []
        for plan in lock_record["candidates"]:
            row = fit_candidate(plan, fitting, validation, store, mean, sd, lock_sha)
            finished.append(row)
            progress = dict(updated_utc=utc(), completed=len(finished), total=110,
                valid=sum(r["status"] == "valid" for r in finished), failed=sum(r["status"] == "failed" for r in finished),
                completed_epoch_seconds=math.fsum(r["completed_epoch_seconds"] for r in finished), last_id=plan["id"])
            write_json(HERE/"sequence_progress.json", progress)
            print(json.dumps(progress), flush=True)
        selected = selection.select(finished)
        selected.update(locked_utc=utc(), implementation_lock_sha256=lock_sha,
            candidate_records=[dict(path=str((HERE/"sequence_candidates"/(r["id"]+".json")).relative_to(HERE)),
                sha256=sha(HERE/"sequence_candidates"/(r["id"]+".json"))) for r in finished])
        destination = HERE/"sequence_selection_lock.json"
        if destination.exists():
            previous = read(destination)
            if {k:v for k,v in previous.items() if k != "locked_utc"} != {k:v for k,v in selected.items() if k != "locked_utc"}:
                raise ValueError("Existing selection differs from the complete candidate record")
        else:
            write_json(destination, selected, immutable=True)
        print(json.dumps(dict(choices=55, nominated_procedure=selected["nominated_procedure"],
                             test_scored=False)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lock", "train"))
    arguments = parser.parse_args()
    {"lock": lock, "train": train}[arguments.stage]()
