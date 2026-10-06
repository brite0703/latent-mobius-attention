"""File-backed residual candidate control; no retained-study launch command.

The discarded-audit constructor is deliberately separate from the future
activation/source-lock orchestrator. Nothing runs when this module is imported.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

import execution
import selection

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = ("models.py", "execution.py", "selection.py", "candidate_lifecycle.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError("Preserved JSON artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_tensor(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if not same(torch.load(path, weights_only=True, map_location="cpu"), value):
            raise ValueError("Preserved tensor artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    torch.save(value, temporary)
    temporary.replace(path)


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verify_artifact(record):
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Changed bound artifact: " + record["path"])


@contextmanager
def exclusive(root):
    # The operating system releases this byte lock if its process disappears.
    import msvcrt
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "active_process.lock").open("a+b") as stream:
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


def create_discarded_study(root, *, epoch_limit=4, batch_size=13, patience=8):
    """Prepare new audit-only files; this cannot activate the proposed study."""
    root = Path(root).resolve()
    path = root / "study_context.json"
    if path.exists():
        raise FileExistsError("Preserve the existing study context")
    if not 1 <= epoch_limit <= 100 or not 1 <= batch_size <= 256 or not 1 <= patience <= 8:
        raise ValueError("Invalid execution allowance")
    context = dict(schema=1, scope="discarded_lifecycle_audit", created_utc=utc(),
                   settings=dict(epoch_limit=epoch_limit, batch_size=batch_size, patience=patience),
                   candidate_plan=selection.candidate_plan(), sources=[artifact(HERE / name) for name in SOURCE_NAMES],
                   retained_residual_study_activated=False, test_scoring_allowed=False)
    atomic_json(path, context, immutable=True)
    return path


def verify_study(root):
    path = Path(root) / "study_context.json"
    context = read(path)
    if context["schema"] != 1 or context["scope"] != "discarded_lifecycle_audit":
        raise ValueError("A retained activation/source-lock orchestrator has not been implemented")
    if context["retained_residual_study_activated"] or context["test_scoring_allowed"]:
        raise ValueError("An audit context cannot authorize retained training or test scoring")
    if context["candidate_plan"] != selection.candidate_plan():
        raise ValueError("The declared candidate allocation changed")
    if [Path(s["path"]).resolve() for s in context["sources"]] != [(HERE / name).resolve() for name in SOURCE_NAMES]:
        raise ValueError("The complete execution-source inventory is required")
    for source in context["sources"]:
        verify_artifact(source)
    return context


def load_cache(record, expected_split, parent_sha):
    verify_artifact(record)
    blob = torch.load(record["path"], weights_only=True, map_location="cpu")
    if blob["split"] != expected_split or blob["parent_checkpoint_sha256"] != parent_sha:
        raise ValueError("Cache split or actual parent binding differs")
    cache = execution.make_cache(blob["split"], blob["ids"], blob["f0"], blob["z"], blob["truth"],
                                 blob["target_mean"], blob["target_sd"], blob["parent_checkpoint_sha256"])
    if cache.digest != blob["content_digest"] or cache.digest != record["content_digest"]:
        raise ValueError("Cache content digest differs")
    return cache


def candidate_folder(root, spec):
    selection.validate_spec(spec)
    return Path(root).resolve() / "candidates" / spec["id"]


def bind_candidate(root, spec, train_path, val_path, parent_path):
    context = verify_study(root)
    folder = candidate_folder(root, spec)
    binding_path = folder / "binding.json"
    if binding_path.exists() or (folder / "result.json").exists():
        raise FileExistsError("A candidate's source binding is immutable")
    parent = artifact(parent_path)
    caches = {}
    for split, path in (("train", train_path), ("val", val_path)):
        record = artifact(path)
        record["content_digest"] = torch.load(path, weights_only=True, map_location="cpu")["content_digest"]
        load_cache(record, split, parent["sha256"])
        caches[split] = record
    binding = dict(spec=deepcopy(spec), study_context=artifact(Path(root) / "study_context.json"),
                   settings=context["settings"], parent_checkpoint=parent, caches=caches)
    atomic_json(binding_path, binding, immutable=True)
    return binding_path


def read_binding(root, spec):
    context = verify_study(root)
    folder = candidate_folder(root, spec)
    path = folder / "binding.json"
    binding = read(path)
    if binding["spec"] != spec or binding["settings"] != context["settings"]:
        raise ValueError("Candidate identity or allowance differs")
    verify_artifact(binding["study_context"])
    if Path(binding["study_context"]["path"]).resolve() != (Path(root) / "study_context.json").resolve():
        raise ValueError("Candidate points to another study")
    verify_artifact(binding["parent_checkpoint"])
    parent_sha = binding["parent_checkpoint"]["sha256"]
    train = load_cache(binding["caches"]["train"], "train", parent_sha)
    val = load_cache(binding["caches"]["val"], "val", parent_sha)
    if set(train.ids) & set(val.ids):
        raise ValueError("Fitting and validation identities overlap")
    return binding, train, val


def publish_snapshot(folder, runtime):
    """Publish the inactive state slot, then atomically advance its hash pointer.

    An interrupted inactive-slot write cannot alter the currently named state.
    At most two full states are retained; the active state includes the history.
    """
    folder = Path(folder)
    pointer_path = folder / "continuation.json"
    previous = read(pointer_path) if pointer_path.exists() else None
    slot = 1 - previous["slot"] if previous else 0
    state_path = folder / f"continuation_{slot}.pt"
    atomic_tensor(state_path, execution.export_runtime(runtime))
    pointer = dict(slot=slot, completed_epochs=runtime["completed_epochs"], state=artifact(state_path),
                   boundary="complete epoch including its scheduled validation" if runtime["completed_epochs"] else "initialized epoch-zero state")
    atomic_json(pointer_path, pointer)
    return pointer


def load_snapshot(folder, train, val):
    folder = Path(folder)
    pointer = read(folder / "continuation.json")
    if pointer["slot"] not in (0, 1) or Path(pointer["state"]["path"]).resolve() != (
        folder / f"continuation_{pointer['slot']}.pt").resolve():
        raise ValueError("Unexpected continuation state path")
    verify_artifact(pointer["state"])
    saved = torch.load(pointer["state"]["path"], weights_only=True, map_location="cpu")
    if pointer["completed_epochs"] != saved["completed_epochs"]:
        raise ValueError("Continuation pointer and state disagree")
    return execution.restore_runtime(saved, train, val), pointer


def attempt_sources(folder):
    return [artifact(p) for p in sorted((Path(folder) / "attempts").glob("attempt_*.json"))]


def fit_candidate(root, spec, *, pause_after_epochs=None):
    """Complete or deliberately pause one audit candidate; never selects tests."""
    folder = candidate_folder(root, spec)
    with exclusive(folder):
        terminal_path = folder / "result.json"
        if terminal_path.exists():
            return verify_candidate(root, spec)
        binding, train, val = read_binding(root, spec)
        if pause_after_epochs is not None and (type(pause_after_epochs) is not int or pause_after_epochs < 1):
            raise ValueError("An audit pause requires a positive number of new epochs")
        previous_attempts = attempt_sources(folder)
        if (folder / "continuation.json").exists():
            runtime, pointer = load_snapshot(folder, train, val)
        else:
            if previous_attempts:
                raise ValueError("An interrupted candidate lacks a published complete state")
            runtime = execution.make_runtime(spec, train, val, **binding["settings"])
            pointer = publish_snapshot(folder, runtime)
        if runtime["spec"] != spec or any(runtime["settings"].get(key) != value for key, value in binding["settings"].items()):
            raise ValueError("Restored candidate identity or allowance differs from its immutable binding")
        attempt_path = folder / "attempts" / f"attempt_{len(previous_attempts)+1:03d}.json"
        attempt = dict(started_utc=utc(), process_id=os.getpid(), resumed_from_epoch=runtime["completed_epochs"],
                       start_pointer=artifact(folder / "continuation.json"), start_state=pointer["state"],
                       previous_attempts=previous_attempts, discarded_partial_work="Replay only work beyond the published complete state")
        atomic_json(attempt_path, attempt, immutable=True)
        started, completed_here = time.perf_counter(), 0
        try:
            while runtime["termination"] is None:
                verify_study(root)
                execution.train_epoch(runtime, train, val)
                pointer = publish_snapshot(folder, runtime)
                completed_here += 1
                if pause_after_epochs is not None and completed_here >= pause_after_epochs and runtime["termination"] is None:
                    attempt.update(finished_utc=utc(), status="paused_at_complete_epoch", elapsed_seconds=time.perf_counter()-started,
                                   completed_epochs=runtime["completed_epochs"], finish_pointer=artifact(folder / "continuation.json"))
                    atomic_json(attempt_path, attempt)
                    return None
            runtime["model"].load_state_dict(runtime["best_state"], strict=True)
            score, pred, original = execution.validation(runtime["model"], val, binding["settings"]["batch_size"])
            if score != runtime["best_score"] or not torch.equal(pred, runtime["best_prediction_std"]) or not torch.equal(original, runtime["best_prediction_original"]):
                raise ValueError("Selected state does not reproduce its saved validation metric and vectors")
            checkpoint_path = folder / "selected_checkpoint.pt"
            prediction_path = folder / "validation_predictions.pt"
            atomic_tensor(checkpoint_path, runtime["best_state"], immutable=True)
            atomic_tensor(prediction_path, dict(ids=list(val.ids), truth=val.truth, prediction_parent_units=pred, prediction_original_units=original), immutable=True)
            row = execution.completed_record(runtime)
            row.update(artifacts=[artifact(checkpoint_path), artifact(prediction_path)])
            status = "valid"
        except execution.NumericalFailure as error:
            # A partly updated/nonfinite epoch is never published as resumable.
            boundary_runtime, pointer = load_snapshot(folder, train, val)
            row = dict(spec=deepcopy(spec), status="failed_numerical", parent_checkpoint_sha256=train.parent_checkpoint_sha256,
                       completed_epochs=boundary_runtime["completed_epochs"], optimizer_updates=boundary_runtime["updates"],
                       best_validation_mse=None, best_epoch=None, artifacts=[],
                       error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            status = "failed_numerical"
        except BaseException as error:
            attempt.update(finished_utc=utc(), status="execution_error", elapsed_seconds=time.perf_counter()-started,
                           error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            atomic_json(attempt_path, attempt)
            raise
        attempt.update(finished_utc=utc(), status=status, elapsed_seconds=time.perf_counter()-started,
                       completed_epochs=row["completed_epochs"], finish_pointer=artifact(folder / "continuation.json"))
        atomic_json(attempt_path, attempt)
        row.update(finished_utc=utc(), binding=artifact(folder / "binding.json"),
                   continuation_pointer=artifact(folder / "continuation.json"), continuation_state=pointer["state"],
                   attempts=attempt_sources(folder), timing_scope="Attempt wall time includes validation and file I/O; an abruptly ended attempt has no fabricated duration")
        atomic_json(terminal_path, row, immutable=True)
        return verify_candidate(root, spec)


def missing_parent(root, domain, seed, reason):
    verify_study(root)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A missing parent needs an explicit provenance reason")
    specs = [s for s in selection.candidate_plan() if s["domain"] == domain and s["seed"] == seed]
    if len(specs) != 6:
        raise ValueError("Unknown parent replicate")
    with exclusive(Path(root) / "missing_parent_control" / f"{domain}_{seed}"):
        context_source = artifact(Path(root) / "study_context.json")
        for spec in specs:
            folder = candidate_folder(root, spec)
            if (folder / "binding.json").exists():
                raise ValueError("Do not replace a bound candidate with a missing parent")
            if (folder / "result.json").exists():
                previous = verify_candidate(root, spec)
                expected = dict(spec=spec, status="missing_parent", reason=reason, study_context=context_source)
                if {k: v for k, v in previous.items() if k != "finished_utc"} != expected:
                    raise ValueError("A prior missing-parent result differs")
        # Existing consistent rows survive an interrupted six-row publication.
        for spec in specs:
            folder = candidate_folder(root, spec)
            if not (folder / "result.json").exists():
                row = dict(spec=spec, status="missing_parent", reason=reason, study_context=context_source, finished_utc=utc())
                atomic_json(folder / "result.json", row, immutable=True)


def verify_candidate(root, spec):
    verify_study(root)
    folder = candidate_folder(root, spec)
    row = read(folder / "result.json")
    if row["spec"] != spec or row["status"] not in ("valid", "failed_numerical", "missing_parent"):
        raise ValueError("An invalid or nonterminal candidate cannot be selected")
    if row["status"] == "missing_parent":
        verify_artifact(row["study_context"])
        if Path(row["study_context"]["path"]).resolve() != (Path(root) / "study_context.json").resolve():
            raise ValueError("Missing-parent record points to another study")
        if not row["reason"].strip() or (folder / "binding.json").exists():
            raise ValueError("Inconsistent missing-parent outcome")
        return row
    for source in [row["binding"], row["continuation_pointer"], row["continuation_state"]] + row["artifacts"] + row["attempts"]:
        verify_artifact(source)
    if Path(row["binding"]["path"]).resolve() != (folder / "binding.json").resolve() or Path(
        row["continuation_pointer"]["path"]).resolve() != (folder / "continuation.json").resolve():
        raise ValueError("Terminal record points to another candidate's binding or continuation")
    expected_artifact_paths = [folder / "selected_checkpoint.pt", folder / "validation_predictions.pt"] if row["status"] == "valid" else []
    if [Path(s["path"]).resolve() for s in row["artifacts"]] != expected_artifact_paths:
        raise ValueError("The retained checkpoint/prediction inventory differs")
    if row["attempts"] != attempt_sources(folder):
        raise ValueError("Completed candidate has unrecorded or changed attempts")
    binding, train, val = read_binding(root, spec)
    runtime, pointer = load_snapshot(folder, train, val)
    if runtime["spec"] != spec or any(runtime["settings"].get(key) != value for key, value in binding["settings"].items()):
        raise ValueError("Terminal runtime identity or allowance differs from its immutable binding")
    if row["continuation_state"] != pointer["state"] or row["parent_checkpoint_sha256"] != train.parent_checkpoint_sha256:
        raise ValueError("Terminal state or parent association differs")
    if row["completed_epochs"] != runtime["completed_epochs"] or row["optimizer_updates"] != runtime["updates"]:
        raise ValueError("Terminal progress differs from the complete-state boundary")
    if row["status"] == "failed_numerical":
        if row["artifacts"] or row["best_validation_mse"] is not None or not row.get("error"):
            raise ValueError("A numerical failure must not nominate its earlier checkpoint")
        return row
    expected = execution.completed_record(runtime)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Terminal candidate metric or stopping metadata differs")
    inspected = runtime["history"]
    chosen = min(inspected, key=lambda r: (r["validation_mse"], r["epoch"]))
    if (chosen["epoch"], chosen["validation_mse"]) != (row["best_epoch"], row["best_validation_mse"]):
        raise ValueError("The selected epoch is not the earliest minimum validation inspection")
    if [r["epoch"] for r in runtime["epoch_rows"]] != list(range(1, runtime["completed_epochs"]+1)):
        raise ValueError("Incomplete epoch history")
    state = torch.load(folder / "selected_checkpoint.pt", weights_only=True, map_location="cpu")
    if not same(state, runtime["best_state"]):
        raise ValueError("The retained checkpoint differs from the best state")
    runtime["model"].load_state_dict(state, strict=True)
    score, pred, original = execution.validation(runtime["model"], val, binding["settings"]["batch_size"])
    saved = torch.load(folder / "validation_predictions.pt", weights_only=True, map_location="cpu")
    expected_predictions = dict(ids=list(val.ids), truth=val.truth, prediction_parent_units=pred, prediction_original_units=original)
    if score != row["best_validation_mse"] or not same(saved, expected_predictions):
        raise ValueError("Checkpoint, labels, prediction vectors and selected metric disagree")
    return row


def lock_selection(root):
    root = Path(root)
    with exclusive(root / "selection_control"):
        context = verify_study(root)
        records = [verify_candidate(root, spec) for spec in context["candidate_plan"]]
        choices = selection.select_all(records)
        body = dict(schema=1, study_context=artifact(root / "study_context.json"),
                    candidates=[artifact(candidate_folder(root, spec) / "result.json") for spec in context["candidate_plan"]],
                    choices=choices, candidate_count=90, selection_count=45,
                    scope="discarded_lifecycle_audit", test_scoring_allowed=False)
        path = root / "selection_lock.json"
        if path.exists():
            prior = read(path)
            if {k: v for k, v in prior.items() if k != "locked_utc"} != body:
                raise ValueError("A previously locked selection differs")
            return prior
        body["locked_utc"] = utc()
        atomic_json(path, body, immutable=True)
        return body
