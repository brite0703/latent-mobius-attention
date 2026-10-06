"""Discarded, single-CPU file lifecycle and common-selection audit."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

import candidate_lifecycle as life
import execution
import selection

HERE = Path(__file__).resolve().parent


def require_failure(call, expected=ValueError):
    try:
        call()
    except expected as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("The intentionally invalid operation was accepted")


def spec(domain="cubic", seed=100, arm="product", rate_index=0):
    return next(s for s in selection.candidate_plan() if (s["domain"], s["seed"], s["arm"], s["rate_index"]) == (domain, seed, arm, rate_index))


def fixtures(root, domain, seed, *, epoch_zero=False):
    directory = Path(root) / "manufactured_inputs" / f"{domain}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    parent = directory / "mock_parent.pt"
    life.atomic_tensor(parent, dict(scope="manufactured lifecycle audit parent; not a trained model", domain=domain, seed=seed), immutable=True)
    binding = life.sha(parent)
    dimension = 12 if domain == "cubic" else 8
    generator = torch.Generator().manual_seed(93000+seed)
    z = torch.randn(1, 8, dimension, generator=generator)
    mean, sd = (0., 1.) if domain == "cubic" else (6.5, 1.2)
    paths = {}
    for split, count in (("train", 27), ("val", 17)):
        f0 = torch.zeros(count)
        truth = torch.full((count,), mean if epoch_zero and split == "val" else mean+sd, dtype=torch.float64)
        cache = execution.make_cache(split, [f"fixture:{domain}:{seed}:{split}:{i}" for i in range(count)],
                                     f0, z.expand(count, -1, -1).clone(), truth, mean, sd, binding)
        blob = dict(split=split, ids=list(cache.ids), f0=cache.f0, z=cache.z, truth=cache.truth,
                    target_mean=cache.target_mean, target_sd=cache.target_sd,
                    parent_checkpoint_sha256=binding, content_digest=cache.digest,
                    scope="manufactured discarded inputs")
        path = directory / f"{split}.pt"
        life.atomic_tensor(path, blob, immutable=True)
        paths[split] = path
    return paths, parent


def setup(root, current_spec, *, epoch_zero=False, epochs=4, patience=8):
    life.create_discarded_study(root, epoch_limit=epochs, batch_size=13, patience=patience)
    paths, parent = fixtures(root, current_spec["domain"], current_spec["seed"], epoch_zero=epoch_zero)
    life.bind_candidate(root, current_spec, paths["train"], paths["val"], parent)
    return paths, parent


def restored(root, current_spec):
    _, train, val = life.read_binding(root, current_spec)
    runtime, pointer = life.load_snapshot(life.candidate_folder(root, current_spec), train, val)
    return execution.export_runtime(runtime), pointer


def main():
    receipt_path = HERE / "candidate_lifecycle_cpu_audit_v2.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve the completed lifecycle audit")
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = HERE / "discarded_lifecycle_audit_20260909_v2"
    if root.exists():
        raise FileExistsError("Preserve any earlier successful or failed audit directory")
    root.mkdir()
    checks = []
    current = spec()

    uninterrupted = root / "uninterrupted"
    setup(uninterrupted, current)
    reference_row = life.fit_candidate(uninterrupted, current)
    reference, _ = restored(uninterrupted, current)
    assert reference_row["status"] == "valid" and reference_row["completed_epochs"] == 4
    assert reference_row["optimizer_updates"] == 12 and reference_row["best_epoch"] > 0
    assert all(row["batch_sizes"] == [13, 13, 1] for row in reference["epoch_rows"])
    folder = life.candidate_folder(uninterrupted, current)
    recorded_files = [folder / "result.json", folder / "selected_checkpoint.pt", folder / "validation_predictions.pt", folder / "continuation.json"]
    before = {str(p): life.sha(p) for p in recorded_files}
    second = life.fit_candidate(uninterrupted, current)
    assert second == reference_row and before == {str(p): life.sha(p) for p in recorded_files}
    assert len(life.attempt_sources(folder)) == 1
    checks.append(dict(name="terminal checkpoint replay and completed-fit idempotence", passed=True,
                       epochs=4, updates=12, exact_prediction_metric_reload=True, existing_artifacts_unchanged=True))

    paused = root / "pause_resume"
    setup(paused, current)
    assert life.fit_candidate(paused, current, pause_after_epochs=2) is None
    pause_folder = life.candidate_folder(paused, current)
    assert not (pause_folder / "result.json").exists()
    pause_state, _ = restored(paused, current)
    assert pause_state["completed_epochs"] == 2
    pause_row = life.fit_candidate(paused, current)
    resumed, _ = restored(paused, current)
    assert life.same(reference, resumed)
    assert pause_row["best_validation_mse"] == reference_row["best_validation_mse"]
    assert len(life.attempt_sources(pause_folder)) == 2
    checks.append(dict(name="serialized complete-epoch pause and resume", passed=True,
                       resumed_from_epoch=2, all_runtime_state_exact=True))

    partial = root / "partial_epoch_error"
    setup(partial, current)
    original_epoch = execution.train_epoch
    def interrupted_epoch(runtime, train, val):
        # Perform a genuine parameter update and consume the shuffle state,
        # then interrupt before any complete-epoch state can be published.
        model, optimizer = runtime["model"], runtime["optimizer"]
        model.train()
        order = torch.randperm(len(train.ids), generator=runtime["generator"])
        idx = order[:runtime["settings"]["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        prediction = execution.predict_cached(train.f0[idx], train.z[idx], model)
        loss = (prediction-train.y_std[idx]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        runtime["updates"] += 1
        raise RuntimeError("Injected discarded partial-epoch interruption")
    execution.train_epoch = interrupted_epoch
    try:
        failure = require_failure(lambda: life.fit_candidate(partial, current), RuntimeError)
    finally:
        execution.train_epoch = original_epoch
    partial_folder = life.candidate_folder(partial, current)
    assert not (partial_folder / "result.json").exists()
    prior, _ = restored(partial, current)
    assert prior["completed_epochs"] == prior["updates"] == 0
    life.fit_candidate(partial, current)
    replay, _ = restored(partial, current)
    assert life.same(reference, replay)
    attempts = life.attempt_sources(partial_folder)
    assert life.read(attempts[0]["path"])["status"] == "execution_error"
    assert life.read(attempts[1]["path"])["resumed_from_epoch"] == 0
    checks.append(dict(name="partial-epoch updates discarded and reproduced", passed=True,
                       failure=failure, all_runtime_state_exact=True, attempt_count=2))

    interrupted_write = root / "inactive_slot_interruption"
    setup(interrupted_write, current)
    assert life.fit_candidate(interrupted_write, current, pause_after_epochs=1) is None
    state, pointer = restored(interrupted_write, current)
    write_folder = life.candidate_folder(interrupted_write, current)
    inactive = write_folder / f"continuation_{1-pointer['slot']}.pt"
    inactive.write_bytes(b"manufactured unfinished inactive-slot write")
    state_after, _ = restored(interrupted_write, current)
    assert life.same(state, state_after)
    life.fit_candidate(interrupted_write, current)
    write_replay, _ = restored(interrupted_write, current)
    assert life.same(reference, write_replay)
    checks.append(dict(name="incomplete inactive state cannot replace active state", passed=True, all_runtime_state_exact=True))

    for label in ("active_state", "cache", "parent", "terminal_metric", "prediction_vector", "study_context", "missing_candidate", "source_inventory", "restored_candidate"):
        case = root / ("rejection_"+label)
        paths, parent = setup(case, current)
        case_folder = life.candidate_folder(case, current)
        if label in ("terminal_metric", "prediction_vector"):
            life.fit_candidate(case, current)
        else:
            life.fit_candidate(case, current, pause_after_epochs=1)
        if label == "active_state":
            pointer = life.read(case_folder / "continuation.json")
            Path(pointer["state"]["path"]).write_bytes(b"intentional active state corruption")
            action = lambda: life.fit_candidate(case, current)
        elif label == "cache":
            paths["train"].write_bytes(b"intentional bound cache corruption")
            action = lambda: life.fit_candidate(case, current)
        elif label == "parent":
            parent.write_bytes(b"intentional bound parent corruption")
            action = lambda: life.fit_candidate(case, current)
        elif label == "terminal_metric":
            row = life.read(case_folder / "result.json")
            row["best_validation_mse"] += .1
            life.atomic_json(case_folder / "result.json", row)
            action = lambda: life.verify_candidate(case, current)
        elif label == "prediction_vector":
            path = case_folder / "validation_predictions.pt"
            blob = torch.load(path, weights_only=True, map_location="cpu")
            blob["prediction_original_units"][0] += .25
            life.atomic_tensor(path, blob)
            # Even with its file checksum refreshed, independent replay rejects it.
            row = life.read(case_folder / "result.json")
            row["artifacts"][1] = life.artifact(path)
            life.atomic_json(case_folder / "result.json", row)
            action = lambda: life.verify_candidate(case, current)
        elif label == "study_context":
            context = life.read(case / "study_context.json")
            context["sources"][0]["sha256"] = "0"*64
            life.atomic_json(case / "study_context.json", context)
            action = lambda: life.fit_candidate(case, current)
        elif label == "source_inventory":
            context = life.read(case / "study_context.json")
            context["sources"] = context["sources"][:-1]
            life.atomic_json(case / "study_context.json", context)
            action = lambda: life.verify_study(case)
        elif label == "restored_candidate":
            pointer = life.read(case_folder / "continuation.json")
            saved = torch.load(pointer["state"]["path"], weights_only=True, map_location="cpu")
            saved["spec"] = spec(rate_index=1)
            life.atomic_tensor(pointer["state"]["path"], saved)
            pointer["state"] = life.artifact(pointer["state"]["path"])
            life.atomic_json(case_folder / "continuation.json", pointer)
            assert len(life.attempt_sources(case_folder)) == 1
            action = lambda: life.fit_candidate(case, current)
        else:
            action = lambda: life.lock_selection(case)
        failure = require_failure(action, (ValueError, FileNotFoundError))
        if label == "restored_candidate":
            assert "Restored candidate identity" in failure["message"]
            assert len(life.attempt_sources(case_folder)) == 1
        if label == "source_inventory":
            assert "complete execution-source inventory" in failure["message"]
        checks.append(dict(name="reject "+label, passed=True, failure=failure))

    zero = root / "epoch_zero"
    setup(zero, current, epoch_zero=True, epochs=10, patience=2)
    zero_row = life.fit_candidate(zero, current)
    assert zero_row["best_epoch"] == 0 and zero_row["best_validation_mse"] == 0
    assert zero_row["termination"] == "early_stopped" and zero_row["completed_epochs"] == 5
    checks.append(dict(name="eligible epoch zero and exact early-stopping inspections", passed=True, selected_epoch=0, stopped_epoch=5))

    common = root / "common_gate"
    life.create_discarded_study(common, epoch_limit=2, batch_size=13, patience=8)
    for domain, seeds in selection.DOMAINS.items():
        for seed in seeds:
            if (domain, seed) not in (("cubic", 100), ("ligand_contact", 42)):
                life.missing_parent(common, domain, seed, "Manufactured missing-parent fixture, not an unavailable scientific baseline")
            else:
                paths, parent = fixtures(common, domain, seed)
                for current_spec in selection.candidate_plan():
                    if (current_spec["domain"], current_spec["seed"]) != (domain, seed):
                        continue
                    life.bind_candidate(common, current_spec, paths["train"], paths["val"], parent)
                    numerical_case = domain == "cubic" and current_spec["arm"] == "product"
                    if numerical_case:
                        def numerical_failure(runtime, train, val):
                            raise execution.NumericalFailure("Injected discarded finite-arithmetic failure")
                        execution.train_epoch = numerical_failure
                    try:
                        row = life.fit_candidate(common, current_spec)
                    finally:
                        execution.train_epoch = original_epoch
                    assert row["status"] == ("failed_numerical" if numerical_case else "valid")
                    if numerical_case:
                        assert row["completed_epochs"] == 0 and row["best_validation_mse"] is None
                        before_failure = life.sha(life.candidate_folder(common, current_spec) / "result.json")
                        assert life.fit_candidate(common, current_spec)["status"] == "failed_numerical"
                        assert life.sha(life.candidate_folder(common, current_spec) / "result.json") == before_failure
    locked = life.lock_selection(common)
    assert len(locked["candidates"]) == 90 and len(locked["choices"]) == 45
    assert sum(r["outcome"] == "missing_parent" for r in locked["choices"]) == 39
    assert sum(r["outcome"] == "no_valid_residual" for r in locked["choices"]) == 1
    assert sum(r["selected_id"] is not None for r in locked["choices"]) == 5
    lock_hash = life.sha(common / "selection_lock.json")
    assert life.lock_selection(common) == locked and life.sha(common / "selection_lock.json") == lock_hash
    assert not locked["test_scoring_allowed"]
    existing_missing = life.candidate_folder(common, spec(seed=101)) / "result.json"
    missing_hash = life.sha(existing_missing)
    life.missing_parent(common, "cubic", 101, "Manufactured missing-parent fixture, not an unavailable scientific baseline")
    assert life.sha(existing_missing) == missing_hash
    require_failure(lambda: life.missing_parent(common, "cubic", 101, "A changed reason"))
    checks.append(dict(name="all-declared-candidate gate and numerical/missing-parent outcomes", passed=True,
                       candidates=90, choices=45, numerical_failures=2, missing_parent_candidates=78,
                       valid_candidates=10, no_valid_residual_choices=1, missing_parent_choices=39,
                       selected_choices=5, selection_idempotent=True, test_scoring_allowed=False))

    # Restore has been checked against the complete actual optimizer and best-state
    # structures, not merely a chosen prediction or loss tolerance.
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), groups=len(checks), checks=checks,
                  scope="Discarded manufactured single-CPU lifecycle work; no actual dataset or retained parent loaded",
                  source_files=[life.artifact(HERE / name) for name in life.SOURCE_NAMES + (Path(__file__).name,)],
                  sources=[life.artifact(HERE / name) for name in life.SOURCE_NAMES + (Path(__file__).name,)],
                  audit_directory=str(root), cuda_initialized=False, retained_residual_fits=0, new_test_predictions=False,
                  preceding_audit=life.artifact(HERE / "candidate_lifecycle_cpu_audit.json"),
                  preceding_sources=life.artifact(HERE / "lifecycle_source_history_v1/source_map.json"),
                  revision_reason="Reject altered restored-candidate identity before creating an attempt or updating, and require the complete source inventory",
                  remaining="Retained activation/source lock, molecular parent/cache verification, actual test scoring and cost measurements")
    receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("passed", "groups", "cuda_initialized", "retained_residual_fits", "new_test_predictions")}))


if __name__ == "__main__":
    main()
