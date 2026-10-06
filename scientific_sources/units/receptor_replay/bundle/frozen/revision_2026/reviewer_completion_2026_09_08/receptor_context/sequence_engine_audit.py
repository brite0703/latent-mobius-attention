"""Disposable CPU checks of actual accumulated updates and checkpoint continuation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import sequence_execution as execution
from sequence_data import SequenceStore
import sequence_models as models

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right, (left, right)


def make_toy(folder):
    strings = ["AX", "GG:WY", "ACDEFGHIK", "TT:ACDX:W", "K", "YVV:GX", "WWWW:AC"]
    path = folder/"toy_sequences.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pdbid", "split", "sequence"])
        writer.writeheader()
        writer.writerows(dict(pdbid=f"toy{i}", split="train", sequence=s) for i, s in enumerate(strings))
    store = SequenceStore(path)
    torch.manual_seed(2026090835)
    x = torch.randn(7, 5, 53, dtype=torch.float64)
    mask = torch.arange(5)[None] < torch.tensor([5, 3, 2, 4, 1, 3, 2])[:, None]
    x *= mask.unsqueeze(-1)
    adj = (mask[:, :, None] & mask[:, None, :]).double()/mask.sum(1)[:, None, None]
    # Toy validation deliberately reuses this tiny numerical fixture. It is not an experiment.
    blob = execution.indexed_blob(dict(X=x, mask=mask, adj=adj,
        y=torch.tensor([-1.1, .2, .5, 1.8, -.4, .9, -.7], dtype=torch.float64), ids=[f"toy{i}" for i in range(7)]))
    return store, blob


def run():
    execution.configure("cpu")
    temporary_root = HERE/"tmp"
    temporary_root.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="engine_audit_", dir=temporary_root) as directory:
        folder = Path(directory)
        assert folder.resolve().parent == temporary_root.resolve()
        assert folder.resolve().is_relative_to(HERE.resolve())
        store, blob = make_toy(folder)
        mean, sd = execution.scalers(blob)
        np.testing.assert_allclose([mean, sd], [blob["y"].numpy().mean(), blob["y"].numpy().std(ddof=0)], atol=1e-15, rtol=1e-15)
        for setting, head in models.configurations():
            def fresh():
                return execution.make_runtime(42, setting, head, .001, dtype=torch.float64, epochs=2)
            unsplit, micro = fresh(), fresh()
            a = execution.effective_step(unsplit["model"], unsplit["optimizer"], blob, store, blob["ids"], mean, sd, max_records=7)
            b = execution.effective_step(micro["model"], micro["optimizer"], blob, store, blob["ids"], mean, sd, max_records=2)
            assert a["optimizer_steps"] == b["optimizer_steps"] == 1 and a["microbatches"] == 1 and b["microbatches"] == 4
            differences = []
            for p, q in zip(unsplit["model"].parameters(), micro["model"].parameters()):
                torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-9)
                differences.append(float((p-q).detach().abs().max()))
            assert abs(a["preclip_gradient_norm"]-b["preclip_gradient_norm"]) < 1e-10
            uninterrupted = fresh()
            for _ in range(2):
                execution.finish_epoch(uninterrupted, blob, blob, store, mean, sd, effective_batch=5, max_records=2)
            uninterrupted_prediction = execution.predict(uninterrupted["model"], blob, store, mean, sd, max_records=2)
            expected_draws = (torch.rand(4), np.random.random(4), [random.random() for _ in range(4)])
            interrupted = fresh()
            execution.finish_epoch(interrupted, blob, blob, store, mean, sd, effective_batch=5, max_records=2)
            saved = folder/f"{setting}_{head}.pt"
            execution.save_snapshot(saved, interrupted, mean, sd)
            # Change global RNGs and construct a fresh model before restoring the real file.
            torch.manual_seed(9)
            np.random.seed(9)
            random.seed(9)
            resumed = fresh()
            boundary = execution.restore_snapshot(saved, resumed, mean, sd)
            assert boundary == "complete epoch including its scheduled validation"
            assert resumed["epochs_completed"] == 1 and resumed["optimizer_steps"] == 2
            execution.finish_epoch(resumed, blob, blob, store, mean, sd, effective_batch=5, max_records=2)
            resumed_prediction = execution.predict(resumed["model"], blob, store, mean, sd, max_records=2)
            actual_draws = (torch.rand(4), np.random.random(4), [random.random() for _ in range(4)])
            equal_tree(uninterrupted["model"].state_dict(), resumed["model"].state_dict())
            equal_tree(uninterrupted["optimizer"].state_dict(), resumed["optimizer"].state_dict())
            equal_tree(uninterrupted["scheduler"].state_dict(), resumed["scheduler"].state_dict())
            equal_tree(uninterrupted["generator"].get_state(), resumed["generator"].get_state())
            equal_tree(uninterrupted["best_state"], resumed["best_state"])
            assert uninterrupted["optimizer_steps"] == resumed["optimizer_steps"] == 4
            assert uninterrupted["epochs_completed"] == resumed["epochs_completed"] == 2
            assert uninterrupted["scheduler"].last_epoch == resumed["scheduler"].last_epoch == 2
            for left, right in zip(uninterrupted["history"], resumed["history"]):
                equal_tree({k: v for k, v in left.items() if k != "completed_epoch_seconds"},
                           {k: v for k, v in right.items() if k != "completed_epoch_seconds"})
            np.testing.assert_array_equal(uninterrupted_prediction, resumed_prediction)
            torch.testing.assert_close(expected_draws[0], actual_draws[0], atol=0, rtol=0)
            np.testing.assert_array_equal(expected_draws[1], actual_draws[1])
            assert expected_draws[2] == actual_draws[2]
            score = execution.metric(blob["y"].numpy(), resumed_prediction)
            assert abs(score["rmse"]-np.sqrt(np.mean((blob["y"].numpy()-resumed_prediction)**2))) < 1e-12
            rejected = 0
            for wrong_mean, wrong_lr in [(mean+.01, .001), (mean, .0003)]:
                other = execution.make_runtime(42, setting, head, wrong_lr, dtype=torch.float64, epochs=2)
                try:
                    execution.restore_snapshot(saved, other, wrong_mean, sd)
                except ValueError:
                    rejected += 1
            assert rejected == 2
            results.append(dict(setting=setting, head=head, accumulated_update_max_delta=max(differences),
                clipping_norm_delta=abs(a["preclip_gradient_norm"]-b["preclip_gradient_norm"]),
                exact_cpu_continuation=True, exact_optimizer_scheduler_generator_states=True,
                torch_numpy_python_rngs_restored=True, exact_prediction_reload=True,
                optimizer_steps=4, epochs=2, mismatched_continuations_rejected=2))
    sources = [HERE/name for name in ("sequence_execution.py", "sequence_engine_audit.py", "sequence_models.py", "sequence_data.py", "sequence_encoder.py")]
    output = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True, configurations=len(results), checks=results,
        sources=[dict(path=str(path), sha256=sha(path)) for path in sources],
        qualification="Disposable seven-record CPU numerical fixture, not retained affinity training. The actual accumulation/update and serialized continuation path is checked. CUDA continuation, feasibility, timings and study-level selection remain pending.")
    (HERE/"sequence_engine_cpu_audit.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(passed=True, configurations=len(results),
        maximum_accumulated_update_delta=max(r["accumulated_update_max_delta"] for r in results),
        all_cpu_continuations_exact=True, optimizer_scheduler_and_rngs_verified=True)), flush=True)


if __name__ == "__main__":
    run()
