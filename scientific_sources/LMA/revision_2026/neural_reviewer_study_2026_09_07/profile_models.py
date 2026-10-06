"""Whole-model costs following the workload frozen in protocol.md."""
import copy
import json
import statistics
import time
import numpy as np
import torch
from torch import nn
import models
import study

HERE = study.HERE


def measure_blocks(model, state, x, mask, adj, target, mode, lr):
    wall_ms, peaks, baselines, increments = [], [], [], []
    for block in range(5):
        model.load_state_dict(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) if mode == "train" else None
        torch.manual_seed(94000+block)
        if mode == "train":
            model.train()
            def operation():
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, mask, adj)
                loss = (prediction-target).square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
        else:
            model.eval()
            def operation():
                with torch.inference_mode():
                    return model(x, mask, adj)
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(20):
            operation()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter()-start)*1000/20
        peak = torch.cuda.max_memory_allocated()
        wall_ms.append(elapsed_ms)
        peaks.append(peak/2**20)
        baselines.append(baseline/2**20)
        increments.append((peak-baseline)/2**20)
        with torch.no_grad():
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        del optimizer
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    model.eval()
    return dict(mode="sampled_train_optimizer_step" if mode=="train" else "deterministic_eval_forward",
                block_average_ms=wall_ms, median_block_average_ms=statistics.median(wall_ms),
                minimum_block_average_ms=min(wall_ms), maximum_block_average_ms=max(wall_ms),
                baseline_allocated_mib=baselines, peak_allocated_mib=peaks, incremental_peak_mib=increments)


def main():
    study.configure()
    study.verify_lock()
    selection = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    assert (HERE/"evaluation.json").exists()
    # Load only the declared training workload, leaving other GPU jobs absent.
    blob = torch.load(study.DATA/"pdbbind_train.pt", weights_only=True, map_location="cpu")
    n = int(blob["mask"][:64].sum(1).max())
    workload = {k: blob[k][:64].cuda() for k in ("X", "mask", "adj", "y")}
    workload["X"] = workload["X"][:, :n]
    workload["mask"] = workload["mask"][:, :n]
    workload["adj"] = workload["adj"][:, :n, :n]
    rows = []
    for choice in selection["selections"]:
        if choice["seed"] != 42:
            continue
        if choice["selected_id"] is None:
            rows.append(dict(**choice, status="no_selected_checkpoint"))
            continue
        row = json.loads((HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
        path = HERE/row["checkpoint"]
        assert study.sha(path) == choice["checkpoint_sha256"]
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        model = models.build_model(42, choice["head"], choice["depth"]).cuda()
        state = ckpt["state_dict"]
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            reference = model(workload["X"], workload["mask"], workload["adj"]).clone()
        target = ((workload["y"].double()-ckpt["target_mean"])/ckpt["target_sd"]).float()
        timings = []
        for batch in (1, 64):
            for mode in ("eval", "train"):
                timings.append(dict(batch_size=batch, padded_n=n,
                    **measure_blocks(model, state, workload["X"][:batch], workload["mask"][:batch],
                                     workload["adj"][:batch], target[:batch], mode, row["spec"]["lr"])))
        with torch.no_grad():
            restored = model(workload["X"], workload["mask"], workload["adj"])
        delta = float((restored-reference).abs().max())
        assert delta == 0.
        assert study.sha(path) == choice["checkpoint_sha256"]
        rows.append(dict(**choice, status="profiled", total_parameters=models.parameter_count(model),
                         restored_prediction_max_delta=delta, timings=timings))
        del model
        torch.cuda.empty_cache()
        study.write_json(HERE/"profiles.json", dict(profiled_utc=study.now(), complete=False, rows=rows))
        print(json.dumps(dict(stage="profile", depth=choice["depth"], head=choice["head"])), flush=True)
    study.write_json(HERE/"profiles.json", dict(profiled_utc=study.now(), complete=True,
        source_sha256=study.sha(__file__), implementation_lock_sha256=study.sha(HERE/"implementation_lock.json"),
        selected_seed=42, hardware=torch.cuda.get_device_name(), precision="float32 except CP product+tanh float64",
        workload_ids=blob["ids"][:64], valid_atom_counts=blob["mask"][:64].sum(1).tolist(),
        warmups_per_block=10, repetitions_per_block=20, blocks=5, padded_n=n,
        boundary="preloaded GPU tensors; complete model; training includes zero_grad, MSE, backward, clip and AdamW; excludes disk/data transfer/target de-standardization",
        interpretation="median and range of five block-average times, not per-request latency quantiles",
        rows=rows))


if __name__ == "__main__":
    main()
