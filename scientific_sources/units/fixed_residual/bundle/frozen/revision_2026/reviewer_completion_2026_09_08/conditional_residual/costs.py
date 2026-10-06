"""Incremental cached-residual CPU timing; no native-parent or dataset reader."""
from copy import deepcopy
import math
import time

import numpy as np
import torch

import execution
from models import ARMS, PairResidual, predict_cached


def workload_plan():
    rows = []
    for domain, dimension in (("cubic", 12), ("ligand_contact", 8)):
        for arm in ARMS:
            for batch in (1, 32, 256):
                rows.append(dict(domain=domain, dimension=dimension, arm=arm, batch_size=batch, mode="inference"))
            for batch in ((256,) if domain == "cubic" else (256, 72)):
                for mode in ("fresh_optimizer_step", "steady_optimizer_step"):
                    rows.append(dict(domain=domain, dimension=dimension, arm=arm, batch_size=batch, mode=mode))
    assert len(rows) == 36
    return rows


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def storage_bytes(values):
    if isinstance(values, torch.Tensor):
        return values.numel()*values.element_size()
    if isinstance(values, dict):
        return sum(storage_bytes(value) for value in values.values())
    if isinstance(values, (list, tuple)):
        return sum(storage_bytes(value) for value in values)
    return 0


def optimizer_for(model, rate):
    if rate not in (.0003, .001):
        raise ValueError("The benchmark must use a declared candidate rate")
    return torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0001)


def optimizer_step(model, optimizer, f0, z, target_parent_units):
    """One batch only: no validation, epoch scheduler, snapshot or I/O."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = predict_cached(f0, z, model)
    loss = (prediction-target_parent_units).square().mean()
    if not torch.isfinite(loss):
        raise execution.NumericalFailure("Nonfinite profiling loss")
    loss.backward()
    parameters = list(model.parameters())
    if any(p.grad is None for p in parameters):
        raise ValueError("A benchmark parameter has no intended gradient path")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(norm):
        raise execution.NumericalFailure("Nonfinite profiling gradient")
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in parameters):
        raise execution.NumericalFailure("Nonfinite profiling parameter")
    return float(loss.detach())


def timing_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("At least two finite nonnegative timings are required")
    return dict(repetitions=len(array), median_ms=float(np.median(array)),
                p10_ms=float(np.quantile(array, .1, method="linear")),
                p90_ms=float(np.quantile(array, .9, method="linear")),
                minimum_ms=float(array.min()), maximum_ms=float(array.max()),
                mean_ms=float(array.mean()), sample_sd_ms=float(array.std(ddof=1)))


def profile(model, cache, workload, *, rate=.001, warmups=5, repeats=30):
    """Profile a copy and fitting rows only; scientific artifacts stay unchanged.

    Audit fixtures may use fewer warmups/repeats. The planned retained timing
    allowance is five warmups and thirty measured repetitions per workload.
    """
    if workload not in workload_plan():
        raise ValueError("Unknown or altered cost workload")
    if type(warmups) is not int or warmups < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("Invalid timing repetition counts")
    if not isinstance(model, PairResidual) or model.arm != workload["arm"] or model.d != workload["dimension"]:
        raise ValueError("Benchmark model and workload differ")
    if torch.get_num_threads() != 1:
        raise ValueError("The declared CPU benchmark uses one thread")
    cache.verify()
    batch = workload["batch_size"]
    if cache.split != "train" or len(cache.ids) < batch or cache.z.shape[-1] != workload["dimension"]:
        raise ValueError("Benchmark inputs must be sufficient fitting-cache rows in the declared dimension")
    if any(p.device.type != "cpu" or p.dtype != cache.z.dtype for p in model.parameters()):
        raise ValueError("Benchmark model must match the detached CPU cache precision")
    original = execution.tensor_copy(model.state_dict())
    original_mode = model.training
    original_grad_flags = [p.requires_grad for p in model.parameters()]
    original_rng = torch.get_rng_state().clone()
    working = deepcopy(model).requires_grad_(True)
    f0, z, target = cache.f0[:batch].clone(), cache.z[:batch].clone(), cache.y_std[:batch].clone()
    initial = execution.tensor_copy(working.state_dict())
    optimizer = None
    mode = workload["mode"]
    steady_state = None
    setup_steps = 0
    if mode == "steady_optimizer_step":
        optimizer = optimizer_for(working, rate)
        optimizer_step(working, optimizer, f0, z, target)
        setup_steps = 1
        steady_state = dict(model=execution.tensor_copy(working.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
    times, starts = [], []
    try:
        for repeat in range(warmups+repeats):
            if mode == "fresh_optimizer_step":
                working.load_state_dict(initial, strict=True)
                optimizer = optimizer_for(working, rate)
                starts.append(len(optimizer.state))
            elif mode == "steady_optimizer_step":
                working.load_state_dict(steady_state["model"], strict=True)
                optimizer.load_state_dict(deepcopy(steady_state["optimizer"]))
                if not same(working.state_dict(), steady_state["model"]) or not same(optimizer.state_dict(), steady_state["optimizer"]):
                    raise ValueError("Benchmark state did not restore exactly")
                starts.append(len(optimizer.state))
            else:
                working.eval()
            started = time.perf_counter_ns()
            if mode == "inference":
                with torch.inference_mode():
                    value = predict_cached(f0, z, working)
                if not torch.isfinite(value).all():
                    raise execution.NumericalFailure("Nonfinite profiling prediction")
            else:
                optimizer_step(working, optimizer, f0, z, target)
            elapsed_ms = (time.perf_counter_ns()-started)/1e6
            if repeat >= warmups:
                times.append(elapsed_ms)
    finally:
        if not same(model.state_dict(), original) or model.training != original_mode or original_grad_flags != [p.requires_grad for p in model.parameters()]:
            raise ValueError("Profiling changed its source model")
        if not torch.equal(torch.get_rng_state(), original_rng):
            raise ValueError("Profiling changed the caller's random state")
        cache.verify()
    tensors = dict(cached_prediction_and_bucket_input_bytes=storage_bytes((f0, z)),
                   fitting_target_bytes=storage_bytes(target) if mode != "inference" else 0,
                   parameter_bytes=storage_bytes(tuple(working.parameters())),
                   constant_buffer_bytes=storage_bytes(tuple(working.buffers())),
                   optimizer_tensor_bytes_after_step=storage_bytes(optimizer.state) if optimizer is not None else 0)
    return dict(workload=deepcopy(workload), rate=rate if mode != "inference" else None,
                warmups=warmups, repetitions_ms=times, summary=timing_summary(times),
                discarded_optimizer_setup_steps=setup_steps,
                optimizer_state_entry_counts_before_repetitions=starts,
                parameter_count=model.parameter_count(), tensor_storage=tensors,
                source_model_and_cache_unchanged=True, caller_rng_unchanged=True,
                measurement_scope="Incremental cached f0+g in parent units or isolated residual batch update on one CPU thread; native parent, cache construction, reporting inverse transform, validation, epoch scheduling and I/O excluded",
                storage_scope="Named tensor storage only, not peak process or activation memory")
