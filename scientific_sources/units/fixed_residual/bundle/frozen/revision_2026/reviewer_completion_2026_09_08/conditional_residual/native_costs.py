"""Complete native frozen-predictor CPU cost kernel; no dataset or test reader."""
from copy import deepcopy
import time

import torch

import costs
import execution
from models import ARMS, FrozenLmaFeatures, PairResidual, predict_cached


def workload_plan():
    return [dict(domain=domain, procedure=procedure, batch_size=batch, mode="complete_inference")
            for domain in ("cubic", "ligand_contact")
            for procedure in ("unchanged",)+ARMS for batch in (1, 32, 256)]


def module_snapshot(module):
    return dict(state=execution.tensor_copy(module.state_dict()),
                modes=[m.training for m in module.modules()],
                flags=[p.requires_grad for p in module.parameters()],
                gradients=[None if p.grad is None else p.grad.detach().clone() for p in module.parameters()],
                hooks=[(tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in module.modules()])


def validate_inputs(batch, cache, domain, size):
    cache.verify()
    dimension = 12 if domain == "cubic" else 8
    if cache.split != "train" or batch.get("split") != "train" or cache.z.dtype != torch.float32 or cache.z.shape[-1] != dimension:
        raise ValueError("Complete profiling requires a float32 fitting cache in the declared domain")
    if tuple(batch["ids"]) != cache.ids[:len(batch["ids"])] or len(batch["ids"]) < size:
        raise ValueError("Native inputs must follow the same sufficient fitting-cache prefix")
    inputs = batch["inputs"]
    if set(inputs) != ({"x", "mask"} if domain == "cubic" else {"x", "mask", "adj", "contact"}):
        raise ValueError("Unexpected native input fields")
    count, nodes, features = (len(batch["ids"]), 12, 12) if domain == "cubic" else (len(batch["ids"]), 57, 53)
    shapes = dict(x=(count, nodes, features), mask=(count, nodes))
    if domain == "ligand_contact":
        shapes.update(adj=(count, nodes, nodes), contact=(count, nodes, 216))
    for name, value in inputs.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad
                or tuple(value.shape) != shapes[name] or value.dtype != (torch.bool if name == "mask" else torch.float32)
                or not torch.isfinite(value).all()):
            raise ValueError("Invalid native fitting tensor: "+name)
    return {name:value[:size].clone() for name,value in inputs.items()}


def original_units(prediction, mean, sd):
    if prediction.ndim != 1 or prediction.dtype != torch.float32:
        raise ValueError("The complete native predictor must emit float32 scalars")
    return prediction.double()*sd+mean


def native_prediction(parent, features, residual, inputs, mean, sd):
    if residual is None:
        value = parent(**inputs)
    else:
        f0, z = features(**inputs)
        value = predict_cached(f0, z, residual)
    return original_units(value, mean, sd)


def profile(parent, layer_path, residual, cache, batch, workload, *, warmups=5, repeats=30):
    """Copy supplied models and time only the complete inference calculation."""
    if workload not in workload_plan() or type(warmups) is not int or warmups < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("Invalid complete-predictor workload or repetition count")
    if torch.get_num_threads() != 1 or torch.cuda.is_initialized():
        raise ValueError("Complete predictor profiling uses one CPU thread and no CUDA runtime")
    domain, procedure, size = workload["domain"], workload["procedure"], workload["batch_size"]
    if (residual is None) != (procedure == "unchanged"):
        raise ValueError("The supplied residual does not match the procedure")
    if residual is not None and (not isinstance(residual, PairResidual) or residual.arm != procedure
                                or residual.d != (12 if domain == "cubic" else 8)):
        raise ValueError("The residual architecture and declared workload differ")
    if layer_path != ("head.layers.0" if domain == "cubic" else "pool.layers.0"):
        raise ValueError("Unexpected native first-order layer path")
    supplied = [parent]+([] if residual is None else [residual])
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for model in supplied for p in model.parameters()):
        raise ValueError("The native models must use CPU float32 parameters")
    inputs = validate_inputs(batch, cache, domain, size)
    saved = [module_snapshot(model) for model in supplied]
    input_saved = deepcopy(batch)
    rng = torch.get_rng_state().clone()
    try:
        working_parent = deepcopy(parent).eval().requires_grad_(False)
        features = FrozenLmaFeatures(working_parent, layer_path)
        working_residual = None if residual is None else deepcopy(residual).eval().requires_grad_(False)
        with torch.inference_mode():
            direct_f0 = working_parent(**inputs)
            f0, z = features(**inputs)
            if not torch.equal(direct_f0, f0):
                raise ValueError("Native feature capture changed the same-shape baseline prediction")
            for name, native, cached in (("f0", f0, cache.f0[:size]), ("Z", z, cache.z[:size])):
                if not torch.isfinite(native).all() or not torch.all((native-cached).abs() <= .00003+.00003*cached.abs()):
                    raise ValueError("The native "+name+" does not reproduce its fitting cache within the declared tolerance")
            expected = original_units(f0 if working_residual is None else predict_cached(f0,z,working_residual), cache.target_mean, cache.target_sd)
            measured = native_prediction(working_parent, features, working_residual, inputs, cache.target_mean, cache.target_sd)
            if not torch.equal(expected, measured):
                raise ValueError("The complete native prediction and same-shape reconstruction differ")
        times = []
        for repeat in range(warmups+repeats):
            started = time.perf_counter_ns()
            with torch.inference_mode():
                value = native_prediction(working_parent, features, working_residual, inputs, cache.target_mean, cache.target_sd)
            elapsed_ms = (time.perf_counter_ns()-started)/1e6
            if not torch.isfinite(value).all():
                raise execution.NumericalFailure("Nonfinite complete-predictor profiling output")
            if not torch.equal(value, expected):
                raise ValueError("Repeated complete inference changed its output")
            if repeat >= warmups:
                times.append(elapsed_ms)
        return dict(workload=deepcopy(workload), warmups=warmups, repetitions_ms=times,
                    summary=costs.timing_summary(times), record_ids=list(batch["ids"][:size]),
                    parent_checkpoint_sha256=cache.parent_checkpoint_sha256, fitting_cache_digest=cache.digest,
                    native_cache_max_delta=dict(f0=float((f0-cache.f0[:size]).abs().max()), z=float((z-cache.z[:size]).abs().max())),
                    same_shape_complete_prediction_exact=True,
                    original_unit_predictions=expected.tolist(),
                    parameters=dict(parent=sum(p.numel() for p in parent.parameters()), residual=0 if residual is None else residual.parameter_count()),
                    tensor_storage=dict(native_input_bytes=costs.storage_bytes(inputs),
                        parent_parameter_bytes=costs.storage_bytes(tuple(parent.parameters())),
                        parent_buffer_bytes=costs.storage_bytes(tuple(parent.buffers())),
                        residual_parameter_bytes=0 if residual is None else costs.storage_bytes(tuple(residual.parameters())),
                        residual_buffer_bytes=0 if residual is None else costs.storage_bytes(tuple(residual.buffers()))),
                    source_models_inputs_cache_and_rng_unchanged=True,
                    measurement_scope="Complete native inference from loaded fitting tensors, including temporary bucket capture/copies when a residual is supplied, float32 addition and once-only float64 inverse target transform; preprocessing, I/O, construction and correctness checks excluded",
                    storage_scope="Named tensor storage only, not peak process or activation memory")
    finally:
        if any(not costs.same(module_snapshot(model), state) for model,state in zip(supplied,saved)):
            raise ValueError("Complete profiling changed a supplied model's state, modes, gradients, flags or hooks")
        if not costs.same(batch, input_saved) or not torch.equal(torch.get_rng_state(),rng):
            raise ValueError("Complete profiling changed its inputs or caller CPU random state")
        cache.verify()
