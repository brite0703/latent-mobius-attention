"""Whole-model accumulation and complete-epoch recovery for the sequence study."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import sequence_models as models
from sequence_data import MAX_RECORDS, MAX_PADDED_TOKENS

EFFECTIVE_BATCH = 256
EPOCHS = 100
PATIENCE = 8


def configure(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The retained study requires the audited CUDA device")


def scalers(blob):
    y = blob["y"].double()
    mean = float(y.mean())
    sd = float((y-mean).square().mean().sqrt())
    if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
        raise ValueError("Invalid fitting-only target scaling")
    return mean, sd


def indexed_blob(blob):
    result = dict(blob)
    ids = [str(key) for key in result["ids"]]
    if len(ids) != len(set(ids)) or len(ids) != len(result["y"]):
        raise ValueError("Nonunique or unaligned record IDs")
    result["ids"] = ids
    result["row_by_id"] = {key: i for i, key in enumerate(ids)}
    return result


def collate(model, blob, store, ids):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    indices = torch.tensor([blob["row_by_id"][key] for key in ids], dtype=torch.long)
    arguments = dict(x=None, mask=None, adj=None, sequence_batch=None)
    if model.encoder is not None:
        mask = blob["mask"].index_select(0, indices)
        valid_columns = torch.where(mask.any(0))[0]
        if not len(valid_columns):
            raise ValueError("Empty ligand graphs are not in the retained cohort")
        end = int(valid_columns[-1])+1
        arguments.update(x=blob["X"].index_select(0, indices)[:, :end].to(device=device, dtype=dtype),
            mask=mask[:, :end].to(device),
            adj=blob["adj"].index_select(0, indices)[:, :end, :end].to(device=device, dtype=dtype))
    if model.sequence is not None:
        arguments["sequence_batch"] = store.collate(ids, device=device)
    truth = blob["y"].index_select(0, indices).to(device=device, dtype=dtype)
    return arguments, truth


def finite_output(model, arguments):
    prediction = model(**arguments)
    if prediction.ndim != 1 or not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite or malformed model prediction")
    product = getattr(model.pool, "last_product", None)
    if product is not None and not bool(torch.isfinite(product).all()):
        raise FloatingPointError("Nonfinite CP product before tanh")
    maximum_product = float(product.abs().max()) if product is not None else None
    return prediction, maximum_product


def effective_step(model, optimizer, blob, store, ids, mean, sd, max_records=MAX_RECORDS, max_tokens=MAX_PADDED_TOKENS):
    if not ids:
        raise ValueError("An effective batch cannot be empty")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    normalized_sse, maximum_product, microbatches, padded_tokens = 0., None, 0, 0
    for chunk in store.chunks(ids, max_records=max_records, max_tokens=max_tokens):
        arguments, truth = collate(model, blob, store, chunk)
        prediction, product = finite_output(model, arguments)
        target = (truth-mean)/sd
        squared = (prediction-target).square().sum()
        if not bool(torch.isfinite(squared)):
            raise FloatingPointError("Nonfinite normalized training loss")
        (squared/len(ids)).backward()
        normalized_sse += float(squared.detach())
        microbatches += 1
        padded_tokens = max(padded_tokens, sum(store.sizes[key][0] for key in chunk)*max(store.sizes[key][1] for key in chunk))
        if product is not None:
            maximum_product = max(maximum_product or 0., product)
    parameters = list(model.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise FloatingPointError("Absent whole-model gradient")
    if not bool(torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).all()):
        raise FloatingPointError("Nonfinite whole-model gradient")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in parameters]).all()):
        raise FloatingPointError("Nonfinite parameter after the optimizer step")
    return dict(records=len(ids), normalized_sse=normalized_sse, preclip_gradient_norm=float(norm),
                clipped=bool(norm > 10), microbatches=microbatches, maximum_padded_tokens=padded_tokens,
                maximum_abs_cp_product=maximum_product, optimizer_steps=1)


@torch.no_grad()
def predict(model, blob, store, mean, sd, max_records=MAX_RECORDS, max_tokens=MAX_PADDED_TOKENS):
    model.eval()
    values = []
    for ids in store.chunks(blob["ids"], max_records=max_records, max_tokens=max_tokens):
        arguments, _ = collate(model, blob, store, ids)
        prediction, _ = finite_output(model, arguments)
        # Inverse scaling is explicitly performed in float64 on CPU.
        values.append(prediction.detach().cpu().double()*sd+mean)
    output = torch.cat(values).numpy()
    if output.shape != (len(blob["ids"]),) or not np.isfinite(output).all():
        raise FloatingPointError("Invalid inverse-scaled predictions")
    return output


def metric(truth, prediction):
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid metric arrays")
    error = p-y
    centered_y = y-math.fsum(y)/len(y)
    centered_p = p-math.fsum(p)/len(p)
    denominator = math.sqrt(math.fsum(centered_y*centered_y)*math.fsum(centered_p*centered_p))
    return dict(n=len(y), rmse=math.sqrt(math.fsum(error*error)/len(y)), mae=math.fsum(abs(error))/len(y),
                pearson=math.fsum(centered_y*centered_p)/denominator if denominator else None)


def make_runtime(seed, setting, head, lr, device="cpu", dtype=torch.float32, epochs=EPOCHS):
    model = models.build_model(seed, setting, head).to(device=device, dtype=dtype)
    torch.manual_seed(30000+seed)
    random.seed(30000+seed)
    np.random.seed(30000+seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return dict(model=model, optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs),
        generator=torch.Generator().manual_seed(20000+seed),
        identity=dict(seed=seed, setting=setting, head=head, lr=lr, epochs=epochs, dtype=str(dtype)),
        epochs_completed=0, optimizer_steps=0, best_validation_rmse=float("inf"), best_epoch=None,
        best_state=None, bad_checks=0, history=[])


def finish_epoch(runtime, train, validation, store, mean, sd, effective_batch=EFFECTIVE_BATCH,
                 max_records=MAX_RECORDS, max_tokens=MAX_PADDED_TOKENS):
    start_time = time.perf_counter()
    model = runtime["model"]
    permutation = torch.randperm(len(train["ids"]), generator=runtime["generator"]).tolist()
    steps = []
    for start in range(0, len(permutation), effective_batch):
        ids = [train["ids"][i] for i in permutation[start:start+effective_batch]]
        step = effective_step(model, runtime["optimizer"], train, store, ids, mean, sd, max_records, max_tokens)
        steps.append(step)
        runtime["optimizer_steps"] += 1
    runtime["scheduler"].step()
    runtime["epochs_completed"] += 1
    epoch = runtime["epochs_completed"]
    products = [s["maximum_abs_cp_product"] for s in steps if s["maximum_abs_cp_product"] is not None]
    row = dict(epoch=epoch, normalized_training_mse=math.fsum(s["normalized_sse"] for s in steps)/len(train["ids"]),
        optimizer_steps=len(steps), microbatches=sum(s["microbatches"] for s in steps),
        clipped_effective_batches=sum(s["clipped"] for s in steps),
        maximum_preclip_gradient_norm=max(s["preclip_gradient_norm"] for s in steps),
        maximum_abs_cp_product=max(products) if products else None,
        next_learning_rate=float(runtime["optimizer"].param_groups[0]["lr"]))
    if epoch == 1 or epoch % 5 == 0 or epoch == runtime["identity"]["epochs"]:
        prediction = predict(model, validation, store, mean, sd, max_records, max_tokens)
        score = metric(validation["y"].numpy(), prediction)["rmse"]
        row["validation_rmse"] = score
        if score < runtime["best_validation_rmse"]:
            runtime.update(best_validation_rmse=score, best_epoch=epoch,
                best_state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, bad_checks=0)
        else:
            runtime["bad_checks"] += 1
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    row["completed_epoch_seconds"] = time.perf_counter()-start_time
    runtime["history"].append(row)
    return row


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return copy.deepcopy(value)


def snapshot(runtime, mean, sd):
    rng = np.random.get_state()
    bookkeeping = {key: to_cpu(value) for key, value in runtime.items() if key not in ("model", "optimizer", "scheduler", "generator")}
    boundary = "complete epoch including its scheduled validation" if runtime["epochs_completed"] else "initialized state before the first epoch"
    return dict(schema=1, boundary=boundary, bookkeeping=bookkeeping,
        target_mean=mean, target_sd=sd, model=to_cpu(runtime["model"].state_dict()),
        optimizer=to_cpu(runtime["optimizer"].state_dict()), scheduler=to_cpu(runtime["scheduler"].state_dict()),
        generator=runtime["generator"].get_state().clone(), torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if next(runtime["model"].parameters()).is_cuda else [],
        python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]))


def save_snapshot(path, runtime, mean, sd):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    torch.save(snapshot(runtime, mean, sd), temporary)
    temporary.replace(path)


def restore_snapshot(path, runtime, mean, sd):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != 1 or saved["bookkeeping"]["identity"] != runtime["identity"]:
        raise ValueError("Snapshot belongs to another candidate or execution specification")
    if (saved["target_mean"], saved["target_sd"]) != (mean, sd):
        raise ValueError("Target-scaling mismatch on continuation")
    runtime["model"].load_state_dict(saved["model"], strict=True)
    runtime["optimizer"].load_state_dict(saved["optimizer"])
    runtime["optimizer"].zero_grad(set_to_none=True)
    runtime["scheduler"].load_state_dict(saved["scheduler"])
    runtime["generator"].set_state(saved["generator"])
    for key, value in saved["bookkeeping"].items():
        runtime[key] = value
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"]:
        if not next(runtime["model"].parameters()).is_cuda:
            raise ValueError("A CUDA continuation requires a CUDA runtime")
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    rng = saved["numpy_rng"]
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
    return saved["boundary"]
