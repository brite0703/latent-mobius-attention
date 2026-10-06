"""CPU execution kernel for detached residual caches; no retained-run entry point."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import torch
from models import PairResidual, predict_cached
from selection import validate_spec

HERE = Path(__file__).resolve().parent


class NumericalFailure(RuntimeError):
    """Finite arithmetic failed; caller must retain a failed trajectory."""


def tensor_copy(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


@dataclass
class Cache:
    split: str
    ids: tuple
    f0: torch.Tensor
    z: torch.Tensor
    truth: torch.Tensor
    y_std: torch.Tensor
    target_mean: float
    target_sd: float
    parent_checkpoint_sha256: str
    digest: str = ""

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(dict(split=self.split, ids=self.ids,
            target_mean=self.target_mean, target_sd=self.target_sd,
            parent=self.parent_checkpoint_sha256), sort_keys=True).encode())
        for tensor in (self.f0, self.z, self.truth, self.y_std):
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            h.update(tensor.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def verify(self):
        if not self.digest or self.fingerprint() != self.digest:
            raise ValueError("Frozen cache content or alignment changed")


def make_cache(split, ids, f0, z, truth, mean, sd, parent_checkpoint_sha256):
    """Contract for a future audited producer; this does not verify real sources."""
    if split not in ("train", "val", "test") or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Unknown split, empty cache, or duplicate record identifier")
    if not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Stable nonempty string record identifiers are required")
    if not isinstance(parent_checkpoint_sha256, str) or len(parent_checkpoint_sha256) != 64 or any(
        c not in '0123456789abcdef' for c in parent_checkpoint_sha256):
        raise ValueError("A lowercase SHA256 parent binding is required")
    if not (math.isfinite(mean) and math.isfinite(sd) and sd > 0):
        raise ValueError("Invalid inherited fitting-set target scale")
    if f0.device.type != 'cpu' or z.device.type != 'cpu' or truth.device.type != 'cpu':
        raise ValueError("This residual execution kernel requires detached CPU caches")
    if any(t.requires_grad for t in (f0, z, truth)):
        raise ValueError("Caches must be detached from the frozen predictor")
    if f0.dtype not in (torch.float32, torch.float64) or f0.dtype != z.dtype:
        raise ValueError("Cache prediction and bucket precision must agree")
    if truth.dtype not in (torch.float32, torch.float64):
        raise ValueError("Original-unit targets must have real floating-point values")
    if f0.shape != (len(ids),) or z.ndim != 3 or z.shape[:2] != (len(ids), 8) or truth.shape != f0.shape:
        raise ValueError("Incorrect or unaligned fixed eight-bucket cache")
    if z.shape[-1] not in (8, 12) or not all(torch.isfinite(t).all() for t in (f0, z, truth)):
        raise ValueError("Unexpected bucket dimension or nonfinite cached input")
    truth = truth.detach().clone().double().contiguous()
    cache = Cache(split, tuple(ids), f0.detach().clone().contiguous(), z.detach().clone().contiguous(),
                  truth, ((truth-mean)/sd).to(f0.dtype), float(mean), float(sd), parent_checkpoint_sha256)
    cache.digest = cache.fingerprint()
    return cache


def predict(model, cache, batch_size=256):
    cache.verify()
    model.eval()
    with torch.no_grad():
        standardized = torch.cat([predict_cached(cache.f0[i:i+batch_size], cache.z[i:i+batch_size], model)
                                  for i in range(0, len(cache.ids), batch_size)])
    if not torch.isfinite(standardized).all():
        raise NumericalFailure("Nonfinite prediction")
    # Transform once, using float64 arithmetic for original-unit metrics.
    original = standardized.double()*cache.target_sd+cache.target_mean
    return standardized.detach().clone(), original.detach().clone()


def validation(model, cache, batch_size=256):
    if cache.split != 'val':
        raise ValueError("Model selection is restricted to the validation cache")
    standardized, original = predict(model, cache, batch_size)
    mse = float((original-cache.truth).square().mean())
    if not math.isfinite(mse):
        raise NumericalFailure("Nonfinite validation loss")
    return mse, standardized, original


def source_identity():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ('execution.py', 'models.py', 'selection.py')}


def make_runtime(spec, train, val, *, epoch_limit=100, batch_size=256, patience=8):
    validate_spec(spec)
    train.verify(); val.verify()
    if train.split != 'train' or val.split != 'val' or set(train.ids) & set(val.ids):
        raise ValueError("Distinct fitting and validation caches are required")
    if (train.target_mean, train.target_sd, train.parent_checkpoint_sha256, train.z.shape[1:], train.z.dtype) != (
        val.target_mean, val.target_sd, val.parent_checkpoint_sha256, val.z.shape[1:], val.z.dtype):
        raise ValueError("The fitting and validation caches need one parent and scale")
    if train.z.shape[-1] != (12 if spec['domain'] == 'cubic' else 8):
        raise ValueError("The domain and cached bucket dimension disagree")
    if not 1 <= epoch_limit <= 100 or not 1 <= batch_size <= 256 or not 1 <= patience <= 8:
        raise ValueError("Settings exceed the proposed bounded allowance")
    model = PairResidual(spec['arm'], train.z.shape[-1], seed=spec['seed']).to(dtype=train.z.dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['rate'], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    generator = torch.Generator().manual_seed(71000+spec['seed'])
    score, standardized, original = validation(model, val, batch_size)
    if not torch.equal(standardized, val.f0):
        raise ValueError("The zero-residual checkpoint does not reproduce its baseline cache")
    return dict(spec=deepcopy(spec), model=model, optimizer=optimizer, scheduler=scheduler, generator=generator,
        settings=dict(epoch_limit=epoch_limit, batch_size=batch_size, patience=patience,
                      device='cpu', dtype=str(train.z.dtype), torch_version=str(torch.__version__)),
        sources=source_identity(), cache_bindings=dict(train=train.digest, val=val.digest),
        parent_checkpoint_sha256=train.parent_checkpoint_sha256,
        completed_epochs=0, updates=0, bad_inspections=0, termination=None,
        best_epoch=0, best_score=score, best_state=tensor_copy(model.state_dict()),
        best_prediction_std=standardized, best_prediction_original=original,
        history=[dict(epoch=0, validation_mse=score, best_epoch=0)], epoch_rows=[])


def train_epoch(runtime, train, val):
    train.verify(); val.verify()
    if runtime['cache_bindings'] != dict(train=train.digest, val=val.digest):
        raise ValueError("Runtime and cache bindings disagree")
    if runtime['sources'] != source_identity():
        raise ValueError("Execution source changed")
    if runtime['termination'] is not None:
        raise ValueError("A completed trajectory cannot take another epoch")
    model, optimizer = runtime['model'], runtime['optimizer']
    model.train()
    epoch = runtime['completed_epochs']+1
    n, batch = len(train.ids), runtime['settings']['batch_size']
    order = torch.randperm(n, generator=runtime['generator'])
    total_loss, batches = 0., []
    for start in range(0, n, batch):
        idx = order[start:start+batch]
        optimizer.zero_grad(set_to_none=True)
        predicted = predict_cached(train.f0[idx], train.z[idx], model)
        loss = (predicted-train.y_std[idx]).square().mean()
        if not torch.isfinite(loss):
            raise NumericalFailure("Nonfinite fitting loss")
        loss.backward()
        parameters = list(model.parameters())
        if any(p.grad is None for p in parameters):
            raise ValueError("An intended parameter has no gradient path")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise NumericalFailure("Nonfinite gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise NumericalFailure("Nonfinite parameter update")
        runtime['updates'] += 1
        total_loss += float(loss.detach())*len(idx)
        batches.append(len(idx))
    # Scheduler advances once per completed epoch, including its final batch.
    runtime['scheduler'].step()
    runtime['completed_epochs'] = epoch
    row = dict(epoch=epoch, updates=len(batches), batch_sizes=batches,
        training_mse_std=total_loss/n,
        order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        next_learning_rate=runtime['optimizer'].param_groups[0]['lr'])
    runtime['epoch_rows'].append(row)
    limit = runtime['settings']['epoch_limit']
    if epoch == 1 or epoch % 5 == 0 or epoch == limit:
        score, standardized, original = validation(model, val, batch)
        if score < runtime['best_score']:
            runtime.update(best_epoch=epoch, best_score=score, best_state=tensor_copy(model.state_dict()),
                           best_prediction_std=standardized, best_prediction_original=original, bad_inspections=0)
        else:
            runtime['bad_inspections'] += 1
        runtime['history'].append(dict(epoch=epoch, validation_mse=score, best_epoch=runtime['best_epoch']))
        if runtime['bad_inspections'] >= runtime['settings']['patience']:
            runtime['termination'] = 'early_stopped'
    if epoch == limit:
        runtime['termination'] = runtime['termination'] or 'epoch_limit'
    train.verify(); val.verify()
    return row


def export_runtime(runtime):
    plain = {key: deepcopy(value) for key, value in runtime.items()
             if key not in ('model', 'optimizer', 'scheduler', 'generator')}
    plain.update(model_state=tensor_copy(runtime['model'].state_dict()),
                 optimizer_state=deepcopy(runtime['optimizer'].state_dict()),
                 scheduler_state=deepcopy(runtime['scheduler'].state_dict()),
                 generator_state=runtime['generator'].get_state().clone())
    return plain


def restore_runtime(saved, train, val):
    if saved['sources'] != source_identity():
        raise ValueError("Saved runtime belongs to different execution sources")
    settings = saved['settings']
    rebuilt = make_runtime(saved['spec'], train, val, epoch_limit=settings['epoch_limit'],
                           batch_size=settings['batch_size'], patience=settings['patience'])
    if rebuilt['settings'] != settings or rebuilt['cache_bindings'] != saved['cache_bindings'] or (
        rebuilt['parent_checkpoint_sha256'] != saved['parent_checkpoint_sha256']):
        raise ValueError("Saved runtime environment, parent or cache changed")
    rebuilt['model'].load_state_dict(saved['model_state'], strict=True)
    rebuilt['optimizer'].load_state_dict(saved['optimizer_state'])
    rebuilt['scheduler'].load_state_dict(saved['scheduler_state'])
    rebuilt['generator'].set_state(saved['generator_state'])
    saved_keys = {'model_state', 'optimizer_state', 'scheduler_state', 'generator_state'}
    for key, value in saved.items():
        if key not in saved_keys:
            rebuilt[key] = deepcopy(value)
    return rebuilt


def completed_record(runtime):
    if runtime['termination'] is None:
        raise ValueError("A running trajectory cannot nominate a checkpoint")
    return dict(spec=deepcopy(runtime['spec']), status='valid',
        parent_checkpoint_sha256=runtime['parent_checkpoint_sha256'],
        best_validation_mse=runtime['best_score'], epoch_zero_validation_mse=runtime['history'][0]['validation_mse'],
        best_epoch=runtime['best_epoch'], completed_epochs=runtime['completed_epochs'],
        optimizer_updates=runtime['updates'], termination=runtime['termination'])
