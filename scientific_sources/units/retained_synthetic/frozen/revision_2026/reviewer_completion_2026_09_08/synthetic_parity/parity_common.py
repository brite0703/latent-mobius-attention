"""Deterministic data, canonical inference, metrics and record utilities."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LENGTHS = [10, 20, 40, 80]
SEEDS = list(range(100, 110))
LRS = [.0003, .001]
EPOCHS, BATCH, PATIENCE = 100, 256, 8


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".pending")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def generate(n, seed):
    rng = np.random.default_rng(2026090803+1000*n+seed)
    x = rng.integers(0, 2, size=(6000, n), dtype=np.uint8)
    # dtype is part of the reproducible generator specification.
    return dict(x=x, y=(x.sum(1, dtype=np.int64) % 2),
                count=x.sum(1, dtype=np.int64))


SLICES = dict(train=slice(0, 4800), val=slice(4800, 5400), test=slice(5400, 6000))


def data_path(n, seed):
    return HERE/"data"/f"n{n}_seed{seed}.npz"


def load_data(n, seed, split, device="cpu"):
    # The split loader returns only the requested rows. Test is never used by fitting.
    sl = SLICES[split]
    with np.load(data_path(n, seed)) as z:
        return dict(x=torch.as_tensor(z["x"][sl].copy(), dtype=torch.float32, device=device),
                    y=torch.as_tensor(z["y"][sl].copy(), dtype=torch.long, device=device),
                    count=z["count"][sl].copy())


def canonical(n, device="cpu", dtype=torch.float32):
    return (torch.arange(n, device=device)[None, :] <
            torch.arange(n+1, device=device)[:, None]).to(dtype)


@torch.no_grad()
def canonical_logits(model, n):
    model.eval()
    p = next(model.parameters())
    # One fixed batch at every validation/test evaluation and checkpoint audit.
    out = model(canonical(n, p.device, p.dtype))
    if out.shape != (n+1, 2) or not bool(torch.isfinite(out).all()):
        raise FloatingPointError("Nonfinite or malformed canonical logits")
    return out.double().cpu().numpy()


def weighted_auc(y, scores, weights):
    y, scores, weights = np.asarray(y), np.asarray(scores), np.asarray(weights, dtype=np.float64)
    positive, negative = y == 1, y == 0
    den = weights[positive].sum()*weights[negative].sum()
    if den == 0:
        return None
    a, b = scores[positive, None], scores[None, negative]
    terms = (a > b).astype(np.float64) + .5*(a == b)
    return float(np.sum(terms*weights[positive, None]*weights[None, negative])/den)


def metrics(y, logits, weights=None):
    y, logits = np.asarray(y, dtype=np.int64), np.asarray(logits, dtype=np.float64)
    assert logits.shape == (len(y), 2) and np.isfinite(logits).all()
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    assert np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0
    score = logits[:, 1]-logits[:, 0]
    ce = np.logaddexp(0., (1-2*y)*score)
    return dict(accuracy=float(np.sum(w*((score > 0) == y))/w.sum()),
                auc=weighted_auc(y, score, w), cross_entropy=float(np.sum(w*ce)/w.sum()))


def population_metrics(n, logits):
    counts = np.arange(n+1)
    weights = np.asarray([math.comb(n, c)/2**n for c in counts], dtype=np.float64)
    result = metrics(counts % 2, logits, weights)
    result["success_at_099"] = result["accuracy"] >= .99
    return result


def count_lookup(n, training):
    counts = np.asarray(training["count"])
    y = training["y"].detach().cpu().numpy()
    totals = np.bincount(counts, minlength=n+1)
    positive = np.bincount(counts, weights=y, minlength=n+1)
    p = (positive+1)/(totals+2)
    return np.column_stack([np.log1p(-p), np.log(p)]), totals


def cp_diagnostics(model, x):
    # Read-only diagnostic at a fixed canonical batch, independent of selection.
    if not hasattr(model, "head") or not hasattr(model.head, "factor"):
        return None
    with torch.no_grad():
        h = model.embedding(x.unsqueeze(-1))
        factors = model.head.factor(h)
        product = factors.double().prod(1)
        bounded = product.tanh()
        finite = torch.isfinite(product)
        return dict(product_entries=product.numel(), finite_entries=int(finite.sum()),
                    maximum_absolute_product=float(product[finite].abs().max()) if bool(finite.any()) else None,
                    exact_zero_products=int((product == 0).sum()),
                    tanh_derivative_below_1e_6=int(((1-bounded.square()) < 1e-6).sum()),
                    tanh_exactly_abs_one=int((bounded.abs() == 1).sum()),
                    exact_zero_factors=int((factors == 0).sum()))
