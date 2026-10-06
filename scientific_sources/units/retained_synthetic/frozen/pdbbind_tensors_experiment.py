import argparse
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Data
# ==========================================
class TensorSetDataset(Dataset):
    """Dataset wrapper for .pt splits with X, mask, y."""

    def __init__(self, path: str):
        blob = torch.load(path)
        self.X = blob["X"].float()
        self.mask = blob["mask"].bool()
        self.y = blob["y"].float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


def load_splits(data_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
    train = TensorSetDataset(os.path.join(data_dir, "pdbbind_train.pt"))
    val = TensorSetDataset(os.path.join(data_dir, "pdbbind_val.pt"))
    test = TensorSetDataset(os.path.join(data_dir, "pdbbind_test.pt"))
    return train, val, test


# ==========================================
# Encoder (shared architecture across heads)
# ==========================================
class NodeEncoder(nn.Module):
    """
    Simple 2-layer per-node encoder (no edges) to produce node embeddings.
    This keeps the "encoder" identical across heads; only the aggregator differs.
    """
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# Aggregator heads
# ==========================================
class DeepSetsHead(nn.Module):
    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
        )

    def forward(self, h, mask):
        z = self.phi(h) * mask.unsqueeze(-1)
        z = z.sum(dim=1)
        return self.rho(z).squeeze(-1)


class TransformerHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, depth: int = 1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h, mask):
        padding_mask = ~mask  # True where pad
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)


class LatentMobiusAttention(nn.Module):
    def __init__(self, d_model: int, num_buckets_M: int, order_k: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.M = num_buckets_M
        self.k = min(order_k, num_buckets_M)
        self.d_latent = d_latent
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_latent)
        self.W_H = nn.Linear(d_model, num_buckets_M)
        self.interaction_projs = nn.ModuleList()
        for r in range(1, self.k + 1):
            self.interaction_projs.append(nn.Linear(d_latent, r * d_latent))
        self.interaction_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_latent),
                nn.Linear(d_latent, d_latent),
                nn.GELU(),
                nn.Linear(d_latent, d_model),
            ) for _ in range(self.k)
        ])
        self.order_gates = nn.Parameter(torch.ones(self.k))
        self.W_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        for r in range(1, self.k + 1):
            combos = torch.combinations(torch.arange(self.M), r)
            self.register_buffer(f"combos_{r}", combos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        u = self.W_H(K)
        pi = F.softmax(u, dim=-1)
        Z = torch.bmm(pi.transpose(1, 2), V)
        interaction_outputs = []
        for r in range(1, self.k + 1):
            proj = self.interaction_projs[r - 1](Z)
            parts = proj.chunk(r, dim=-1)
            combos_r = getattr(self, f"combos_{r}")
            gathered = []
            for i in range(r):
                idx_i = combos_r[:, i]
                gathered_i = parts[i][:, idx_i, :]
                gathered.append(gathered_i)
            gamma_r = gathered[0]
            for i in range(1, r):
                gamma_r = gamma_r * gathered[i]
            feat_r = self.interaction_mlps[r - 1](gamma_r)
            feat_r = self.order_gates[r - 1] * feat_r
            interaction_outputs.append(feat_r)
        Memory = torch.cat(interaction_outputs, dim=1)
        scores = torch.bmm(Q, Memory.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        readout = torch.bmm(attn_weights, Memory)
        out = self.W_out(readout)
        return self.layer_norm(x + out)


class LMAHead(nn.Module):
    def __init__(self, d_model: int, d_latent: int, num_buckets: int, order_k: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([
            LatentMobiusAttention(d_model, num_buckets, order_k, d_latent)
            for _ in range(depth)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h, mask):
        for layer in self.layers:
            h = layer(h)
            h = h + self.ffn(h)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)


# ==========================================
# Training / Evaluation
# ==========================================
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    diff = yt - yp
    rmse = math.sqrt(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mean_y = yt.mean()
    mean_p = yp.mean()
    var_y = np.mean((yt - mean_y) ** 2)
    var_p = np.mean((yp - mean_p) ** 2)
    denom = math.sqrt(var_y * var_p)
    if denom > 1e-12:
        r = float(np.mean((yt - mean_y) * (yp - mean_p)) / denom)
    else:
        r = float("nan")
    return {"rmse": rmse, "mae": mae, "pearson": r}


def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            out = model(x, mask)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    y_true = np.concatenate(trues, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return metrics(y_true, y_pred)


def train_model(model, train_loader, val_loader, epochs: int, lr: float, patience: int):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_rmse = float("inf")
    best_state = None
    wait = 0
    for ep in range(epochs):
        model.train()
        for x, mask, y in train_loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x, mask)
            loss = loss_fn(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        val_metrics = evaluate(model, val_loader)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Ep {ep+1:03d} | Val RMSE={val_metrics['rmse']:.4f}")
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


# ==========================================
# Runner
# ==========================================
def run(args):
    set_seed(args.seed)
    train_ds, val_ds, test_ds = load_splits(args.data_dir)
    sample_x, sample_mask, _ = train_ds[0]
    d_in = sample_x.shape[-1]
    print(f"[INFO] Input feature dim: {d_in}")

    selected = set(m.strip().lower() for m in args.models.split(","))

    def build_encoder():
        return NodeEncoder(d_in=d_in, d_model=args.d_model, dropout=0.1)

    def build_model(head):
        class Model(nn.Module):
            def __init__(self, enc, hd):
                super().__init__()
                self.enc = enc
                self.hd = hd

            def forward(self, x, mask):
                h = self.enc(x)
                return self.hd(h, mask)

        return Model(build_encoder(), head)

    # Prepare loaders once (split fixed)
    def make_loaders():
        return (
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=args.batch_size),
            DataLoader(test_ds, batch_size=args.batch_size),
        )

    # Collect metrics across seeds
    agg_results: Dict[str, List[Dict[str, float]]] = {}

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Report shared encoder param count and per-head param counts (capacity fairness)
    enc_tmp = build_encoder()
    enc_params = count_params(enc_tmp)
    head_param_lookup: Dict[str, int] = {}
    if "deepsets" in selected or "all" in selected:
        head_param_lookup["DeepSets"] = count_params(DeepSetsHead(d_model=args.d_model, d_latent=args.d_latent))
    if "settx" in selected or "all" in selected:
        head_param_lookup["TransformerHead"] = count_params(TransformerHead(d_model=args.d_model, depth=1))
    for k in [1, 2, 3]:
        key = f"LMA (k={k})"
        if f"lma{k}" in selected or "all" in selected:
            buckets = args.num_buckets_k3 if (k == 3 and args.num_buckets_k3 is not None) else args.num_buckets
            head_param_lookup[key] = count_params(
                LMAHead(
                    d_model=args.d_model,
                    d_latent=args.d_latent,
                    num_buckets=buckets,
                    order_k=k,
                    depth=args.lma_depth,
                )
            )

    print(f"[INFO] Encoder params (shared across heads): {enc_params}")
    for name, cnt in head_param_lookup.items():
        print(f"[INFO] Head params [{name}]: {cnt}")

    for seed_idx in range(args.n_seeds):
        cur_seed = args.seed + seed_idx
        set_seed(cur_seed)
        train_loader, val_loader, test_loader = make_loaders()
        print(f"\n===== Seed {cur_seed} =====")

        if "deepsets" in selected or "all" in selected:
            head = DeepSetsHead(d_model=args.d_model, d_latent=args.d_latent)
            model = build_model(head)
            model = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience)
            metrics_ds = evaluate(model, test_loader)
            agg_results.setdefault("DeepSets", []).append(metrics_ds)
            print(f"[DeepSets][seed {cur_seed}] {metrics_ds}")

        if "settx" in selected or "all" in selected:
            head = TransformerHead(d_model=args.d_model, depth=1)
            model = build_model(head)
            model = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience)
            metrics_tx = evaluate(model, test_loader)
            agg_results.setdefault("TransformerHead", []).append(metrics_tx)
            print(f"[TransformerHead][seed {cur_seed}] {metrics_tx}")

        for k in [1, 2, 3]:
            key = f"LMA (k={k})"
            if f"lma{k}" in selected or "all" in selected:
                buckets = args.num_buckets_k3 if (k == 3 and args.num_buckets_k3 is not None) else args.num_buckets
                head = LMAHead(
                    d_model=args.d_model,
                    d_latent=args.d_latent,
                    num_buckets=buckets,
                    order_k=k,
                    depth=args.lma_depth,
                )
                model = build_model(head)
                model = train_model(model, train_loader, val_loader, args.epochs, args.lma_lr, args.patience)
                metrics_lma = evaluate(model, test_loader)
                agg_results.setdefault(key, []).append(metrics_lma)
                print(f"[{key}][seed {cur_seed}] {metrics_lma}")

    print("\nSummary (mean ± std over seeds):")
    for name, runs in agg_results.items():
        if not runs:
            continue
        for metric in ["rmse", "mae", "pearson"]:
            vals = np.array([r[metric] for r in runs], dtype=np.float32)
            print(f"- {name} {metric}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head comparison on preprocessed PDBbind tensors.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with pdbbind_train/val/test.pt.")
    parser.add_argument("--models", type=str, default="all", help="deepsets,settx,lma1,lma2,lma3,all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; seeds will be seed..seed+n_seeds-1.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds to run per model.")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-latent", type=int, default=32)
    parser.add_argument("--num-buckets", type=int, default=12)
    parser.add_argument("--num-buckets-k3", type=int, default=8)
    parser.add_argument("--lma-depth", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lma-lr", type=float, default=3e-4)
    args = parser.parse_args()
    run(args)
