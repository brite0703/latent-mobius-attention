"""Mask-aware research copy of the submitted tensor implementation.

Original parameter names and mathematical operations are preserved for the
product mode. The source manuscript and original experiment file are untouched.
This copy does not repair or claim the manuscript's expressivity theorems.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parents[1] / "pdbbind_tensors_experiment.py"
spec = importlib.util.spec_from_file_location("lma_submitted_tensor", SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class MaskedLatentMobiusAttention(original.LatentMobiusAttention):
    def __init__(self, *args, feature_mode="product", **kwargs):
        super().__init__(*args, **kwargs)
        if feature_mode not in ("product", "additive"):
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        self.last_routing = None
        self.last_mask = None

    def forward(self, x, mask):
        # Values must be masked after their biased projection. Masking x alone
        # or masking only the final pooled output leaves padded values in Z.
        q = self.W_q(x)
        pi = F.softmax(self.W_H(self.W_k(x)), dim=-1)
        valid = mask.unsqueeze(-1).to(x.dtype)
        values = self.W_v(x) * valid
        z = torch.bmm(pi.transpose(1, 2), values)
        self.last_routing, self.last_mask = pi, mask
        outputs = []
        for r in range(1, self.k + 1):
            parts = self.interaction_projs[r-1](z).chunk(r, dim=-1)
            combos = getattr(self, f"combos_{r}")
            gathered = [parts[i][:, combos[:, i], :] for i in range(r)]
            gamma = gathered[0]
            for part in gathered[1:]:
                gamma = gamma * part if self.feature_mode == "product" else gamma + part
            if self.feature_mode == "additive":
                gamma = gamma / r
            outputs.append(self.order_gates[r-1] * self.interaction_mlps[r-1](gamma))
        memory = torch.cat(outputs, dim=1)
        scores = torch.bmm(q, memory.transpose(1, 2)) / self.d_model**0.5
        readout = torch.bmm(F.softmax(scores, dim=-1), memory)
        return self.layer_norm(x + self.W_out(readout)) * valid

    def balance_loss(self):
        if self.last_routing is None:
            raise RuntimeError("Call forward before balance_loss")
        mask = self.last_mask.unsqueeze(-1).to(self.last_routing.dtype)
        mass = (self.last_routing * mask).sum(dim=(0, 1)) / mask.sum().clamp_min(1)
        # This is a load-balancing penalty, not a collision or informativeness
        # guarantee; uniform routing also attains its minimum.
        return self.M * ((mass - 1/self.M)**2).sum()


class MaskedLMAHead(original.LMAHead):
    def __init__(self, d_model, d_latent, num_buckets, order_k, depth, feature_mode="product"):
        super().__init__(d_model, d_latent, num_buckets, order_k, depth)
        self.layers = nn.ModuleList([
            MaskedLatentMobiusAttention(d_model, num_buckets, order_k, d_latent,
                                       feature_mode=feature_mode)
            for _ in range(depth)
        ])

    def forward(self, h, mask):
        for layer in self.layers:
            h = layer(h, mask)
            h = (h + self.ffn(h)) * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        return self.head(self.norm(pooled)).squeeze(-1)

    def balance_loss(self):
        return torch.stack([layer.balance_loss() for layer in self.layers]).mean()


class SetRegressor(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x, mask):
        return self.head(self.encoder(x), mask)
