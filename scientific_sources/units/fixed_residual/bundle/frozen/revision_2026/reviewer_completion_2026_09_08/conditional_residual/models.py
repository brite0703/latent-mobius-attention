"""Conditional residual controls; no data loading, training or CUDA at import."""
from itertools import combinations
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("product", "additive", "pair_mlp")


def dimensions(d):
    if not isinstance(d, int) or d < 2:
        raise ValueError("The bucket dimension must be an integer of at least two")
    width = 2*d
    leg_count = 2*(d+1)*width
    hidden = min(range(1, 257), key=lambda h: (abs((2*d+width+1)*h+width-leg_count), h))
    common = 2*width + width*(width+1) + width+1
    return dict(bucket_dimension=d, width=width, pair_mlp_hidden=hidden,
                product_parameters=leg_count+common, additive_parameters=leg_count+common,
                pair_mlp_parameters=(2*d+width+1)*hidden+width+common)


def seeded(seed, factory):
    # Construction is CPU-only and restores the caller's CPU random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


class PairResidual(nn.Module):
    """Scalar correction in the frozen predictor's standardized target units."""
    def __init__(self, arm, d, buckets=8, seed=42):
        super().__init__()
        if arm not in ARMS or not isinstance(buckets, int) or buckets < 2:
            raise ValueError((arm, buckets))
        self.arm, self.d, self.buckets = arm, d, buckets
        self.sizes = dimensions(d)
        width, hidden = self.sizes["width"], self.sizes["pair_mlp_hidden"]
        self.register_buffer("pairs", torch.tensor(list(combinations(range(buckets), 2)), dtype=torch.long))
        if arm == "pair_mlp":
            self.pair_map = seeded(62000+seed, lambda: nn.Sequential(
                nn.Linear(2*d, hidden), nn.GELU(), nn.Linear(hidden, width)))
        else:
            self.legs = seeded(61000+seed, lambda: nn.Linear(d, 2*width))
        # These parameters have identical initial values in all three arms.
        self.norm = seeded(63000+seed, lambda: nn.LayerNorm(width, eps=1e-5))
        self.post = seeded(64000+seed, lambda: nn.Linear(width, width))
        self.output = seeded(65000+seed, lambda: nn.Linear(width, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        assert self.parameter_count() == self.sizes[arm+"_parameters"]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, z):
        if z.ndim != 3 or z.shape[1:] != (self.buckets, self.d):
            raise ValueError("Expected [records, fixed buckets, bucket dimension]")
        if self.arm == "pair_mlp":
            joined = torch.cat([z[:, self.pairs[:, 0]], z[:, self.pairs[:, 1]]], -1)
            interaction = self.pair_map(joined)
        else:
            left, right = self.legs(z).chunk(2, -1)
            a, b = left[:, self.pairs[:, 0]], right[:, self.pairs[:, 1]]
            interaction = a*b if self.arm == "product" else (a+b)/2
        hidden = F.gelu(self.post(self.norm(interaction)))
        return self.output(hidden.mean(1)).squeeze(-1)


class FrozenLmaFeatures(nn.Module):
    """Read the exact bucket tensor used by an unchanged one-layer k=1 model.

    A temporary native projection hook avoids reimplementing its encoder,
    routing, contact fusion, masking, or scalar prediction. Only constructed
    local models should be passed here: this object disables their gradients.
    """
    def __init__(self, baseline, layer_path):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.layer_path = layer_path
        layer = self.baseline.get_submodule(layer_path)
        if layer.k != 1 or len(layer.interaction_projs) != 1:
            raise ValueError("Only a selected first-order predictor is eligible")

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, *args, **kwargs):
        self.baseline.eval()
        captured = []

        def capture(module, inputs):
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise ValueError("Unexpected native first-order projection input")
            captured.append(inputs[0].detach().clone())

        layer = self.baseline.get_submodule(self.layer_path)
        handle = layer.interaction_projs[0].register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                prediction = self.baseline(*args, **kwargs).detach().clone()
        finally:
            handle.remove()
        if len(captured) != 1 or prediction.ndim != 1 or len(prediction) != len(captured[0]):
            raise ValueError("Expected exactly one native layer call and one scalar per record")
        return prediction, captured[0]


def predict_cached(baseline_prediction, buckets, residual):
    if baseline_prediction.requires_grad or buckets.requires_grad:
        raise ValueError("A residual fit requires detached frozen features and predictions")
    correction = residual(buckets)
    if baseline_prediction.shape != correction.shape:
        raise ValueError("The cached baseline predictions must be record-aligned")
    return baseline_prediction+correction
