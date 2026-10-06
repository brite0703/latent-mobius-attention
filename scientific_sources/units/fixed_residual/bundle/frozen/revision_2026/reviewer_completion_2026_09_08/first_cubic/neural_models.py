"""Paired pointwise encoders and explicitly specified masked set regressors."""
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"synthetic_parity"))
import parity_models as pm
core = pm.core
original, MaskedLMAHead = core.original, core.MaskedLMAHead
D, DL, M = 24, 12, 8
HEADS = ["deepsets_ln", "deepsets_plain", "deepsets_wide", "janossy2", "cp_pool",
         "lma1", "lma2", "lma3", "additive2", "additive3"]


class ExactPairHead(core.Janossy2Head):
    def forward(self, h, mask):
        pair = self.norm(self.pooled_pairs(h, mask, exact=True))
        count = torch.log1p(mask.sum(1, keepdim=True).to(h.dtype))
        return self.readout(torch.cat([pair, count], -1)).squeeze(-1)


class SetModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x, mask):
        return self.head(self.encoder(x), mask)


def count(model):
    return sum(p.numel() for p in model.parameters())


def sizes():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        budget = count(MaskedLMAHead(D, DL, M, 2, 1))
        ds = min(range(4, 257), key=lambda w: (abs(2*w*w+(D+6)*w+1-budget), w))
        jp = min(range(4, 257), key=lambda w: (abs(count(ExactPairHead(D, w))-budget), w))
        cp = min(range(4, 513), key=lambda r: (abs(count(core.CPPoolHead(D, r))-budget), r))
    return dict(reference_head_parameters=budget, deepsets_width=ds, deepsets_wide_width=2*ds,
                janossy_width=jp, cp_rank=cp)


SIZES = sizes()


def template(seed):
    torch.manual_seed(seed)
    encoder = original.NodeEncoder(12, D, dropout=0.)
    torch.manual_seed(10000+seed)
    return SetModel(encoder, MaskedLMAHead(D, DL, M, 3, 1))


def build(seed, name):
    assert name in HEADS
    reference = template(seed)
    torch.manual_seed(seed)
    encoder = original.NodeEncoder(12, D, dropout=0.)
    encoder.load_state_dict(reference.encoder.state_dict())
    torch.manual_seed(10000+seed)
    if name.startswith("lma") or name.startswith("additive"):
        k = int(name[-1])
        head = MaskedLMAHead(D, DL, M, k, 1, feature_mode="additive" if name.startswith("additive") else "product")
        ref = reference.head.state_dict()
        state = head.state_dict()
        for key in state:
            state[key] = ref[key][:k].clone() if key.endswith("order_gates") else ref[key].clone()
        head.load_state_dict(state)
    elif name.startswith("deepsets"):
        width = SIZES["deepsets_wide_width" if name == "deepsets_wide" else "deepsets_width"]
        head = original.DeepSetsHead(D, width)
        if name != "deepsets_ln":
            head.rho[0] = nn.Identity()
    elif name == "janossy2":
        head = ExactPairHead(D, SIZES["janossy_width"])
    else:
        head = core.CPPoolHead(D, SIZES["cp_rank"])
    return SetModel(encoder, head)


def sources():
    return list(dict.fromkeys([Path(__file__).resolve()]+pm.source_paths()))
