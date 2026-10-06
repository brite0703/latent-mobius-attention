"""Independent, declared readout adaptations for the first reviewer study.

Published defining equations are recorded in protocol.md. These are not
reproductions of published benchmark results. Frozen local sources are imported
without modifying them.
"""
from pathlib import Path
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lma_revision import original, MaskedLMAHead
from pdbbind_rerun import GraphEncoder, AffinityModel, PMAHead

D_MODEL = 16
D_LATENT = 8
BUCKETS = 8
HEADS = ["mean_count", "pma", "deepsets", "deepsets_wide",
         "janossy2", "dcnv2", "cp_pool", "lma1", "lma2", "lma3", "additive2"]
DEPTH3_HEADS = ["mean_count", "deepsets_wide", "cp_pool", "lma1", "lma2", "additive2"]


class CountMeanHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d+1, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, mask):
        count = mask.sum(1, keepdim=True)
        mean = (h * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1)
        z = torch.cat([self.norm(mean), torch.log1p(count.to(h.dtype))], -1)
        return self.net(z).squeeze(-1)


class Janossy2Head(nn.Module):
    """Ordered distinct-pair average; MC training, exact evaluation.

    Pair features are permutation-sensitive. The first affine map is evaluated
    in two parts, algebraically equal to an affine map on concatenated inputs.
    For n=1 the sole permuted sequence is padded on the right with one zero
    vector, as in Definition 2.2. The empty-set extension uses the pair (0,0).
    The molecular data contain at least four atoms per record.
    """
    def __init__(self, d, width, samples=64):
        super().__init__()
        self.d, self.width, self.samples = d, width, samples
        self.pair_first = nn.Linear(2*d, width)
        self.pair_second = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Sequential(nn.Linear(width+1, width), nn.GELU(), nn.Linear(width, 1))

    def pair_values(self, left, right):
        a = F.linear(left, self.pair_first.weight[:, :self.d], self.pair_first.bias)
        b = F.linear(right, self.pair_first.weight[:, self.d:])
        return F.gelu(self.pair_second(F.gelu(a+b)))

    def pooled_pairs(self, h, mask, exact=None):
        exact = (not self.training) if exact is None else exact
        if h.shape[1] < 2:
            h = F.pad(h, (0, 0, 0, 2-h.shape[1]))
            mask = F.pad(mask, (0, 2-mask.shape[1]))
        # Compact valid values without imposing a data-dependent canonical order.
        indices = torch.argsort((~mask).long(), dim=1, stable=True)
        packed = torch.gather(h * mask.unsqueeze(-1), 1, indices.unsqueeze(-1).expand_as(h))
        count = mask.sum(1)
        effective = count.clamp_min(2)
        if not exact:
            i = (torch.rand(len(h), self.samples, device=h.device) * effective[:, None]).long()
            j0 = (torch.rand(len(h), self.samples, device=h.device) * (effective-1)[:, None]).long()
            j = j0 + (j0 >= i)
            batch = torch.arange(len(h), device=h.device)[:, None]
            sampled = self.pair_values(packed[batch, i], packed[batch, j]).mean(1)
            short = self.pair_values(packed[:, 0], torch.zeros_like(packed[:, 0]))
            return torch.where((count < 2)[:, None], short, sampled)
        n = h.shape[1]
        values = self.pair_values(packed[:, :, None, :], packed[:, None, :, :])
        valid = torch.arange(n, device=h.device)[None, :] < effective[:, None]
        pair_mask = valid[:, :, None] & valid[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=h.device)[None]
        average = (values * pair_mask.unsqueeze(-1)).sum((1, 2)) / (effective*(effective-1))[:, None]
        short = self.pair_values(packed[:, 0], torch.zeros_like(packed[:, 0]))
        return torch.where((count < 2)[:, None], short, average)

    def forward(self, h, mask):
        pooled = self.norm(self.pooled_pairs(h, mask))
        count = torch.log1p(mask.sum(1, keepdim=True).to(h.dtype))
        return self.readout(torch.cat([pooled, count], -1)).squeeze(-1)


class DCNV2Head(nn.Module):
    """Two full-matrix cross layers and a parallel ReLU deep branch.

    x0 is an invariant, explicitly specified input: normalized masked mean and
    log(1+n). No nonlinearity is inserted inside the DCN-V2 cross equation.
    """
    def __init__(self, d, width):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.cross = nn.ModuleList(nn.Linear(d+1, d+1) for _ in range(2))
        self.deep = nn.Sequential(nn.Linear(d+1, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU())
        self.out = nn.Linear(d+1+width, 1)

    def representation(self, h, mask):
        count = mask.sum(1, keepdim=True)
        mean = (h*mask.unsqueeze(-1)).sum(1)/count.clamp_min(1)
        return torch.cat([self.norm(mean), torch.log1p(count.to(h.dtype))], -1)

    def forward(self, h, mask):
        x0 = self.representation(h, mask)
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.out(torch.cat([x, self.deep(x0)], -1)).squeeze(-1)


class CPPoolHead(nn.Module):
    """CP global readout plus the paper's additive low-order branch.

    Defining CP operation: relu(M tanh(prod_i(W^T [h_i;1]))).
    Masked factors must be ONE. Tanh is applied after multiplication. Product
    and tanh use float64 before casting back; affine projections remain in the
    model dtype. No clipping, log-product surrogate, or root normalization.
    """
    def __init__(self, d, rank):
        super().__init__()
        self.rank = rank
        self.factor = nn.Linear(d, rank)
        self.mix = nn.Linear(rank, d, bias=False)
        self.low_order = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Declared local initialization; no claim that the source used this.
        nn.init.normal_(self.factor.weight, std=0.1/math.sqrt(d))
        nn.init.ones_(self.factor.bias)
        self.last_product = None

    def cp_features(self, h, mask):
        factors = self.factor(h).masked_fill(~mask.unsqueeze(-1), 1.)
        product = factors.double().prod(dim=1)
        self.last_product = product.detach()
        bounded = torch.tanh(product).to(h.dtype)
        return F.relu(self.mix(bounded))

    def forward(self, h, mask):
        low = F.relu(self.low_order((h*mask.unsqueeze(-1)).sum(1)))
        return self.readout(self.norm(self.cp_features(h, mask)+low)).squeeze(-1)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def head_sizes():
    # Parameter-only selection under an isolated RNG context; never fit data.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        budget = parameter_count(MaskedLMAHead(D_MODEL, D_LATENT, BUCKETS, 2, 1))
        ds_width = min(range(4, 129), key=lambda w: (abs((2*w*w+(D_MODEL+6)*w+1)-budget), w))
        jp_width = min(range(4, 129), key=lambda w: (abs(parameter_count(Janossy2Head(D_MODEL, w))-budget), w))
        dcn_width = min(range(4, 129), key=lambda w: (abs(parameter_count(DCNV2Head(D_MODEL, w))-budget), w))
        cp_rank = min(range(4, 257), key=lambda r: (abs(parameter_count(CPPoolHead(D_MODEL, r))-budget), r))
    return dict(reference_head_parameters=budget, deepsets_width=ds_width,
                deepsets_wide_width=2*ds_width, janossy_width=jp_width,
                dcn_width=dcn_width, cp_rank=cp_rank)


SIZES = head_sizes()


def build_model(seed, name, depth, d_in=53):
    if name not in HEADS or depth not in (1, 3):
        raise ValueError((name, depth))
    torch.manual_seed(seed)
    encoder = GraphEncoder(d_in, D_MODEL, depth)
    torch.manual_seed(10000+seed)
    if name == "mean_count":
        head = CountMeanHead(D_MODEL)
    elif name == "pma":
        head = PMAHead(D_MODEL)
    elif name in ("deepsets", "deepsets_wide"):
        head = original.DeepSetsHead(D_MODEL, SIZES[name+"_width"])
    elif name == "janossy2":
        head = Janossy2Head(D_MODEL, SIZES["janossy_width"])
    elif name == "dcnv2":
        head = DCNV2Head(D_MODEL, SIZES["dcn_width"])
    elif name == "cp_pool":
        head = CPPoolHead(D_MODEL, SIZES["cp_rank"])
    else:
        order = 2 if name == "additive2" else int(name[-1])
        head = MaskedLMAHead(D_MODEL, D_LATENT, BUCKETS, order, 1,
                             feature_mode="additive" if name == "additive2" else "product")
    return AffinityModel(encoder, head, "gcn")
