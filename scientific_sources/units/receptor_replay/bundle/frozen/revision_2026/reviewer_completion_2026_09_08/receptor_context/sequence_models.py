"""Declared ligand/sequence late fusion; no retained fit is performed here."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
sys.path.insert(0, str(REVISION))
from lma_revision import MaskedLMAHead
from pdbbind_rerun import GraphEncoder
from neural_reviewer_study_2026_09_07.models import CPPoolHead
from sequence_encoder import ChainSequenceEncoder

HEADS = ("deepsets_plain70", "lma1", "lma2", "additive2", "cp_pool")
SETTINGS = ("ligand_only", "ligand_sequence", "sequence_only")


def count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def seeded(seed, factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


class PlainSumEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(16, 70), nn.GELU(), nn.Linear(70, 70), nn.GELU())
        self.project = nn.Linear(70, 16)

    def forward(self, h, mask):
        return self.project((self.phi(h)*mask.unsqueeze(-1)).sum(1))


def lma_embedding(order, feature_mode="product"):
    module = MaskedLMAHead(16, 8, 8, order, 1, feature_mode=feature_mode)
    module.head = nn.Identity()
    return module


def cp_embedding(rank):
    module = CPPoolHead(16, rank)
    module.readout = nn.Identity()
    return module


def cp_rank():
    def choose():
        reference = count(lma_embedding(2))
        # CP factor:17*r; mix:16*r; low-order:256; LayerNorm:32.
        rank = min(range(4, 257), key=lambda r: (abs(33*r+288-reference), r))
        assert count(cp_embedding(rank)) == 33*rank+288
        return rank
    return seeded(2026090833, choose)


CP_RANK = cp_rank()


def build_pool(seed, head):
    if head not in HEADS:
        raise ValueError(head)
    if head == "deepsets_plain70":
        return seeded(42000+seed, PlainSumEmbedding)
    if head == "cp_pool":
        return seeded(42000+seed, lambda: cp_embedding(CP_RANK))
    template = seeded(42000+seed, lambda: lma_embedding(2))
    order = 1 if head == "lma1" else 2
    module = seeded(42000+seed, lambda: lma_embedding(order, "additive" if head == "additive2" else "product"))
    template_state = template.state_dict()
    state = module.state_dict()
    retained = {}
    for key, value in state.items():
        source = template_state[key]
        if key == "layers.0.order_gates":
            source = source[:order]
        assert value.shape == source.shape, key
        retained[key] = source.clone()
    module.load_state_dict(retained, strict=True)
    return module


class SequenceAffinityModel(nn.Module):
    def __init__(self, seed, setting, head):
        super().__init__()
        if setting not in SETTINGS or (setting == "sequence_only") != (head == "none"):
            raise ValueError((setting, head))
        self.setting, self.head_name = setting, head
        self.encoder = self.pool = self.ligand_projection = None
        self.sequence = self.sequence_projection = None
        if setting != "sequence_only":
            self.encoder = seeded(41000+seed, lambda: GraphEncoder(53, 16, 1))
            self.pool = build_pool(seed, head)
            self.ligand_projection = seeded(44000+seed, lambda: nn.Linear(17, 64, bias=False))
        if setting != "ligand_only":
            self.sequence = seeded(43000+seed, ChainSequenceEncoder)
            self.sequence_projection = seeded(45000+seed, lambda: nn.Linear(98, 64, bias=False))
        self.fusion_bias = nn.Parameter(torch.zeros(64))
        self.regression = seeded(46000+seed, lambda: nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)))

    def forward(self, x=None, mask=None, adj=None, sequence_batch=None):
        if self.encoder is None:
            if sequence_batch is None:
                raise ValueError("Sequence input is required")
            records = sequence_batch[3]
        else:
            if x is None or mask is None or adj is None or x.shape[:2] != mask.shape:
                raise ValueError("Aligned ligand tensors are required")
            records = len(x)
        fused = self.fusion_bias[None].expand(records, -1)
        if self.encoder is not None:
            h = self.encoder(x, mask, adj)
            ligand = self.pool(h, mask)
            if ligand.shape != (records, 16):
                raise ValueError("The ligand readout must return 16 coordinates")
            context = torch.cat([ligand, torch.log1p(mask.sum(1, keepdim=True).to(ligand.dtype))], -1)
            fused = fused+self.ligand_projection(context)
        if self.sequence is not None:
            if sequence_batch is None or sequence_batch[3] != records:
                raise ValueError("Aligned sequence records are required")
            fused = fused+self.sequence_projection(self.sequence(*sequence_batch))
        return self.regression(F.gelu(fused)).squeeze(-1)

    def parameter_counts(self):
        return dict(total=count(self), ligand_encoder=count(self.encoder), ligand_pool=count(self.pool),
                    ligand_projection=count(self.ligand_projection), sequence_encoder=count(self.sequence),
                    sequence_projection=count(self.sequence_projection), shared_fusion_bias=self.fusion_bias.numel(),
                    shared_regression=count(self.regression))


def configurations():
    return [(setting, head) for setting in SETTINGS[:2] for head in HEADS]+[("sequence_only", "none")]


def build_model(seed, setting, head):
    return SequenceAffinityModel(seed, setting, head)
