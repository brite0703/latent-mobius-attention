"""Source parity networks and explicitly named, parameter-matched adaptations."""
from pathlib import Path
import importlib.util
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
ROOT = REVISION.parent
CORE = REVISION / "neural_reviewer_study_2026_09_07"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = import_file("available_parity_source", ROOT / "lma.py")
core = import_file("parity_core_readouts", CORE / "models.py")
D, DL, M = 32, 16, 8
HEADS = ["deepsets_ln", "deepsets_plain", "deepsets_wide", "transformer",
         "janossy2", "cp_pool", "lma1", "lma2", "lma3", "lma3_clip1"]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


class PairClassifier(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Linear(1, D)
        self.head = core.Janossy2Head(D, width)
        self.head.readout[-1] = nn.Linear(width, 2)

    def forward(self, x):
        # Exactly all ordered, distinct-index pairs for this binary domain.
        # Pair types can coincide in value while their indices remain distinct.
        zero_one = torch.tensor([[0.], [1.]], dtype=x.dtype, device=x.device)
        emb = self.embedding(zero_one)
        left, right = emb[[0, 0, 1, 1]], emb[[0, 1, 0, 1]]
        values = self.head.pair_values(left, right)
        n = x.shape[1]
        assert n >= 2
        n1 = x.sum(1)
        n0 = n - n1
        multiplicity = torch.stack([n0*(n0-1), n0*n1, n1*n0, n1*(n1-1)], 1)
        pooled = multiplicity @ values / (n*(n-1))
        count = x.new_full((len(x), 1), float(n)).log1p()
        return self.head.readout(torch.cat([self.head.norm(pooled), count], 1))

    def literal(self, x):
        h = self.embedding(x.unsqueeze(-1))
        mask = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        pooled = self.head.pooled_pairs(h, mask, exact=True)
        count = x.new_full((len(x), 1), float(x.shape[1])).log1p()
        return self.head.readout(torch.cat([self.head.norm(pooled), count], 1))


class CPClassifier(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.embedding = nn.Linear(1, D)
        self.head = core.CPPoolHead(D, rank)
        self.head.readout[-1] = nn.Linear(D, 2)

    def forward(self, x):
        h = self.embedding(x.unsqueeze(-1))
        return self.head(h, torch.ones(x.shape, dtype=torch.bool, device=x.device))


def sizes():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        budget = parameter_count(source.LMANetwork(10, D, M, 2, DL, 1, False))
    ds = min(range(4, 257), key=lambda w: (abs(2*w*w+8*w+2-budget), w))
    jp = min(range(4, 257), key=lambda w: (abs(2*w*w+72*w+66-budget), w))
    cp = min(range(4, 513), key=lambda r: (abs(2274+65*r-budget), r))
    return dict(reference_total_parameters=budget, deepsets_width=ds,
                deepsets_wide_width=2*ds, janossy_width=jp, cp_rank=cp)


SIZES = sizes()


def template(seed, n):
    torch.manual_seed(seed)
    return source.LMANetwork(n, D, M, 3, DL, 1, False)


def build_model(seed, name, n):
    assert name in HEADS and n >= 2
    reference = template(seed, n)
    torch.manual_seed(10000+seed)
    if name.startswith("lma"):
        k = 3 if name == "lma3_clip1" else int(name[-1])
        model = source.LMANetwork(n, D, M, k, DL, 1, False)
        state = model.state_dict()
        ref = reference.state_dict()
        for key in state:
            state[key] = ref[key][:k].clone() if key.endswith("order_gates") else ref[key].clone()
        model.load_state_dict(state)
    elif name.startswith("deepsets"):
        width = SIZES["deepsets_wide_width" if name == "deepsets_wide" else "deepsets_width"]
        model = source.DeepSets(width)
        if name != "deepsets_ln":
            model.rho[0] = nn.Identity()
    elif name == "transformer":
        model = source.StandardTransformer(n, D, 4, 1, .1, False)
    elif name == "janossy2":
        model = PairClassifier(SIZES["janossy_width"])
    else:
        model = CPClassifier(SIZES["cp_rank"])
    if hasattr(model, "embedding"):
        model.embedding.load_state_dict(reference.embedding.state_dict())
    return model


def clip_threshold(name):
    return 1. if name == "lma3_clip1" else 10.


def source_paths():
    return [HERE/"parity_models.py", ROOT/"lma.py", CORE/"models.py",
            REVISION/"lma_revision.py", REVISION/"pdbbind_rerun.py",
            ROOT/"pdbbind_tensors_experiment.py"]
