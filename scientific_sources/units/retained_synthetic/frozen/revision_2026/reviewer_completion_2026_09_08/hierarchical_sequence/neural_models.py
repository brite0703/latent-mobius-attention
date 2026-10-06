"""All sequence procedures receive the same trainable token and position inputs."""
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"synthetic_parity"))
import parity_models as pm
source, core = pm.source, pm.core
D, DL, M = 32, 16, 8
HEADS = ["lma1", "lma2", "lma3", "transformer", "deepsets_wide", "cp_pool"]


class PositionClassifier(nn.Module):
    def __init__(self, n, head):
        super().__init__()
        self.embedding = nn.Linear(1,D)
        self.pos_embedding = nn.Parameter(torch.randn(1,n,D))
        self.head = head

    def encode(self,x):
        return self.embedding(x.unsqueeze(-1))+self.pos_embedding

    def forward(self,x):
        h=self.encode(x)
        mask=torch.ones(x.shape,dtype=torch.bool,device=x.device)
        return self.head(h,mask)


def sizes():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        budget=pm.parameter_count(source.LMANetwork(9,D,M,2,DL,1,True))-D*9-2*D
    ds=min(range(4,257),key=lambda w:(abs(2*w*w+(D+5)*w+2-budget),w))
    return dict(reference_excluding_token_and_position_parameters=budget,
                deepsets_matched_plain_width=ds,deepsets_wide_width=2*ds,cp_rank=pm.SIZES["cp_rank"])


SIZES=sizes()


def template(seed,n):
    torch.manual_seed(seed)
    return source.LMANetwork(n,D,M,3,DL,1,True)


def build(seed,name,n):
    assert name in HEADS
    reference=template(seed,n)
    torch.manual_seed(10000+seed)
    if name.startswith("lma"):
        k=int(name[-1])
        model=source.LMANetwork(n,D,M,k,DL,1,True)
        state=model.state_dict()
        ref=reference.state_dict()
        for key in state:
            state[key]=ref[key][:k].clone() if key.endswith("order_gates") else ref[key].clone()
        model.load_state_dict(state)
    elif name=="transformer":
        model=source.StandardTransformer(n,D,4,1,.1,True)
    elif name=="deepsets_wide":
        head=core.original.DeepSetsHead(D,SIZES["deepsets_wide_width"])
        head.rho[0]=nn.Identity()
        head.rho[-1]=nn.Linear(SIZES["deepsets_wide_width"],2)
        model=PositionClassifier(n,head)
    else:
        head=core.CPPoolHead(D,SIZES["cp_rank"])
        head.readout[-1]=nn.Linear(D,2)
        model=PositionClassifier(n,head)
    model.embedding.load_state_dict(reference.embedding.state_dict())
    with torch.no_grad():
        model.pos_embedding.copy_(reference.pos_embedding)
    return model


def sources():
    return list(dict.fromkeys([Path(__file__).resolve()]+pm.source_paths()))
