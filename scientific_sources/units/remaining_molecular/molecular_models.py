"""Declared GAT and PPGN adaptations, plus exact source reuse for label sensitivity."""
from pathlib import Path
import importlib.util
import sys
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
REVISION=HERE.parents[1]
CORE=REVISION/"neural_reviewer_study_2026_09_07"
sys.path.insert(0,str(CORE))
import models as core


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controls=imported("remaining_frozen_cardinality",CORE/"cardinality_controls/controls.py")
D=16
EXACT_HEADS=["mean_count","deepsets_raw70","cp_pool","lma1","lma2","additive2"]
GAT_HEADS=["mean_count","deepsets_raw70","cp_pool","lma1","lma2"]
CONFIGS=([dict(block="exact_labels",depth=1,head=head,width=None) for head in EXACT_HEADS]+
         [dict(block="gat",depth=depth,head=head,width=None) for depth in (1,3) for head in GAT_HEADS]+
         [dict(block="ppgn",depth=3,head="ppgn",width=width) for width in (16,32)])


class GATLayer(nn.Module):
    def __init__(self,d_in,d_out=16,heads=4):
        super().__init__()
        assert d_out%heads==0
        self.heads,self.width=heads,d_out//heads
        self.projection=nn.Linear(d_in,d_out,bias=False)
        self.attention_left=nn.Parameter(torch.empty(heads,self.width))
        self.attention_right=nn.Parameter(torch.empty(heads,self.width))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_left)
        nn.init.xavier_uniform_(self.attention_right)

    def forward(self,x,mask,adj):
        b,n,_=x.shape
        h=self.projection(x).reshape(b,n,self.heads,self.width).transpose(1,2)
        left=(h*self.attention_left[None,:,None,:]).sum(-1)
        right=(h*self.attention_right[None,:,None,:]).sum(-1)
        scores=F.leaky_relu(left[:,:,:,None]+right[:,:,None,:],negative_slope=.2)
        valid=mask[:,:,None]&mask[:,None,:]
        eye=torch.eye(n,dtype=torch.bool,device=x.device)[None]
        neighbors=((adj!=0)|eye)&valid
        # Padded queries receive a harmless self entry to avoid an all-minus-inf softmax row.
        safe_neighbors=neighbors|((~mask)[:,:,None]&eye)
        alpha=scores.masked_fill(~safe_neighbors[:,None],-torch.inf).softmax(-1)
        out=torch.matmul(alpha,h).transpose(1,2).reshape(b,n,-1)
        return out*mask.unsqueeze(-1)


class GATEncoder(nn.Module):
    def __init__(self,d_in,d,depth):
        super().__init__()
        self.layers=nn.ModuleList(GATLayer(d_in if i==0 else d,d) for i in range(depth))
        self.norms=nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))

    def forward(self,x,mask,adj):
        h=x
        for layer,norm in zip(self.layers,self.norms):
            h=F.gelu(norm(layer(h,mask,adj)))*mask.unsqueeze(-1)
        return h


class MaskedPointwise(nn.Module):
    def __init__(self,d_in,width):
        super().__init__()
        self.first=nn.Linear(d_in,width)
        self.second=nn.Linear(width,width)

    def forward(self,x,pair_mask):
        h=F.relu(self.first(x))*pair_mask.unsqueeze(-1)
        return F.relu(self.second(h))*pair_mask.unsqueeze(-1)


class PairBlock(nn.Module):
    def __init__(self,d_in,width):
        super().__init__()
        self.left=MaskedPointwise(d_in,width)
        self.right=MaskedPointwise(d_in,width)
        self.compress=nn.Linear(d_in+width,width)

    def forward(self,x,pair_mask):
        x=x*pair_mask.unsqueeze(-1)
        left=self.left(x,pair_mask).permute(0,3,1,2)
        right=self.right(x,pair_mask).permute(0,3,1,2)
        product=torch.matmul(left,right).permute(0,2,3,1)*pair_mask.unsqueeze(-1)
        return F.relu(self.compress(torch.cat([x,product],-1)))*pair_mask.unsqueeze(-1)


class PPGNModel(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width=width
        self.blocks=nn.ModuleList(PairBlock(54 if i==0 else width,width) for i in range(3))
        self.outputs=nn.ModuleList(nn.Linear(2*width,1) for _ in range(3))

    def initial_pairs(self,x,mask,adj):
        n=x.shape[1]
        eye=torch.eye(n,dtype=x.dtype,device=x.device)[None,:,:,None]
        pair_mask=mask[:,:,None]&mask[:,None,:]
        diagonal=x[:,:,None,:]*eye
        return torch.cat([diagonal,adj.unsqueeze(-1)],-1)*pair_mask.unsqueeze(-1),pair_mask

    def pool(self,h,mask):
        n=h.shape[1]
        diagonal=h.diagonal(dim1=1,dim2=2).transpose(1,2)
        diagonal_max=diagonal.masked_fill(~mask.unsqueeze(-1),-torch.inf).amax(1)
        pair_mask=mask[:,:,None]&mask[:,None,:]
        off=pair_mask&~torch.eye(n,dtype=torch.bool,device=h.device)[None]
        off_max=h.masked_fill(~off.unsqueeze(-1),-torch.inf).amax((1,2))
        off_max=torch.where((mask.sum(1)>1)[:,None],off_max,torch.zeros_like(off_max))
        diagonal_max=torch.where(mask.any(1)[:,None],diagonal_max,torch.zeros_like(diagonal_max))
        return torch.cat([diagonal_max,off_max],-1)

    def forward(self,x,mask,adj):
        h,pair_mask=self.initial_pairs(x,mask,adj)
        predictions=[]
        for block,output in zip(self.blocks,self.outputs):
            h=block(h,pair_mask)
            predictions.append(output(self.pool(h,mask)).squeeze(-1))
        return torch.stack(predictions).sum(0)


def source_model(seed,head,depth):
    if head=="deepsets_raw70":
        if depth==1:
            return controls.build(seed,head,1)
        model=core.build_model(seed,"deepsets_wide",depth)
        model.head.rho=nn.Sequential(*list(model.head.rho.children())[1:])
        return model
    return core.build_model(seed,head,depth)


def build(spec):
    seed=spec["seed"]
    if spec["block"]=="exact_labels":
        return source_model(seed,spec["head"],1)
    if spec["block"]=="ppgn":
        torch.manual_seed(seed)
        return PPGNModel(spec["width"])
    # Head tensors and their construction seed are the original source procedure.
    model=source_model(seed,spec["head"],spec["depth"])
    torch.manual_seed(seed)
    model.encoder=GATEncoder(53,D,spec["depth"])
    model.backbone="gat"
    return model


def count(model):
    return sum(p.numel() for p in model.parameters())


def sources():
    return [Path(__file__).resolve(),CORE/"models.py",CORE/"study.py",CORE/"cardinality_controls/controls.py",
            REVISION/"lma_revision.py",REVISION/"pdbbind_rerun.py",REVISION.parent/"pdbbind_tensors_experiment.py"]
