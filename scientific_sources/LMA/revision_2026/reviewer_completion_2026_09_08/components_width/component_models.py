"""Original memory geometry with declared routing/query/width interventions."""
import copy
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
REVISION=HERE.parents[1]
CORE=REVISION/"neural_reviewer_study_2026_09_07"
sys.path.insert(0,str(CORE))
import models as original_models
from lma_revision import MaskedLMAHead,MaskedLatentMobiusAttention
from pdbbind_rerun import GraphEncoder,AffinityModel

CONFIGS={}
for order in (1,2):
    for procedure,mode,penalty,query in [
        ("learned0","learned",0.,True),("learned001","learned",.01,True),
        ("learned01","learned",.1,True),("fixed","fixed",0.,True),
        ("uniform","uniform",0.,True),("noquery","learned",0.,False)]:
        CONFIGS[f"m8_k{order}_{procedure}"]=dict(M=8,k=order,routing=mode,penalty=penalty,query=query)
for M,order in ((4,1),(4,2),(4,3),(8,3),(16,1),(16,2),(16,3)):
    CONFIGS[f"m{M}_k{order}_learned0"]=dict(M=M,k=order,routing="learned",penalty=0.,query=True)
HEADS=list(CONFIGS)


class ComponentLayer(MaskedLatentMobiusAttention):
    def __init__(self,M,k,routing,query):
        super().__init__(16,M,k,8)
        self.routing_mode,self.query_conditioned=routing,query
        self.external_routes=None

    def forward(self,h,mask):
        valid=mask.unsqueeze(-1).to(h.dtype)
        if self.routing_mode=="uniform":
            pi=h.new_full((h.shape[0],h.shape[1],self.M),1/self.M)
        elif self.routing_mode=="fixed":
            if self.external_routes is None:
                raise RuntimeError("The frozen input path must supply the actual assignments")
            pi=self.external_routes
            assert pi.shape==(*h.shape[:2],self.M)
        else:
            pi=F.softmax(self.W_H(self.W_k(h)),dim=-1)
        self.last_routing,self.last_mask=pi,mask
        z=torch.bmm(pi.transpose(1,2),self.W_v(h)*valid)
        memories=[]
        for order in range(1,self.k+1):
            parts=self.interaction_projs[order-1](z).chunk(order,dim=-1)
            tuples=getattr(self,f"combos_{order}")
            gamma=parts[0][:,tuples[:,0],:]
            for leg in range(1,order):
                gamma=gamma*parts[leg][:,tuples[:,leg],:]
            memories.append(self.order_gates[order-1]*self.interaction_mlps[order-1](gamma))
        memory=torch.cat(memories,dim=1)
        if self.query_conditioned:
            q=self.W_q(h)
            scores=torch.bmm(q,memory.transpose(1,2))/self.d_model**.5
            readout=torch.bmm(F.softmax(scores,dim=-1),memory)
        else:
            readout=memory.mean(1,keepdim=True).expand(-1,h.shape[1],-1)
        return self.layer_norm(h+self.W_out(readout))*valid


class ComponentModel(AffinityModel):
    def __init__(self,encoder,head,routing):
        super().__init__(encoder,head,"gcn")
        self.routing_mode=routing
        if routing=="fixed":
            self.fixed_encoder=copy.deepcopy(encoder)
            self.fixed_encoder.requires_grad_(False)
            self.head.layers[0].W_k.requires_grad_(False)
            self.head.layers[0].W_H.requires_grad_(False)

    def forward(self,x,mask,adj):
        if self.routing_mode=="fixed":
            with torch.no_grad():
                fixed_h=self.fixed_encoder(x,mask,adj)
                layer=self.head.layers[0]
                layer.external_routes=F.softmax(layer.W_H(layer.W_k(fixed_h)),dim=-1)
        return super().forward(x,mask,adj)

    def balance_loss(self):
        return self.head.balance_loss()


def template(seed,d_in=53):
    torch.manual_seed(seed)
    encoder=GraphEncoder(d_in,16,1)
    torch.manual_seed(10000+seed)
    head=MaskedLMAHead(16,8,16,3,1)
    return encoder,head


def build_model(seed,name,depth,d_in=53):
    assert name in CONFIGS and depth==1
    config=CONFIGS[name]
    encoder,reference=template(seed,d_in)
    head=MaskedLMAHead(16,8,config["M"],config["k"],1)
    head.layers[0]=ComponentLayer(config["M"],config["k"],config["routing"],config["query"])
    state=head.state_dict()
    refstate=reference.state_dict()
    with torch.no_grad():
        for key,value in state.items():
            if ".combos_" in key:
                continue
            original=refstate[key]
            if key.endswith("order_gates") or ".W_H." in key:
                value.copy_(original[:value.shape[0]])
            else:
                assert value.shape==original.shape,(key,value.shape,original.shape)
                value.copy_(original)
    layer=head.layers[0]
    if config["routing"]=="uniform":
        del layer.W_k,layer.W_H
    if not config["query"]:
        del layer.W_q
    return ComponentModel(encoder,head,config["routing"])


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def trainable_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
