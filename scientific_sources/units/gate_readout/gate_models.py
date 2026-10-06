"""Paired first-order construction and a separately normalized order branch."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
REVISION=HERE.parents[1]
CORE=REVISION/"neural_reviewer_study_2026_09_07"
sys.path.insert(0,str(CORE))
import models as original_models
from lma_revision import MaskedLatentMobiusAttention

ORIGINAL_BUILDER=original_models.build_model
HEADS=["legacy1","legacy2_product","legacy2_additive",
       "separated2_product_zero","separated2_additive_zero",
       "separated2_product_one","separated2_additive_one"]


class SeparatedOrders(MaskedLatentMobiusAttention):
    def forward(self,x,mask):
        q=self.W_q(x)
        pi=F.softmax(self.W_H(self.W_k(x)),dim=-1)
        valid=mask.unsqueeze(-1).to(x.dtype)
        z=torch.bmm(pi.transpose(1,2),self.W_v(x)*valid)
        self.last_routing,self.last_mask=pi,mask
        readout=None
        for order in range(1,self.k+1):
            parts=self.interaction_projs[order-1](z).chunk(order,dim=-1)
            tuples=getattr(self,f"combos_{order}")
            gamma=parts[0][:,tuples[:,0],:]
            for leg in range(1,order):
                value=parts[leg][:,tuples[:,leg],:]
                gamma=gamma*value if self.feature_mode=="product" else gamma+value
            if self.feature_mode=="additive":
                gamma=gamma/order
            memory=self.interaction_mlps[order-1](gamma)
            if order==1:
                memory=self.order_gates[0]*memory
            logits=torch.bmm(q,memory.transpose(1,2))/self.d_model**.5
            branch=torch.bmm(F.softmax(logits,dim=-1),memory)
            readout=branch if order==1 else readout+self.order_gates[order-1]*branch
        return self.layer_norm(x+self.W_out(readout))*valid


def copy_common(reference,model,second_gate):
    destination=model.state_dict()
    with torch.no_grad():
        for name,value in reference.state_dict().items():
            if name.endswith("order_gates"):
                destination[name][:len(value)].copy_(value)
                destination[name][len(value):].fill_(second_gate)
            else:
                assert destination[name].shape==value.shape,(name,destination[name].shape,value.shape)
                destination[name].copy_(value)


def build(seed,name,depth,d_in=53):
    if name not in HEADS or depth!=1:
        raise ValueError((name,depth))
    reference=ORIGINAL_BUILDER(seed,"lma1",1,d_in)
    if name=="legacy1":
        return reference
    feature_mode="additive" if "additive" in name else "product"
    model=ORIGINAL_BUILDER(seed,"additive2" if feature_mode=="additive" else "lma2",1,d_in)
    if name.startswith("separated"):
        layer=SeparatedOrders(16,8,2,8,feature_mode=feature_mode)
        layer.load_state_dict(model.head.layers[0].state_dict())
        model.head.layers[0]=layer
    copy_common(reference,model,0. if name.endswith("zero") else 1.)
    return model
