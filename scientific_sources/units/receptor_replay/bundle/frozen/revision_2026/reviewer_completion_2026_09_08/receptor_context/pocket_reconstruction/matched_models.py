"""Fixed common-cohort models; no fitting or CUDA action at import."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sequence_models as base

SETTINGS = ("ligand_only", "ligand_sequence", "ligand_contact")
HEADS = base.HEADS


class MatchedAffinityModel(base.SequenceAffinityModel):
    def __init__(self, seed, setting, head):
        if setting not in SETTINGS or head not in HEADS:
            raise ValueError((setting,head))
        super().__init__(seed, "ligand_only" if setting == "ligand_contact" else setting, head)
        self.setting = setting
        self.contact_map = (base.seeded(47000+seed, lambda: nn.Sequential(nn.Linear(216,32),nn.GELU(),nn.Linear(32,16)))
                            if setting == "ligand_contact" else None)

    def forward(self, x=None, mask=None, adj=None, sequence_batch=None, contact=None):
        if self.contact_map is None:
            return super().forward(x,mask,adj,sequence_batch)
        if x is None or mask is None or adj is None or contact is None:
            raise ValueError("The contact setting requires aligned ligand and contact inputs")
        if x.shape[:2] != mask.shape or contact.shape != (*mask.shape,216):
            raise ValueError("Contact rows must align with the common ligand atoms")
        clean_contact = torch.where(mask.unsqueeze(-1),contact,torch.zeros_like(contact))
        local = self.contact_map(clean_contact)*mask.unsqueeze(-1)
        h = (self.encoder(x,mask,adj)+local)*mask.unsqueeze(-1)
        ligand = self.pool(h,mask)
        context = torch.cat([ligand,torch.log1p(mask.sum(1,keepdim=True).to(ligand.dtype))],-1)
        fused = self.ligand_projection(context)+self.fusion_bias
        return self.regression(F.gelu(fused)).squeeze(-1)

    def parameter_counts(self):
        result = super().parameter_counts()
        result["contact_map"] = base.count(self.contact_map)
        assert result["total"] == sum(value for key,value in result.items() if key != "total")
        return result


def configurations():
    return [(setting,head) for setting in SETTINGS for head in HEADS]


def build_model(seed,setting,head):
    return MatchedAffinityModel(seed,setting,head)
