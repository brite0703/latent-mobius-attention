"""Chain-aware sequence context; a disclosed CNN adaptation, not DeepDTA reproduction."""
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET="ACDEFGHIKLMNPQRSTVWYX"
VOCAB={symbol:index+1 for index,symbol in enumerate(ALPHABET)}


def encode_strings(strings,device="cpu"):
    chains,owners=[],[]
    for owner,string in enumerate(strings):
        parts=string.split(":")
        if not all(parts) or not all(set(part)<=set(ALPHABET) for part in parts):
            raise ValueError("Empty chain or unsupported symbol in a published sequence")
        chains.extend(parts)
        owners.extend([owner]*len(parts))
    if not chains:
        raise ValueError("At least one sequence is required")
    tokens=torch.zeros(len(chains),max(map(len,chains)),dtype=torch.long,device=device)
    for i,chain in enumerate(chains):
        tokens[i,:len(chain)]=torch.tensor([VOCAB[residue] for residue in chain],device=device)
    return tokens,tokens!=0,torch.tensor(owners,dtype=torch.long,device=device),len(strings)


class ChainSequenceEncoder(nn.Module):
    """Residue order within chains; mean of chain vectors plus declared length context."""
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(len(VOCAB)+1,64,padding_idx=0)
        self.convolutions=nn.ModuleList([nn.Conv1d(64,32,9,padding=4),
            nn.Conv1d(32,64,9,padding=4),nn.Conv1d(64,96,9,padding=4)])
        self.output_dimension=98

    def chain_vectors(self,tokens,mask):
        h=self.embedding(tokens).transpose(1,2)*mask[:,None]
        for layer in self.convolutions:
            h=F.relu(layer(h))*mask[:,None]
        return h.masked_fill(~mask[:,None],-torch.inf).amax(-1)

    def forward(self,tokens,mask,owners,records):
        h=self.chain_vectors(tokens,mask)
        output=h.new_zeros(records,96).index_add(0,owners,h)
        counts=h.new_zeros(records).index_add(0,owners,torch.ones_like(owners,dtype=h.dtype))
        residues=h.new_zeros(records).index_add(0,owners,mask.sum(-1).to(h.dtype))
        if not bool((counts>0).all()):
            raise ValueError("Every record must contain a nonempty supplied chain")
        return torch.cat([output/counts[:,None],torch.log1p(counts)[:,None],torch.log1p(residues)[:,None]],-1)
