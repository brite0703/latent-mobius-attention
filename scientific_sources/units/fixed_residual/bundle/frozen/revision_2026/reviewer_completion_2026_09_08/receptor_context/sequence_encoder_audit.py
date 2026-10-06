"""Independent convolution/gradient and chain/mask checks before fitting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from sequence_encoder import ChainSequenceEncoder,encode_strings,VOCAB

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent(model,tokens,mask,owners,records):
    vectors=[]
    for i in range(len(tokens)):
        length=int(mask[i].sum())
        h=model.embedding.weight[tokens[i,:length]].T
        for layer in model.convolutions:
            padded=F.pad(h,(4,4))
            # Explicit sliding windows and scalar tensor contractions, separate from conv1d.
            h=torch.stack([torch.einsum("oik,ik->o",layer.weight,padded[:,j:j+9])+layer.bias for j in range(length)],1).relu()
        vectors.append(h.amax(-1))
    output=[]
    for owner in range(records):
        indices=[i for i in range(len(tokens)) if int(owners[i])==owner]
        pooled=torch.stack([vectors[i] for i in indices]).mean(0)
        lengths=torch.tensor([len(indices),sum(int(mask[i].sum()) for i in indices)],dtype=pooled.dtype).log1p()
        output.append(torch.cat([pooled,lengths]))
    return torch.stack(output)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(2026090810)
    model=ChainSequenceEncoder().double()
    strings=["ACDX", "W", "GGT:KX", "ACDEFGHIKLMNPQ:STV:AX"]
    inputs=encode_strings(strings)
    actual=model(*inputs)
    reference=independent(model,*inputs)
    torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-11)
    weights=torch.linspace(-.8,1.2,actual.numel(),dtype=torch.float64).reshape_as(actual)
    params=tuple(model.parameters())
    lhs=torch.autograd.grad((actual*weights).sum(),params,retain_graph=True)
    rhs=torch.autograd.grad((reference*weights).sum(),params)
    for a,z in zip(lhs,rhs):
        torch.testing.assert_close(a,z,atol=1e-11,rtol=1e-10)
    assert bool(lhs[0][0].eq(0).all()) and bool(lhs[0][VOCAB["X"]].ne(0).any())
    tokens,mask,owners,records=inputs
    padded=model(F.pad(tokens,(0,17)),F.pad(mask,(0,17)),owners,records)
    individual=torch.cat([model(*encode_strings([s])) for s in strings])
    reordered=model(*encode_strings([":".join(reversed(s.split(":"))) for s in strings]))
    for value in (padded,individual,reordered):
        torch.testing.assert_close(actual,value,atol=1e-12,rtol=1e-11)
    all_states=model.state_dict()
    z=ChainSequenceEncoder().double()
    z.load_state_dict(all_states)
    torch.testing.assert_close(z(*inputs),actual,atol=0,rtol=0)
    failures=0
    for malformed in ("", "ACD:", ":ACD", "A-C", "ACD::EF"):
        try:
            encode_strings([malformed])
        except ValueError:
            failures+=1
    assert failures==5
    result=dict(passed=True,audited_utc=datetime.now(timezone.utc).isoformat(),
        independent_forward_max_delta=float((actual-reference).detach().abs().max()),
        independent_gradient_max_delta=max(float((a-z).abs().max()) for a,z in zip(lhs,rhs)),
        padding_max_delta=float((actual-padded).detach().abs().max()),
        batch_max_delta=float((actual-individual).detach().abs().max()),
        chain_order_max_delta=float((actual-reordered).detach().abs().max()),
        padding_embedding_gradient_zero=True,unknown_embedding_gradient_nonzero=True,
        malformed_inputs_rejected=5,parameters=sum(p.numel() for p in model.parameters()),output_dimension=98,
        source_sha256=sha(HERE/"sequence_encoder.py"),audit_source_sha256=sha(__file__),
        qualification="CPU formula/gradient/schema checks only. No retained fitting, GPU feasibility, affinity result or full-pipeline audit completed.")
    (HERE/"sequence_encoder_cpu_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    run()
