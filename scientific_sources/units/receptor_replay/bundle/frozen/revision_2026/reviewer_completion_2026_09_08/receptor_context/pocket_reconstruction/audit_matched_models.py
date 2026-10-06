"""Discarded CPU equation, differentiation and invariance checks for all 15 models."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import matched_models as models
from sequence_encoder import encode_strings

HERE=Path(__file__).resolve().parent


def main():
    target=HERE/"matched_models_cpu_audit.json"
    if target.exists():
        raise FileExistsError("Preserve completed model audit")
    torch.set_num_threads(2)
    torch.manual_seed(2026090840)
    x=torch.randn(3,6,53,dtype=torch.float64)
    mask=torch.arange(6)[None,:]<torch.tensor([5,3,4])[:,None]
    x=x*mask[:,:,None]
    a=torch.zeros(3,6,6,dtype=torch.float64)
    for b,n in enumerate(mask.sum(1)):
        for i in range(int(n)):
            a[b,i,i]=1
            if i+1<int(n):a[b,i,i+1]=a[b,i+1,i]=1
    inv=a.sum(2).clamp_min(1).rsqrt()
    adj=a*inv[:,:,None]*inv[:,None,:]
    contact=torch.rand(3,6,216,dtype=torch.float64)*mask[:,:,None]
    strings=["ACDEFGHIK:LMNPQ","RSTVWYACDEFG","HIKLMNP:QRSTVWY"]
    sequence=encode_strings(strings)
    perm=torch.tensor([3,0,5,1,4,2])
    rows=[]
    parameter_tables={}
    for setting,head in models.configurations():
        model=models.build_model(42,setting,head).double()
        counts=model.parameter_counts()
        assert counts["contact_map"]==(7472 if setting=="ligand_contact" else 0)
        parameter_tables[setting+"/"+head]=counts
        args=dict(x=x,mask=mask,adj=adj,sequence_batch=sequence,contact=contact)
        prediction=model(**args)
        assert prediction.shape==(3,) and torch.isfinite(prediction).all()
        target_values=torch.tensor([.7,-1.2,.4],dtype=torch.float64)
        ((prediction-target_values)**2).mean().backward()
        gradient_norms={}
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),(setting,head,name)
            assert p.grad.abs().sum()>0,("Unused parameter in manufactured input",setting,head,name)
            gradient_norms[name]=float(p.grad.norm())
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            moved=model(x[:,perm],mask[:,perm],adj[:,perm][:,:,perm],sequence,contact[:,perm])
            permutation_difference=float((moved-prediction).abs().max())
            assert permutation_difference<3e-10,(setting,head,"ligand permutation")
            separate=torch.cat([model(x[b:b+1],mask[b:b+1],adj[b:b+1],encode_strings(strings[b:b+1]),contact[b:b+1]) for b in range(3)])
            separate_difference=float((separate-prediction).abs().max())
            assert separate_difference<3e-10,(setting,head,"separate-record evaluation")
            extended_x=F.pad(x,(0,0,0,3));extended_mask=F.pad(mask,(0,3));extended_adj=F.pad(adj,(0,3,0,3))
            extended_contact=F.pad(contact,(0,0,0,3),value=731.)
            padded=model(extended_x,extended_mask,extended_adj,sequence,extended_contact)
            padding_difference=float((padded-prediction).abs().max())
            assert padding_difference<3e-10,(setting,head,"padding")
            chain_reordered=encode_strings([":".join(reversed(s.split(":"))) for s in strings])
            chain_prediction=model(x,mask,adj,chain_reordered,contact)
            assert float((chain_prediction-prediction).abs().max())<3e-10,(setting,head,"chain order")
            equation_difference=None
            if setting=="ligand_contact":
                h=model.encoder(x,mask,adj)
                clean=torch.where(mask[:,:,None],contact,torch.zeros_like(contact))
                first=F.gelu(F.linear(clean,model.contact_map[0].weight,model.contact_map[0].bias))
                augmented=h+F.linear(first,model.contact_map[2].weight,model.contact_map[2].bias)*mask[:,:,None]
                pooled=model.pool(augmented,mask)
                joined=torch.cat((pooled,torch.log1p(mask.sum(1,keepdim=True).double())),dim=1)
                fused=F.gelu(F.linear(joined,model.ligand_projection.weight)+model.fusion_bias)
                hidden=F.gelu(F.linear(fused,model.regression[0].weight,model.regression[0].bias))
                manual=F.linear(hidden,model.regression[2].weight,model.regression[2].bias).flatten()
                equation_difference=float((manual-prediction).abs().max())
                assert equation_difference<3e-12,(head,"explicit augmentation/fusion equation")
                sensitivity=contact.clone();sensitivity[0,0,0]+=.75
                assert float((model(x,mask,adj,sequence,sensitivity)-prediction).abs().max())>1e-12,(head,"contact input ignored")
            else:
                base=models.base.build_model(42,setting,head).double()
                assert torch.equal(prediction,base(x,mask,adj,sequence)),(setting,head,"unchanged base interface")
            stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
            rebuilt=models.build_model(99,setting,head).double()
            rebuilt.load_state_dict(torch.load(stream,weights_only=True,map_location="cpu"),strict=True)
            assert torch.equal(prediction,rebuilt(**args)),(setting,head,"exact serialized state reload")
        rows.append(dict(setting=setting,head=head,parameters=counts,
            all_parameter_gradients_finite_and_nonzero=True,gradient_norms=gradient_norms,
            input_permutation_max_difference=permutation_difference,
            separate_record_max_difference=separate_difference,padding_max_difference=padding_difference,
            explicit_contact_equation_max_difference=equation_difference,
            exact_state_reload=True))
    # Matched initial tensors across modalities/heads; all models are separate instances.
    initial_checks=0
    for seed in range(42,47):
        reference=models.build_model(seed,"ligand_contact","lma2")
        for setting,head in models.configurations():
            current=models.build_model(seed,setting,head)
            for module_name in ("encoder","ligand_projection","regression"):
                left=getattr(reference,module_name).state_dict()
                right=getattr(current,module_name).state_dict()
                for key,value in left.items():
                    assert torch.equal(value,right[key]),(seed,setting,head,module_name,key)
                    initial_checks+=1
            if setting=="ligand_contact":
                for key,value in reference.contact_map.state_dict().items():
                    assert torch.equal(value,current.contact_map.state_dict()[key])
                    initial_checks+=1
    sources=[Path(__file__),Path(models.__file__),Path(models.base.__file__),
             HERE.parent/"sequence_encoder.py",HERE/"radial_contacts.py",
             HERE/"matched_protocol_draft.md",HERE/"matched_inputs/tensors_v1/manifest.json"]
    output=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),configurations=15,
        initial_tensor_equality_checks=initial_checks,parameter_tables=parameter_tables,rows=rows,
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        scope="Discarded manufactured-input CPU equations, gradients, permutations, padding, base equivalence and state reload. No optimizer lifecycle, selection, GPU or predictive certification.")
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:output[k] for k in ("passed","completed_utc","configurations","initial_tensor_equality_checks","parameter_tables")},indent=2))


if __name__=="__main__":
    main()
