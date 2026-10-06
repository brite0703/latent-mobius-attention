"""Pre-fit equation, pairing, differentiation and CUDA checks; no test data."""
from itertools import combinations
import json
import time
import torch
from torch import nn
from torch.nn import functional as F
import gate_models as gm
import study as base
from prefit_audit import graph_batch

HERE=gm.HERE


def direct_layer(layer,x,mask):
    """Explicit per-graph, per-tuple reference for the separated equation."""
    results=[]
    for b in range(len(x)):
        q=layer.W_q(x[b])
        pi=F.softmax(layer.W_H(layer.W_k(x[b])),dim=-1)
        values=layer.W_v(x[b])
        buckets=torch.stack([sum(pi[i,j]*values[i] for i in range(len(x[b])) if mask[b,i])
                             for j in range(layer.M)])
        branches=[]
        for order in (1,2):
            parts=layer.interaction_projs[order-1](buckets).chunk(order,dim=-1)
            terms=[]
            for ids in combinations(range(layer.M),order):
                legs=torch.stack([parts[i][j] for i,j in enumerate(ids)])
                terms.append(legs.prod(0) if layer.feature_mode=="product" else legs.mean(0))
            memory=layer.interaction_mlps[order-1](torch.stack(terms))
            if order==1:
                memory=layer.order_gates[0]*memory
            token_outputs=[]
            for query in q:
                logits=torch.stack([torch.dot(query,v) for v in memory])/layer.d_model**.5
                weights=F.softmax(logits,dim=0)
                token_outputs.append(sum(a*v for a,v in zip(weights,memory)))
            branch=torch.stack(token_outputs)
            branches.append(branch if order==1 else layer.order_gates[1]*branch)
        results.append(layer.layer_norm(x[b]+layer.W_out(sum(branches)))*mask[b,:,None])
    return torch.stack(results)


def audit():
    base.configure()
    base.verify_lock()
    x,mask,adj=graph_batch()
    perm=torch.tensor([3,0,5,1,4,2])
    checks=[]
    for seed in base.SEEDS:
        reference=gm.build(seed,"legacy1",1).double().eval()
        refstate=reference.state_dict()
        oldproduct=gm.build(seed,"legacy2_product",1).double().eval()
        for name in gm.HEADS:
            model=gm.build(seed,name,1).double().eval()
            state=model.state_dict()
            for key,value in refstate.items():
                actual=state[key][:len(value)] if key.endswith("order_gates") else state[key]
                assert torch.equal(actual,value),(name,seed,key)
            # All non-gate parameters/buffers match across second-order models.
            if name!="legacy1":
                for key,value in oldproduct.state_dict().items():
                    if not key.endswith("order_gates"):
                        assert torch.equal(state[key],value),(name,seed,key)
            expected=3714 if name=="legacy1" else 4091
            assert gm.original_models.parameter_count(model)==expected
            p=model(x,mask,adj)
            q=model(x[:,perm],mask[:,perm],adj[:,perm][:,:,perm])
            padded=F.pad(x,(0,0,0,3))
            padded[:,-3:]=19.
            padded.requires_grad_()
            pm=F.pad(mask,(0,3))
            pa=F.pad(adj,(0,3,0,3))
            pp=model(padded,pm,pa)
            single=torch.cat([model(x[i:i+1],mask[i:i+1],adj[i:i+1]) for i in range(3)])
            errors={k:float(v.detach()) for k,v in dict(permutation=(p-q).abs().max(),
                padding=(p-pp).abs().max(),batch=(p-single).abs().max()).items()}
            assert max(errors.values())<1e-11,(name,seed,errors)
            pp.square().sum().backward()
            assert bool(torch.isfinite(padded.grad).all())
            assert float(padded.grad[~pm].abs().max())==0.
            off_delta=equation_delta=gate_gradient=branch_gradient=None
            if name.startswith("separated"):
                layer=model.head.layers[0]
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    layer.order_gates[1]=0.
                off_delta=float((model(x,mask,adj)-reference(x,mask,adj)).abs().max().detach())
                assert off_delta<1e-12,(name,seed,off_delta)
                target=torch.tensor([-.7,.2,1.1],dtype=torch.float64)
                (model(x,mask,adj)-target).square().mean().backward()
                gate_gradient=float(layer.order_gates.grad[1])
                branch_params=list(layer.interaction_projs[1].parameters())+list(layer.interaction_mlps[1].parameters())
                branch_gradient=max(float(v.grad.abs().max()) for v in branch_params)
                assert abs(gate_gradient)>1e-12 and branch_gradient==0.
                # Nonzero alpha exercises the new branch, independently of off-state.
                with torch.no_grad():
                    layer.order_gates[1]=.37
                h=model.encoder(x,mask,adj).detach().requires_grad_()
                actual=layer(h,mask)
                explicit=direct_layer(layer,h,mask)
                equation_delta=float((actual-explicit).abs().max().detach())
                assert equation_delta<1e-11
                ga,=torch.autograd.grad(actual.square().sum(),h,retain_graph=True)
                ge,=torch.autograd.grad(explicit.square().sum(),h)
                gradient_delta=float((ga-ge).abs().max())
                assert gradient_delta<1e-10
                # Also check invariance at nonzero alpha for zero-initialized procedures.
                nonzero=model(x,mask,adj)
                nonzero_perm=model(x[:,perm],mask[:,perm],adj[:,perm][:,:,perm])
                nonzero_pad=model(padded,pm,pa)
                assert float((nonzero-nonzero_perm).abs().max().detach())<1e-11
                assert float((nonzero-nonzero_pad).abs().max().detach())<1e-11
            else:
                gradient_delta=None
            checks.append(dict(seed=seed,head=name,total_parameters=expected,
                common_tensors_equal=True,second_order_tensors_equal=name!="legacy1",
                invariance_max_deltas=errors,masked_input_gradient=0.,off_state_max_delta=off_delta,
                separate_equation_max_delta=equation_delta,separate_input_gradient_max_delta=gradient_delta,
                zero_gate_gradient=gate_gradient,zero_gate_branch_parameter_max_gradient=branch_gradient))
    # Discarded train-only update; no validation/test outcome influences design.
    train=base.load_split("train")
    idx=torch.arange(256,device="cuda")
    xb,mb,ab=base.get_batch(train,idx)
    mean=float(train["y"].double().mean())
    sd=float(train["y"].double().std(unbiased=False))
    target=((train["y"][idx].double()-mean)/sd).float()
    resource=[]
    for name in gm.HEADS:
        model=gm.build(42,name,1).cuda()
        if name.endswith("zero"):
            one=gm.build(42,"legacy1",1).cuda()
            with torch.no_grad():
                delta=float((model(xb,mb,ab)-one(xb,mb,ab)).abs().max())
            assert delta<2e-6
            del one
        else:
            delta=None
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        pred=model(xb,mb,ab)
        assert pred.shape==target.shape==(256,)
        loss=(pred-target).square().mean()
        loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
        optimizer.step()
        with torch.no_grad():
            assert bool(torch.isfinite(model(xb,mb,ab)).all())
        torch.cuda.synchronize()
        resource.append(dict(head=name,batch=256,padded_atoms=xb.shape[1],
            off_state_float32_cuda_max_delta=delta,preclip_gradient_norm=float(norm),
            update_and_finite_forward_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer
    sources=[HERE/"gate_models.py",HERE/"audit.py",HERE/"campaign.py",
             gm.CORE/"models.py",gm.CORE/"study.py",gm.CORE/"prefit_audit.py",
             gm.REVISION/"lma_revision.py",gm.REVISION/"pdbbind_rerun.py",
             gm.REVISION.parent/"pdbbind_tensors_experiment.py"]
    record=dict(passed=True,audited_utc=base.now(),checks=checks,cuda_resource=resource,
        test_data_loaded=False,validation_data_loaded=False,retained_fits=0,
        tolerances=dict(float64_invariance=1e-11,float64_off=1e-12,float64_gradient=1e-10,float32_cuda_off=2e-6),
        sources=[dict(path=str(p),sha256=base.sha(p)) for p in sources])
    base.write_json(HERE/"prefit_audit.json",record)
    print(json.dumps(dict(stage="gate_prefit_audit",passed=True,procedures_checked=len(checks),
                         cuda_updates_discarded=len(resource))),flush=True)


if __name__=="__main__":
    audit()
