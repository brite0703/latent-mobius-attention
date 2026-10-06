"""Independent graph operations, source reuse, filtering, accumulation and CUDA feasibility."""
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import molecular_models as m
import molecular_data as d
import campaign as c

HERE,b=m.HERE,d.base


def close(x,y,atol=2e-10,rtol=2e-9):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    return float((x-y).detach().abs().max())


def gradients(model,left,right):
    weights=torch.linspace(-.7,1.3,left.numel(),dtype=left.dtype).reshape_as(left)
    params=tuple(model.parameters())
    lhs=torch.autograd.grad((left*weights).sum(),params,retain_graph=True)
    rhs=torch.autograd.grad((right*weights).sum(),params)
    return max(close(a,z,1e-9,2e-8) for a,z in zip(lhs,rhs))


def graph_batch():
    gen=torch.Generator().manual_seed(817)
    x=torch.randn(4,6,53,generator=gen,dtype=torch.float64)
    mask=torch.arange(6)[None]<torch.tensor([6,4,2,1])[:,None]
    raw=torch.zeros(4,6,6,dtype=torch.float64)
    for sample,n in enumerate(mask.sum(1)):
        n=int(n)
        raw[sample,:n,:n]=torch.eye(n,dtype=torch.float64)
        for j in range(n-1):
            raw[sample,j,j+1]=raw[sample,j+1,j]=1.
    degree=raw.sum(-1).clamp_min(1).rsqrt()
    return x*mask.unsqueeze(-1),mask,raw*degree[:,:,None]*degree[:,None,:]


def independent_gat(layer,x,mask,adj):
    h=F.linear(x,layer.projection.weight).reshape(len(x),x.shape[1],layer.heads,layer.width)
    batch=[]
    for sample in range(len(x)):
        nodes=[]
        for i in range(x.shape[1]):
            heads=[]
            for head in range(layer.heads):
                if not mask[sample,i]:
                    heads.append(h.new_zeros(layer.width))
                    continue
                neighbors=[j for j in range(x.shape[1]) if mask[sample,j] and (adj[sample,i,j]!=0 or i==j)]
                logits=torch.stack([F.leaky_relu((h[sample,i,head]*layer.attention_left[head]).sum()+
                    (h[sample,j,head]*layer.attention_right[head]).sum(),negative_slope=.2) for j in neighbors])
                probability=logits.softmax(0)
                heads.append(sum(p*h[sample,j,head] for p,j in zip(probability,neighbors)))
            nodes.append(torch.cat(heads))
        batch.append(torch.stack(nodes))
    return torch.stack(batch)


def independent_pointwise(model,x,mask):
    h=F.relu(F.linear(x,model.first.weight,model.first.bias))*mask.unsqueeze(-1)
    return F.relu(F.linear(h,model.second.weight,model.second.bias))*mask.unsqueeze(-1)


def independent_pair(block,x,mask):
    x=x*mask.unsqueeze(-1)
    left=independent_pointwise(block.left,x,mask)
    right=independent_pointwise(block.right,x,mask)
    product=torch.stack([torch.stack([sum(left[:,i,k]*right[:,k,j] for k in range(x.shape[1]))
                                    for j in range(x.shape[1])],1) for i in range(x.shape[1])],1)
    product=product*mask.unsqueeze(-1)
    return F.relu(F.linear(torch.cat([x,product],-1),block.compress.weight,block.compress.bias))*mask.unsqueeze(-1)


def independent_ppgn(model,x,mask,adj):
    h,pairs=model.initial_pairs(x,mask,adj)
    result=x.new_zeros(len(x))
    for block,output in zip(model.blocks,model.outputs):
        h=independent_pair(block,h,pairs)
        pooled=[]
        for sample in range(len(x)):
            valid=torch.where(mask[sample])[0].tolist()
            diagonal=torch.stack([h[sample,i,i] for i in valid]).amax(0) if valid else h.new_zeros(model.width)
            off=torch.stack([h[sample,i,j] for i in valid for j in valid if i!=j]).amax(0) if len(valid)>1 else h.new_zeros(model.width)
            pooled.append(torch.cat([diagonal,off]))
        result=result+F.linear(torch.stack(pooled),output.weight,output.bias).squeeze(-1)
    return result


def audit_cpu():
    data_audit=d.audit_data()
    x,mask,adj=graph_batch()
    checks,counts=[],[]
    for config in m.CONFIGS:
        spec=dict(**config,seed=42)
        model=m.build(spec).double().eval()
        counts.append(dict(**config,parameters=m.count(model)))
        original=model(x,mask,adj)
        perm=torch.tensor([3,0,5,1,4,2])
        permutation=close(original,model(x[:,perm],mask[:,perm],adj[:,perm][:,:,perm]))
        xp=F.pad(x,(0,0,0,3))
        xp[:,6:]=31.
        xp.requires_grad_()
        mp=F.pad(mask,(0,3))
        ap=F.pad(adj,(0,3,0,3))
        padded=model(xp,mp,ap)
        padding=close(padded,original)
        input_gradient=torch.autograd.grad(padded.sum(),xp)[0]
        assert float(input_gradient[~mp].abs().max())==0.
        batch=close(original,torch.cat([model(x[i:i+1],mask[i:i+1],adj[i:i+1]) for i in range(len(x))]))
        record=dict(**config,permutation_max_delta=permutation,padding_max_delta=padding,batch_max_delta=batch)
        if config["block"]=="ppgn":
            direct=model(x,mask,adj)
            explicit=independent_ppgn(model,x,mask,adj)
            record["independent_formula_max_delta"]=close(direct,explicit)
            record["independent_all_parameter_gradient_max_delta"]=gradients(model,direct,explicit)
            # Non-divisible microbatches independently verify correct effective-batch weighting.
            y=torch.tensor([-.7,.2,1.1,-.3],dtype=torch.float64)
            for micro in (1,3,4):
                model.zero_grad(set_to_none=True)
                loss=(model(x,mask,adj)-y).square().mean()
                loss.backward()
                expected=[p.grad.clone() for p in model.parameters()]
                model.zero_grad(set_to_none=True)
                total=c.accumulated_backward(model,(x,mask,adj),y,micro)
                assert abs(float(total)/len(y)-float(loss.detach()))<1e-12
                delta=max(close(p.grad,q,1e-9,2e-8) for p,q in zip(model.parameters(),expected))
                record[f"accumulation_micro{micro}_gradient_max_delta"]=delta
        checks.append(record)
    for d_in in (53,16):
        torch.manual_seed(731+d_in)
        layer=m.GATLayer(d_in).double()
        features=x[:,:,:d_in]
        direct=layer(features,mask,adj)
        reference=independent_gat(layer,features,mask,adj)
        checks.append(dict(operation="GAT",d_in=d_in,independent_formula_max_delta=close(direct,reference),
            independent_all_parameter_gradient_max_delta=gradients(layer,direct,reference)))
    for seed in c.SEEDS:
        for head in m.EXACT_HEADS:
            actual=m.build(dict(block="exact_labels",head=head,depth=1,width=None,seed=seed))
            old=m.controls.build(seed,head,1) if head=="deepsets_raw70" else m.core.build_model(seed,head,1)
            assert actual.state_dict().keys()==old.state_dict().keys()
            assert all(torch.equal(value,old.state_dict()[key]) for key,value in actual.state_dict().items())
        for depth in (1,3):
            refs=[]
            for head in m.GAT_HEADS:
                model=m.build(dict(block="gat",head=head,depth=depth,width=None,seed=seed))
                refs.append(model.encoder.state_dict())
                old=m.source_model(seed,head,depth)
                assert all(torch.equal(value,old.head.state_dict()[key]) for key,value in model.head.state_dict().items())
            assert all(all(torch.equal(value,ref[key]) for key,value in refs[0].items()) for ref in refs[1:])
        shallow=m.build(dict(block="gat",head="lma1",depth=1,width=None,seed=seed))
        deep=m.build(dict(block="gat",head="lma1",depth=3,width=None,seed=seed))
        assert all(torch.equal(value,deep.encoder.layers[0].state_dict()[key]) for key,value in shallow.encoder.layers[0].state_dict().items())
    specs=c.specifications()
    assert len(specs)==180 and len({s["id"] for s in specs})==180
    cycles=[]
    for edges in [[(i,(i+1)%6) for i in range(6)],[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]]:
        matrix=np.zeros((6,6),dtype=np.int64)
        for i,j in edges:
            matrix[i,j]=matrix[j,i]=1
        direct=int(np.trace(matrix@matrix@matrix))
        explicit=sum(int(matrix[i,j]*matrix[j,k]*matrix[k,i]) for i in range(6) for j in range(6) for k in range(6))
        assert direct==explicit
        cycles.append(direct)
    assert cycles==[0,12]
    b.write_json(HERE/"cpu_audit.json",dict(passed=True,completed_utc=b.now(),checks=checks,parameter_counts=counts,
        data_audit_sha256=b.sha(HERE/"data_prefit_audit.json"),exact_filter=data_audit["exclusion_reasons"],
        first_layer_pairing_seeds=c.SEEDS,cycle_triangle_trace_A3=cycles,
        qualification="Trace calculation audits a separating pair-tensor operation, not learned PPGN accuracy or a finite-width WL guarantee",
        sources=[dict(path=str(p),sha256=b.sha(p)) for p in c.sources()]))
    print(json.dumps(dict(stage="remaining_molecular_cpu_audit_passed",checks=len(checks),configurations=len(counts))),flush=True)


def audit_cuda():
    cpu=json.loads((HERE/"cpu_audit.json").read_text(encoding="utf-8"))
    assert cpu["passed"] and cpu["data_audit_sha256"]==b.sha(HERE/"data_prefit_audit.json")
    for item in cpu["sources"]:
        assert b.sha(item["path"])==item["sha256"]
    full=d.load("train",device="cuda")
    # Use the largest 256 graphs to verify the declared maximal padded training shape.
    indices=torch.argsort(full["mask"].sum(1),descending=True,stable=True)[:256]
    inputs=d.batch(full,indices)
    y=full["y"].double()
    target=((y-y.mean())/y.std(unbiased=False)).float()[indices]
    checks=[]
    for config in m.CONFIGS:
        spec=dict(**config,seed=42)
        model=m.build(spec).cuda()
        torch.manual_seed(30042)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started=time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro=32 if config["block"]=="ppgn" else 256
        total=c.accumulated_backward(model,inputs,target,micro)
        assert bool(torch.isfinite(total)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            assert bool(torch.isfinite(model(*(x[:64] for x in inputs))).all())
        torch.cuda.synchronize()
        checks.append(dict(**config,discarded_update=True,effective_batch=256,microbatch=micro,
            padded_atoms=inputs[0].shape[1],loss=float(total/len(target)),preclip_gradient_norm=float(norm),
            wall_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer
        torch.cuda.empty_cache()
    b.write_json(HERE/"prefit_audit.json",dict(passed=True,completed_utc=b.now(),cpu_audit_sha256=b.sha(HERE/"cpu_audit.json"),
        discarded_updates=checks))
    print(json.dumps(dict(stage="remaining_molecular_cuda_audit_passed",updates=len(checks))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["cpu","cuda"])
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    {"cpu":audit_cpu,"cuda":audit_cuda}[args.stage]()
