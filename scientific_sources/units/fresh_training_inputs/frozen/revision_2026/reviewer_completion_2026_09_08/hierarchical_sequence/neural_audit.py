"""Explicit sequence semantics, formula/gradient, data, pairing and GPU preflight."""
from pathlib import Path
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import finite_neural_engine as engine
a=engine.load_adapter("hierarchical_sequence")
m,u=a.m,a.u
pa=m.pm.import_file("hierarchy_reused_independent_layer",HERE.parent/"synthetic_parity/audit.py")


def close(x,y,atol=2e-10,rtol=2e-9):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    return float((x-y).detach().abs().max())


def from_encoded(model,name,h,independent=False):
    if name.startswith("lma"):
        for layer in model.layers:
            h=pa.independent_layer(layer,h) if independent else layer(h)
            h=h+model.ffn(h)
        return model.classifier(h.mean(1))
    if name=="transformer":
        return model.classifier(model.transformer_encoder(h).mean(1))
    head=model.head
    if name=="deepsets_wide":
        pooled=sum(head.phi(h[:,i]) for i in range(h.shape[1]))
        return head.rho(pooled)
    factors=head.factor(h)
    product=torch.ones_like(factors[:,0]).double()
    for i in range(h.shape[1]):
        product=product*factors[:,i].double()
    cp=torch.relu(nn.functional.linear(product.tanh().to(h.dtype),head.mix.weight))
    low=torch.relu(nn.functional.linear(sum(h[:,i] for i in range(h.shape[1])),head.low_order.weight))
    return head.readout(head.norm(cp+low))


def audit_cpu():
    a.prepare()
    checks,counts=[],{}
    for task in a.TASKS:
        depth=int(task[-1])
        n=3**depth
        # Audit the recursive generator on every prepared row, using independent signs.
        for seed in a.SEEDS:
            with np.load(a.data_path(task,seed)) as z:
                signs=2*z["x"].astype(np.int64)-1
                for _ in range(depth):
                    blocks=signs.reshape(len(signs),-1,3)
                    signs=(blocks.sum(-1)-blocks.prod(-1))//2
                np.testing.assert_array_equal((signs[:,0]>0).astype(np.int64),z["y"])
        data=a.load(dict(task=task,seed=200),"train")
        x=data["inputs"][0][:5].double()
        permutation=torch.randperm(n,generator=torch.Generator().manual_seed(818+n))
        for name in a.HEADS:
            spec=dict(task=task,seed=200,head=name)
            model=a.build(spec).double().eval()
            counts[task+"/"+name]=m.pm.parameter_count(model)
            output=model(x)
            assert bool(torch.isfinite(output).all())
            batch=close(torch.cat([model(x[i:i+1]) for i in range(len(x))]),output)
            h=model.embedding(x.unsqueeze(-1))+model.pos_embedding
            same=close(from_encoded(model,name,h),output)
            joint=close(from_encoded(model,name,h[:,permutation]),output)
            # Exchanging bits at fixed positions is intentionally not imposed as a symmetry.
            bit_only_delta=float((model(x[:,permutation])-output).detach().abs().max())
            record=dict(task=task,head=name,batch_max_delta=batch,encoded_path_max_delta=same,
                joint_token_position_permutation_max_delta=joint,
                bit_permutation_fixed_position_output_change=bit_only_delta)
            if name!="transformer":
                lhs=model(x)
                h=model.embedding(x.unsqueeze(-1))+model.pos_embedding
                rhs=from_encoded(model,name,h,independent=True)
                record["independent_formula_max_delta"]=close(lhs,rhs)
                record["independent_all_parameter_gradient_max_delta"]=pa.gradient_comparison(model,lhs,rhs)
            # Every model has an active position pathway; confirm finite, nonzero position gradients.
            weights=torch.linspace(-.7,1.1,output.numel(),dtype=torch.float64).reshape_as(output)
            grad=torch.autograd.grad((model(x)*weights).sum(),model.pos_embedding)[0]
            assert bool(torch.isfinite(grad).all()) and bool((grad!=0).any())
            record["position_gradient_norm"]=float(grad.norm())
            checks.append(record)
        for seed in a.SEEDS:
            ref=m.template(seed,n)
            state=ref.state_dict()
            for name in a.HEADS:
                model=m.build(seed,name,n)
                assert torch.equal(model.pos_embedding,ref.pos_embedding)
                assert all(torch.equal(v,ref.embedding.state_dict()[key]) for key,v in model.embedding.state_dict().items())
                if name.startswith("lma"):
                    for key,v in model.state_dict().items():
                        expected=state[key][:int(name[-1])] if key.endswith("order_gates") else state[key]
                        assert torch.equal(v,expected)
    specs=engine.specifications(a)
    assert len(specs)==360 and len({s["id"] for s in specs})==360
    # Tie-aware AUC and CE independently checked; degenerate label batches retain unavailable AUC.
    for y,p in [(np.array([0,1,0,1]),np.array([[0.,0.],[0.,1.],[0.,1.],[0.,-2.]])),
                (np.array([0,0]),np.array([[0.,0.],[1.,0.]]))]:
        computed,expected=engine.metrics(a,y,p),engine.independent_metrics(a,y,p)
        for key in computed:
            assert computed[key] is None and expected[key] is None or abs(computed[key]-expected[key])<1e-12
    for name in a.HEADS:
        m.build(200,name,9)
        order=torch.randperm(4800,generator=torch.Generator().manual_seed(20200))
        assert torch.equal(order,torch.randperm(4800,generator=torch.Generator().manual_seed(20200)))
    u.write_json(HERE/"cpu_audit.json",dict(passed=True,completed_utc=u.now(),datasets=30,
        model_checks=checks,parameter_counts=counts,sizes=a.SIZES,paired_seeds=a.SEEDS,
        sources=[dict(path=str(p),sha256=u.sha(p)) for p in engine.sources(a)]))
    print(json.dumps(dict(stage="hierarchy_cpu_audit_passed",datasets=30,model_checks=len(checks),parameters=counts)),flush=True)


def audit_cuda():
    cpu=engine.read(HERE/"cpu_audit.json")
    assert cpu["passed"]
    for item in cpu["sources"]:
        assert u.sha(item["path"])==item["sha256"]
    checks=[]
    for name in a.HEADS:
        spec=dict(task="depth4",head=name,seed=200)
        data=a.load(spec,"train","cuda")
        model=a.build(spec).cuda()
        torch.manual_seed(30200)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started=time.perf_counter()
        model.train()
        pred=model(data["inputs"][0][:a.BATCH])
        loss=engine.loss_value(a,pred,data["y"][:a.BATCH])
        loss.backward()
        assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm=nn.utils.clip_grad_norm_(model.parameters(),a.CLIP)
        optimizer.step()
        assert np.isfinite(engine.predict(a,model,a.load(spec,"val","cuda"))).all()
        torch.cuda.synchronize()
        checks.append(dict(head=name,discarded_update=True,n=81,batch_size=a.BATCH,loss=float(loss.detach()),
            preclip_gradient_norm=float(norm),wall_seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer,data
        torch.cuda.empty_cache()
    u.write_json(HERE/"prefit_audit.json",dict(passed=True,completed_utc=u.now(),
        cpu_audit_sha256=u.sha(HERE/"cpu_audit.json"),discarded_updates=checks))
    print(json.dumps(dict(stage="hierarchy_cuda_audit_passed",updates=len(checks))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["cpu","cuda"])
    args=parser.parse_args()
    u.configure()
    {"cpu":audit_cpu,"cuda":audit_cuda}[args.stage]()
