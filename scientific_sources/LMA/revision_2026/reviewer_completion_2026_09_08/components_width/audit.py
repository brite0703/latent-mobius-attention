"""Source/equation/component audit before retained fitting."""
import argparse
import copy
import json
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
import component_models as cm
import study as base
from prefit_audit import graph_batch
from lma_revision import MaskedLMAHead

HERE=cm.HERE


def sources():
    return [HERE/n for n in ("component_models.py","fit.py","audit.py","campaign.py")]+[
        cm.CORE/"models.py",cm.CORE/"study.py",cm.CORE/"prefit_audit.py",
        cm.REVISION/"lma_revision.py",cm.REVISION/"pdbbind_rerun.py",
        cm.REVISION.parent/"pdbbind_tensors_experiment.py"]


def cpu():
    import campaign
    torch.set_num_threads(4)
    base.verify_lock()
    specs=campaign.specifications()
    assert len(specs)==len({s["id"] for s in specs})==190
    for name in cm.HEADS:
        for seed in base.SEEDS:
            assert sorted(s["lr"] for s in specs if s["head"]==name and s["seed"]==seed)==base.LRS
    x,mask,adj=graph_batch()
    perm=torch.tensor([3,0,5,1,4,2])
    rows=[]
    for seed in base.SEEDS:
        encoder,head=cm.template(seed)
        refstate=head.double().state_dict()
        for name in cm.HEADS:
            config=cm.CONFIGS[name]
            model=cm.build_model(seed,name,1).double().eval()
            layer=model.head.layers[0]
            for key,v in model.encoder.state_dict().items():
                assert torch.equal(v,encoder.double().state_dict()[key])
            for key,v in model.head.state_dict().items():
                if ".combos_" in key:
                    continue
                ref=refstate[key]
                if key.endswith("order_gates") or ".W_H." in key:
                    ref=ref[:v.shape[0]]
                assert torch.equal(v,ref),(seed,name,key)
            assert sum(len(getattr(layer,f"combos_{k}")) for k in range(1,config["k"]+1))==sum(math.comb(config["M"],k) for k in range(1,config["k"]+1))
            p=model(x,mask,adj)
            pi=layer.last_routing.detach().clone()
            mass=(pi*mask[:,:,None]).sum((0,1))/mask.sum()
            raw=config["M"]*((mass-1/config["M"])**2).sum()
            assert float((raw-model.balance_loss()).abs())<1e-15
            q=model(x[:,perm],mask[:,perm],adj[:,perm][:,:,perm])
            route_perm=layer.last_routing.detach().clone()
            assert float((pi[:,perm]-route_perm).abs().max())<1e-12
            padded=F.pad(x,(0,0,0,3))
            padded[:,-3:]=23.
            padded.requires_grad_()
            pm=F.pad(mask,(0,3))
            pa=F.pad(adj,(0,3,0,3))
            padp=model(padded,pm,pa)
            single=torch.cat([model(x[i:i+1],mask[i:i+1],adj[i:i+1]) for i in range(3)])
            error=max(float((p-q).abs().max().detach()),float((p-padp).abs().max().detach()),
                      float((p-single).abs().max().detach()))
            assert error<1e-11,(name,seed,error)
            padp.square().sum().backward()
            assert float(padded.grad[~pm].abs().max())==0.
            # Reference is the original unchanged head on exactly the same weights.
            original=MaskedLMAHead(16,8,config["M"],config["k"],1).double().eval()
            state=original.state_dict()
            with torch.no_grad():
                for key,v in model.head.state_dict().items():
                    state[key].copy_(v)
                if config["routing"]=="uniform":
                    original.layers[0].W_H.weight.zero_()
                    original.layers[0].W_H.bias.zero_()
                if not config["query"]:
                    original.layers[0].W_q.weight.zero_()
                    original.layers[0].W_q.bias.zero_()
            h=model.encoder(x,mask,adj)
            ref=original(h,mask)
            initial_route_delta=float((pi-original.layers[0].last_routing).abs().max().detach())
            assert initial_route_delta<1e-12
            equation_delta=float((p-ref).abs().max().detach())
            assert equation_delta<1e-11,(name,seed,equation_delta)
            model.zero_grad(set_to_none=True)
            pred=model(x,mask,adj)
            target=x.new_tensor([-.7,.2,1.1])
            mse=(pred-target).square().mean()
            balance=model.balance_loss()
            if config["routing"]=="uniform":
                assert float(balance)==0.
                assert torch.equal(layer.last_routing,torch.full_like(layer.last_routing,1/config["M"]))
            penalty_gradient=None
            if config["routing"]=="learned":
                pg,=torch.autograd.grad(balance,layer.W_H.weight,retain_graph=True)
                penalty_gradient=float(pg.norm())
                assert penalty_gradient>1e-12
            loss=mse+config["penalty"]*balance if config["penalty"] else mse
            loss.backward()
            trainable=[v for v in model.parameters() if v.requires_grad]
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in trainable)
            assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
            before={k:v.clone() for k,v in model.state_dict().items()}
            old_routes=layer.last_routing.detach().clone()
            opt=torch.optim.AdamW(trainable,lr=.001,weight_decay=1e-4)
            norm=nn.utils.clip_grad_norm_(trainable,10.)
            assert bool(torch.isfinite(norm))
            opt.step()
            model(x,mask,adj)
            fixed_delta=None
            if config["routing"]=="fixed":
                fixed_delta=float((old_routes-layer.last_routing).abs().max())
                assert fixed_delta==0.
                after=model.state_dict()
                for key,v in before.items():
                    if key.startswith("fixed_encoder.") or ".W_k." in key or ".W_H." in key:
                        assert torch.equal(v,after[key])
                assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith("encoder."))
                model.train()
                model(x,mask,adj)
                assert torch.equal(layer.last_routing,old_routes)
            rows.append(dict(seed=seed,head=name,configuration=config,
                total_parameters=cm.parameter_count(model),trainable_parameters=cm.trainable_count(model),
                frozen_parameters=cm.parameter_count(model)-cm.trainable_count(model),
                memory_tokens=sum(math.comb(config["M"],k) for k in range(1,config["k"]+1)),
                common_tensors_equal=True,original_equation_max_delta=equation_delta,
                matched_initial_assignment_max_delta=initial_route_delta,
                invariance_max_delta=error,masked_input_gradient=0.,
                raw_penalty_weight_gradient_norm=penalty_gradient,
                fixed_assignment_max_delta_after_prediction_update=fixed_delta))
    record=dict(passed=True,audited_utc=base.now(),cpu_dtype="float64",checks=rows,
        sources=[dict(path=str(p),sha256=base.sha(p)) for p in sources()],
        validation_or_test_data_loaded=False,retained_fits=0)
    base.write_json(HERE/"cpu_audit.json",record)
    print(json.dumps(dict(stage="components_cpu_audit",passed=True,checks=len(rows))),flush=True)


def cuda():
    base.configure()
    record=json.loads((HERE/"cpu_audit.json").read_text(encoding="utf-8"))
    assert record["passed"]
    for item in record["sources"]:
        assert base.sha(item["path"])==item["sha256"]
    blob=torch.load(base.DATA/"pdbbind_train.pt",weights_only=True,map_location="cpu")
    largest=blob["mask"].sum(1).argsort(descending=True)[:256]
    x,mask,adj=(blob[k][largest].cuda() for k in ("X","mask","adj"))
    mean=float(blob["y"].double().mean())
    sd=float(blob["y"].double().std(unbiased=False))
    target=((blob["y"][largest].double()-mean)/sd).float().cuda()
    checks=[]
    for name in cm.HEADS:
        model=cm.build_model(42,name,1).cuda()
        params=[v for v in model.parameters() if v.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=1e-4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        p=model(x,mask,adj)
        assert p.shape==target.shape
        mse=(p-target).square().mean()
        penalty=model.balance_loss()
        loss=mse+cm.CONFIGS[name]["penalty"]*penalty
        loss.backward()
        norm=nn.utils.clip_grad_norm_(params,10.)
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
        optimizer.step()
        with torch.no_grad():
            assert bool(torch.isfinite(model(x[:64],mask[:64],adj[:64])).all())
        torch.cuda.synchronize()
        checks.append(dict(head=name,batch_size=256,padded_atoms=x.shape[1],
            one_update_and_eval_seconds=time.perf_counter()-start,preclip_gradient_norm=float(norm),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer
        torch.cuda.empty_cache()
    record.update(cuda_audited_utc=base.now(),cuda_checks=checks,training_data_only=True,
                  cpu_audit_sha256=base.sha(HERE/"cpu_audit.json"))
    base.write_json(HERE/"prefit_audit.json",record)
    print(json.dumps(dict(stage="components_cuda_audit",passed=True,discarded_updates=len(checks))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["cpu","cuda"])
    args=parser.parse_args()
    {"cpu":cpu,"cuda":cuda}[args.stage]()
