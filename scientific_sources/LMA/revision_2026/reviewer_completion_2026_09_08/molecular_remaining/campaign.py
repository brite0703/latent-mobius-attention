"""Remaining 180 molecular candidates, fixed choices, then separate test scoring."""
import argparse
import json
import platform
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import molecular_models as m
import molecular_data as d

HERE=m.HERE
b=d.base
SEEDS,LRS=[42,43,44,45,46],[.0003,.001]
EPOCHS,BATCH,PATIENCE=150,256,8


def specifications():
    rows=[dict(**c,seed=seed,lr=lr,lr_index=j,
        id=f'{c["block"]}_d{c["depth"]}_{c["head"]}_w{c["width"] or 0}_seed{seed}_lr{j}')
        for c in m.CONFIGS for seed in SEEDS for j,lr in enumerate(LRS)]
    return [rows[int(i)] for i in np.random.default_rng(2026090809).permutation(len(rows))]


def sources():
    return list(dict.fromkeys(m.sources()+[HERE/"molecular_data.py",HERE/"campaign.py",HERE/"prefit_audit.py"]))


def lock():
    assert not (HERE/"implementation_lock.json").exists()
    cpu,gpu=(json.loads((HERE/name).read_text(encoding="utf-8")) for name in ("cpu_audit.json","prefit_audit.json"))
    assert cpu["passed"] and gpu["passed"] and gpu["cpu_audit_sha256"]==b.sha(HERE/"cpu_audit.json")
    for item in cpu["sources"]:
        assert b.sha(item["path"])==item["sha256"]
    analysis= json.loads((HERE/"analysis_prefit_audit.json").read_text(encoding="utf-8"))
    assert analysis["passed"] and analysis["analysis_source_sha256"]==b.sha(HERE/"analyze.py")
    files=sources()+[HERE/name for name in ["protocol.md","implementation_supplement.md","cpu_audit.json","prefit_audit.json","data_prefit_audit.json",
        "analyze.py","analysis_plan.md","analysis_prefit_audit.json","web_review_30_disposition.md"]]
    files += [d.DATA/"sample_manifest.csv",d.DATA/"data_audit.json",m.REVISION/"data_interpretation_audit.py"]
    files += [d.DATA/f"pdbbind_{s}.pt" for s in ("train","val","test")]
    counts=[dict(**config,total_parameters=m.count(m.build(dict(**config,seed=42)))) for config in m.CONFIGS]
    b.write_json(HERE/"implementation_lock.json",dict(locked_utc=b.now(),candidates=specifications(),candidate_count=180,
        selection_count=90,configurations=counts,seeds=SEEDS,learning_rates=LRS,epochs=EPOCHS,
        effective_batch_size=BATCH,ppgn_microbatch_size=32,prediction_batch_size=64,clip_norm=10.,patience_checks=PATIENCE,
        optimizer="AdamW weight_decay=0.0001; cosine horizon150",candidate_order_seed=2026090809,
        minibatch_seed="CPU torch.Generator 20000+seed",stochastic_seed="Torch 30000+seed after construction",
        target_scaling="Float64 fitting mean/population SD; separately recomputed for exact-label training subset",
        selection="Earliest strict minimum validation RMSE; lower numerical learning rate on exact cross-candidate tie",
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
            gpu=torch.cuda.get_device_name(),threads=4,tf32=False),
        files=[dict(path=str(p),sha256=b.sha(p),bytes=p.stat().st_size) for p in dict.fromkeys(files)]))
    print(json.dumps(dict(stage="remaining_molecular_locked",candidates=180,selections=90)),flush=True)


def verify():
    record=json.loads((HERE/"implementation_lock.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert b.sha(item["path"])==item["sha256"],item["path"]
    assert record["candidates"]==specifications()
    return record


def accumulated_backward(model,inputs,target,microbatch):
    # Sum / actual effective-batch size preserves the declared mean-MSE objective.
    total=torch.zeros((),device=target.device,dtype=target.dtype)
    for start in range(0,len(target),microbatch):
        y=target[start:start+microbatch]
        p=model(*(x[start:start+microbatch] for x in inputs))
        assert p.shape==y.shape
        squared=(p-y).square().sum()
        (squared/len(target)).backward()
        total+=squared.detach()
    return total


def fit(spec,train,val,mean,sd,lock_sha):
    path=HERE/"candidates"/(spec["id"]+".json")
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"))
        assert old["spec"]==spec and old["implementation_lock_sha256"]==lock_sha
        assert old["status"] in ("valid","failed")
        return old
    attempts=HERE/"attempts"
    attempts.mkdir(exist_ok=True)
    attempt=1+len(list(attempts.glob(spec["id"]+"_attempt*.json")))
    attempt_path=attempts/f'{spec["id"]}_attempt{attempt}.json'
    b.write_json(attempt_path,dict(started_utc=b.now(),spec=spec,attempt=attempt))
    model=m.build(spec).cuda()
    torch.manual_seed(30000+spec["seed"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec["lr"],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    generator=torch.Generator().manual_seed(20000+spec["seed"])
    target=((train["y"].double()-mean)/sd).float()
    val_truth=val["y"].double().cpu().numpy()
    microbatch=32 if spec["block"]=="ppgn" else BATCH
    best,best_epoch,state,bad_checks=float("inf"),0,None,0
    history,fatal=[],None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    result=dict(spec=spec,attempt=attempt,started_utc=b.now(),implementation_lock_sha256=lock_sha,
        total_parameters=m.count(model),target_mean=mean,target_sd=sd,training_records=len(target),
        effective_batch_size=BATCH,microbatch_size=microbatch)
    try:
        for epoch in range(1,EPOCHS+1):
            ep_started=time.perf_counter()
            model.train()
            order=torch.randperm(len(target),generator=generator).cuda()
            total=torch.zeros((),device="cuda")
            maximum_norm=torch.zeros((),device="cuda")
            for start in range(0,len(order),BATCH):
                idx=order[start:start+BATCH]
                inputs=d.batch(train,idx)
                optimizer.zero_grad(set_to_none=True)
                total+=accumulated_backward(model,inputs,target[idx],microbatch)
                norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
                maximum_norm=torch.maximum(maximum_norm,norm.detach())
                optimizer.step()
            scheduler.step()
            if not bool(torch.isfinite(total)) or not bool(torch.isfinite(maximum_norm)):
                raise FloatingPointError("Nonfinite training loss or gradient at epoch "+str(epoch))
            row=dict(epoch=epoch,training_standardized_mse=float(total/len(target)),
                     maximum_preclip_gradient_norm=float(maximum_norm))
            if epoch==1 or epoch%5==0 or epoch==EPOCHS:
                pred=d.predict(model,val,mean,sd)
                value=b.metrics(val_truth,pred)["rmse"]
                row["validation_rmse"]=value
                if value<best:
                    best,best_epoch,bad_checks=value,epoch,0
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad_checks+=1
            row["epoch_wall_seconds"]=time.perf_counter()-ep_started
            history.append(row)
            if epoch%25==0:
                print(json.dumps(dict(stage="training",id=spec["id"],epoch=epoch,best_validation_rmse=best)),flush=True)
            if bad_checks>=PATIENCE:
                break
        if state is None:
            raise FloatingPointError("No valid validation checkpoint")
        model.load_state_dict(state)
        prediction=d.predict(model,val,mean,sd)
        assert abs(b.metrics(val_truth,prediction)["rmse"]-best)<1e-10
        checkpoint=HERE/"checkpoints"/(spec["id"]+".pt")
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict=state,spec=spec,target_mean=mean,target_sd=sd,implementation_lock_sha256=lock_sha),checkpoint)
        validation=HERE/"validation_predictions"/(spec["id"]+".npz")
        validation.parent.mkdir(exist_ok=True)
        np.savez_compressed(validation,ids=np.asarray(val["ids"]),truth=val_truth,prediction=prediction)
        result.update(status="valid",best_epoch=best_epoch,best_validation_rmse=best,
            checkpoint=str(checkpoint.relative_to(HERE)),checkpoint_sha256=b.sha(checkpoint),
            validation_predictions=str(validation.relative_to(HERE)),validation_prediction_sha256=b.sha(validation))
    except Exception as exc:
        scientific=isinstance(exc,(FloatingPointError,torch.OutOfMemoryError))
        result.update(status="failed" if scientific else "implementation_error",error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),best_epoch=None,best_validation_rmse=None)
        if not scientific:
            fatal=exc
    torch.cuda.synchronize()
    result.update(finished_utc=b.now(),history=history,epochs_run=len(history),reached_epoch_cap=len(history)==EPOCHS,
        wall_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    b.write_json(path,result)
    b.write_json(attempt_path,dict(finished_utc=b.now(),spec=spec,attempt=attempt,status=result["status"]))
    print(json.dumps(dict(stage="candidate_complete",id=spec["id"],status=result["status"],seconds=result["wall_seconds"])),flush=True)
    del model,optimizer,scheduler
    torch.cuda.empty_cache()
    if fatal is not None:
        raise fatal
    return result


def train():
    record=verify()
    assert not (HERE/"selection_lock.json").exists()
    full,reduced,val=d.load("train",device="cuda"),d.load("train","exact_labels","cuda"),d.load("val",device="cuda")
    done=[]
    for spec in record["candidates"]:
        data=reduced if spec["block"]=="exact_labels" else full
        y=data["y"].double()
        mean,sd=float(y.mean()),float(y.std(unbiased=False))
        done.append(fit(spec,data,val,mean,sd,b.sha(HERE/"implementation_lock.json")))
        b.write_json(HERE/"progress.json",dict(updated_utc=b.now(),completed=len(done),total=180,
            valid=sum(r["status"]=="valid" for r in done),failed=sum(r["status"]=="failed" for r in done),
            aggregate_fit_seconds=sum(r["wall_seconds"] for r in done),last_id=spec["id"]))
    choices=[]
    for config in m.CONFIGS:
        for seed in SEEDS:
            group=[r for r in done if r["spec"]["seed"]==seed and all(r["spec"][key]==value for key,value in config.items())]
            assert len(group)==2
            valid=[r for r in group if r["status"]=="valid"]
            chosen=min(valid,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if valid else None
            choices.append(dict(**config,seed=seed,selected_id=chosen["spec"]["id"] if chosen else None,
                validation_rmse=chosen["best_validation_rmse"] if chosen else None,
                checkpoint_sha256=chosen["checkpoint_sha256"] if chosen else None,
                candidate_ids=[r["spec"]["id"] for r in group]))
    b.write_json(HERE/"selection_lock.json",dict(locked_utc=b.now(),selections=choices,
        implementation_lock_sha256=b.sha(HERE/"implementation_lock.json"),
        candidate_records=[dict(path=f'candidates/{r["spec"]["id"]}.json',sha256=b.sha(HERE/"candidates"/(r["spec"]["id"]+".json"))) for r in done]))
    print(json.dumps(dict(stage="remaining_molecular_selections_locked",choices=len(choices))),flush=True)


def verify_selection():
    verify()
    record=json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    assert len(record["selections"])==90 and len(record["candidate_records"])==180
    assert record["implementation_lock_sha256"]==b.sha(HERE/"implementation_lock.json")
    for item in record["candidate_records"]:
        assert b.sha(HERE/item["path"])==item["sha256"]
    return record


def restore(choice):
    candidate=json.loads((HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
    path=HERE/candidate["checkpoint"]
    assert b.sha(path)==candidate["checkpoint_sha256"]==choice["checkpoint_sha256"]
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    assert checkpoint["spec"]==candidate["spec"] and checkpoint["implementation_lock_sha256"]==b.sha(HERE/"implementation_lock.json")
    model=m.build(candidate["spec"]).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    return model,candidate


def evaluate():
    selected=verify_selection()
    assert not (HERE/"evaluation.json").exists()
    test=d.load("test",device="cuda")
    rows=[]
    destination=HERE/"test_predictions"
    destination.mkdir(exist_ok=True)
    for choice in selected["selections"]:
        if choice["selected_id"] is None:
            rows.append(dict(**choice,status="all_candidates_failed",test=None))
            continue
        model,candidate=restore(choice)
        prediction=d.predict(model,test,candidate["target_mean"],candidate["target_sd"])
        truth=test["y"].double().cpu().numpy()
        path=destination/(choice["selected_id"]+".npz")
        np.savez_compressed(path,ids=np.asarray(test["ids"]),truth=truth,prediction=prediction)
        rows.append(dict(**choice,status="valid",test=b.metrics(truth,prediction),
            total_parameters=candidate["total_parameters"],selected_lr=candidate["spec"]["lr"],
            best_epoch=candidate["best_epoch"],epochs_run=candidate["epochs_run"],training_records=candidate["training_records"],
            prediction_file=str(path.relative_to(HERE)),prediction_sha256=b.sha(path)))
        del model
    b.write_json(HERE/"evaluation.json",dict(evaluated_utc=b.now(),rows=rows,selection_lock_sha256=b.sha(HERE/"selection_lock.json")))
    print(json.dumps(dict(stage="remaining_molecular_evaluated",selected=len(rows))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["lock","train","evaluate"])
    args=parser.parse_args()
    b.configure()
    {"lock":lock,"train":train,"evaluate":evaluate}[args.stage]()
