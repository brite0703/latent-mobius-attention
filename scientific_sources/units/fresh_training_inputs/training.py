"""CPU training from unchanged scientific models, with separate new run records."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, math, os, platform, sys, time, traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMP_REL = Path("revision_2026/reviewer_completion_2026_09_08")
SCIENTIFIC_COMPILES = set()
ENVIRONMENT = None


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".pending")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError,TypeError,OSError):
        return False


def verify_package():
    manifest=read(HERE/"package_manifest.json")
    for item in manifest["files"]:
        assert sha(HERE/item["path"])==item["sha256"],item["path"]
    assert not list((HERE/"frozen").rglob("*.pyc"))
    assert not list((HERE/"frozen").rglob("*.pt"))
    assert manifest["imported_historical_checkpoints"]==0
    return manifest


def initialize():
    global np, torch, ENVIRONMENT
    manifest=verify_package()
    original=Path(manifest["historical_root"]).resolve()
    def hook(event,args):
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            value=os.fsdecode(args[0])
            if inside(value,original):
                raise PermissionError("Original-workspace access is excluded from this training unit: "+value)
        if event=="compile" and len(args)>1 and isinstance(args[1],str):
            path=Path(args[1])
            if inside(path,HERE/"frozen"):
                SCIENTIFIC_COMPILES.add(path.resolve())
    sys.addaudithook(hook)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized(), "This verified entry point is CPU-only"
    ENVIRONMENT=dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,
        base_prefix=sys.base_prefix,torch=str(torch.__version__),numpy=str(np.__version__),platform=platform.platform(),
        cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        cuda_initialized=torch.cuda.is_initialized(),package_manifest_sha256=sha(HERE/"package_manifest.json"))
    return manifest


def import_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class Study:
    def __init__(self,name):
        self.name=name
        self.folder=HERE/"frozen"/COMP_REL/name
        self.parity=name=="synthetic_parity"
        if self.parity:
            self.models=import_at("portable_training_parity",self.folder/"parity_models.py")
            self.utilities=import_at("portable_training_parity_utilities",self.folder/"parity_common.py")
            self.adapter=None
        else:
            self.adapter=import_at("portable_training_"+name,self.folder/"adapter.py")
            self.utilities=self.adapter.u
            self.engine=import_at("portable_training_finite_engine_"+name,self.folder.parent/"finite_neural_engine.py")
        self.kind="classification" if self.parity else self.adapter.KIND
        self.epochs,self.batch,self.patience=100,256,8
        self.cache={}

    def data_check(self,spec):
        if self.parity:
            expected=self.utilities.generate(spec["n"],spec["seed"])
            path=self.utilities.data_path(spec["n"],spec["seed"])
        elif self.name=="first_cubic":
            g=self.adapter.g
            bits=g.population()
            expected=dict(bits=bits,**{task+"_target":g.target_numerators(bits,task)/math.sqrt(8) for task in self.adapter.TASKS},
                          **{key+"_ids":value for key,value in g.split(spec["seed"]).items()})
            path=self.adapter.data_path(spec["seed"])
        else:
            expected=self.adapter.g.generate(int(spec["task"][-1]),spec["seed"])
            path=self.adapter.data_path(spec["task"],spec["seed"])
        with np.load(path,allow_pickle=False) as saved:
            assert set(saved.files)==set(expected)
            for key,value in expected.items():
                np.testing.assert_array_equal(saved[key],value)
        return dict(study=self.name,path=path.relative_to(HERE).as_posix(),sha256=sha(path),arrays=len(expected),passed=True)

    def load(self,spec,split):
        key=(spec.get("task",spec.get("n")),spec["seed"],split)
        if key not in self.cache:
            if self.parity:
                d=self.utilities.load_data(spec["n"],spec["seed"],split,"cpu")
                d["inputs"]=(d["x"],)
                d["ids"]=np.arange(6000)[self.utilities.SLICES[split]]
            else:
                d=self.adapter.load(spec,split,"cpu")
            self.cache[key]=d
        return self.cache[key]

    def build(self,spec):
        return self.models.build_model(spec["seed"],spec["head"],spec["n"]) if self.parity else self.adapter.build(spec)

    def truth(self,data):
        return data.get("truth",data["y"].detach().numpy())

    def clip(self,spec):
        return self.models.clip_threshold(spec["head"]) if self.parity else self.adapter.CLIP

    def predict(self,model,spec,data):
        if self.parity:
            return self.utilities.canonical_logits(model,spec["n"])[data["count"]]
        return self.engine.predict(self.adapter,model,data)

    def metrics(self,truth,pred):
        return self.utilities.metrics(truth,pred) if self.parity else self.engine.metrics(self.adapter,truth,pred)

    def score(self,metrics):
        return metrics["cross_entropy" if self.kind=="classification" else "mse"]

    def diagnostics(self,model,spec,data):
        if self.parity:
            return self.utilities.cp_diagnostics(model,self.utilities.canonical(spec["n"],"cpu"))
        return self.adapter.diagnostics(model,data)


def binding():
    return dict(package_manifest_sha256=sha(HERE/"package_manifest.json"),plan_sha256=sha(HERE/"plan.json"),
                training_source_sha256=sha(__file__),environment=ENVIRONMENT)


def verified_record(folder,expected_binding=None):
    record=read(folder/"record.json")
    if expected_binding is not None:
        assert record["binding"]==expected_binding
    for item in record["artifacts"]:
        assert sha(folder/item["path"])==item["sha256"],item["path"]
    return record


def fit(study,spec,folder,fixture=False):
    folder=Path(folder)
    bound=binding()
    if (folder/"record.json").exists():
        old=verified_record(folder,bound)
        assert old["spec"]==spec and old["fixture"]==fixture and old["status"] in ("valid","failed")
        return old
    if folder.exists():
        raise RuntimeError("An unfinished attempt is preserved; do not overwrite it: "+str(folder))
    folder.mkdir(parents=True)
    write(folder/"started.json",dict(started_utc=now(),pid=os.getpid(),spec=spec,binding=bound,fixture=fixture))
    train,val=study.load(spec,"train"),study.load(spec,"val")
    model=study.build(spec).cpu()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    torch.save(initial,folder/"initial_state.pt")
    torch.manual_seed(30000+spec["seed"])
    generator=torch.Generator().manual_seed(20000+spec["seed"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec["lr"],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=study.epochs)
    best,best_epoch,bad,state=float("inf"),0,0,None
    history=[]
    started=time.perf_counter()
    record=dict(spec=spec,study=study.name,fixture=fixture,binding=bound,started_utc=now(),
                parameters=sum(p.numel() for p in model.parameters()),artifacts=[])
    try:
        for epoch in range(1,study.epochs+1):
            model.train()
            order=torch.randperm(len(train["y"]),generator=generator)
            total,maximum,correct=torch.zeros(()),torch.zeros(()),torch.zeros(())
            for start in range(0,len(order),study.batch):
                idx=order[start:start+study.batch]
                optimizer.zero_grad(set_to_none=True)
                pred=model(*(t[idx] for t in train["inputs"]))
                loss=(torch.nn.functional.cross_entropy(pred,train["y"][idx]) if study.kind=="classification"
                      else torch.nn.functional.mse_loss(pred,train["y"][idx]))
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),study.clip(spec))
                optimizer.step()
                total+=loss.detach()*len(idx)
                maximum=torch.maximum(maximum,norm.detach())
                if study.parity:
                    correct+=(pred.detach().argmax(1)==train["y"][idx]).sum()
            scheduler.step()
            if not bool(torch.isfinite(total)) or not bool(torch.isfinite(maximum)):
                raise FloatingPointError("Nonfinite epoch loss or preclip gradient")
            row=dict(epoch=epoch,training_loss=float(total/len(order)),maximum_preclip_gradient_norm=float(maximum),
                     next_learning_rate=float(scheduler.get_last_lr()[0]))
            if study.parity:
                row["training_accuracy"]=float(correct/len(order))
            if epoch==1 or epoch%5==0 or epoch==study.epochs:
                pred=study.predict(model,spec,val)
                metrics=study.metrics(study.truth(val),pred)
                score=study.score(metrics)
                row.update(validation_loss=score,validation=metrics)
                diagnostics=study.diagnostics(model,spec,val)
                if diagnostics is not None:
                    row["validation_diagnostics"]=diagnostics
                if score<best:
                    best,best_epoch,bad=score,epoch,0
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad+=1
            history.append(row)
            write(folder/"progress.json",dict(updated_utc=now(),pid=os.getpid(),epoch=epoch,best_epoch=best_epoch))
            if epoch%25==0:
                print(json.dumps(dict(stage="fresh_training",study=study.name,id=spec["id"],epoch=epoch,best_epoch=best_epoch)),flush=True)
            if fixture or bad>=study.patience:
                break
        assert state is not None
        model.load_state_dict(state,strict=True)
        pred=study.predict(model,spec,val)
        assert abs(study.score(study.metrics(study.truth(val),pred))-best)<=1e-12
        changed=sum(not torch.equal(initial[k],state[k]) for k in state)
        assert changed>0,"Fresh training must perform actual parameter updates"
        torch.save(dict(state_dict=state,spec=spec,binding=bound),folder/"checkpoint.pt")
        np.savez_compressed(folder/"validation.npz",ids=val["ids"],truth=study.truth(val),prediction=pred)
        record.update(status="valid",best_validation_loss=best,best_epoch=best_epoch,changed_state_tensors=changed)
    except Exception as exc:
        record.update(status="failed" if isinstance(exc,(FloatingPointError,torch.OutOfMemoryError)) else "implementation_error",
                      error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    record.update(finished_utc=now(),history=history,epochs_run=len(history),wall_seconds=time.perf_counter()-started,
                  reached_epoch_cap=len(history)==study.epochs,optimizer_updates=len(history)*math.ceil(len(train["y"])/study.batch))
    for name in ("initial_state.pt","checkpoint.pt","validation.npz"):
        if (folder/name).exists():
            record["artifacts"].append(dict(path=name,sha256=sha(folder/name)))
    write(folder/"record.json",record)
    if record["status"]=="implementation_error":
        raise RuntimeError(record["traceback"])
    print(json.dumps(dict(stage="fresh_candidate_complete",study=study.name,id=spec["id"],status=record["status"],epochs=record["epochs_run"])),flush=True)
    return record


def studies_and_plan():
    plan=read(HERE/"plan.json")
    studies={name:Study(name) for name in dict.fromkeys(c["study"] for c in plan["candidates"])}
    checks={}
    for c in plan["candidates"]:
        s=studies[c["study"]]
        lock=read(s.folder/"implementation_lock.json")
        assert c["spec"] in lock["candidates"]
        result=s.data_check(c["spec"])
        checks[result["path"]]=result
    write(HERE/"data_verification.json",dict(passed=True,files=list(checks.values())))
    return studies,plan


def train(studies,plan):
    preflight=read(HERE/"preflight/report.json")
    assert preflight["passed"] and preflight["binding"]==binding()
    for row in preflight["comparisons"]:
        assert sha(HERE/row["portable_record"])==row["portable_record_sha256"]
        assert sha(HERE/row["reference_record"])==row["reference_record_sha256"]
    write(HERE/"run_control.json",dict(status="running",pid=os.getpid(),started_utc=now(),binding=binding()))
    records=[]
    for c in plan["candidates"]:
        folder=HERE/"runs"/c["study"]/c["spec"]["id"]
        r=fit(studies[c["study"]],c["spec"],folder)
        records.append(dict(study=c["study"],spec=c["spec"],record=str((folder/"record.json").relative_to(HERE)),
                            sha256=sha(folder/"record.json"),status=r["status"],best=r.get("best_validation_loss")))
    choices=[]
    groups={}
    for r in records:
        key=(r["study"],r["spec"].get("task",r["spec"].get("n")),r["spec"]["head"],r["spec"]["seed"])
        groups.setdefault(key,[]).append(r)
    for group in groups.values():
        valid=[r for r in group if r["status"]=="valid"]
        winner=min(valid,key=lambda r:(r["best"],r["spec"]["lr"])) if valid else None
        choices.append(dict(candidates=group,selected=winner))
    write(HERE/"selection.json",dict(created_utc=now(),binding=binding(),candidates=records,choices=choices,
                                    used_test_for_selection=False))
    write(HERE/"run_control.json",dict(status="training_complete",pid=os.getpid(),finished_utc=now(),candidates=len(records),choices=len(choices)))


def evaluate(studies):
    selection=read(HERE/"selection.json")
    assert selection["binding"]==binding()
    for r in selection["candidates"]:
        assert sha(HERE/r["record"])==r["sha256"]
    rows=[]
    for choice in selection["choices"]:
        selected=choice["selected"]
        if selected is None:
            rows.append(dict(status="unavailable",reason="Both fixed candidates failed"))
            continue
        study,spec=studies[selected["study"]],selected["spec"]
        folder=(HERE/selected["record"]).parent
        verified_record(folder,binding())
        checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
        assert checkpoint["spec"]==spec and checkpoint["binding"]==binding()
        model=study.build(spec)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        data=study.load(spec,"test")
        pred=study.predict(model,spec,data)
        path=folder/"test.npz"
        np.savez_compressed(path,ids=data["ids"],truth=study.truth(data),prediction=pred)
        rows.append(dict(study=study.name,spec=spec,status="valid",prediction_file=str(path.relative_to(HERE)),
                         prediction_sha256=sha(path),metrics=study.metrics(study.truth(data),pred)))
    write(HERE/"evaluation.json",dict(created_utc=now(),selection_sha256=sha(HERE/"selection.json"),rows=rows,
                                      included_in_manuscript_results=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["train","evaluate"])
    args=parser.parse_args()
    initialize()
    studies,plan=studies_and_plan()
    if args.mode=="train":
        train(studies,plan)
    else:
        evaluate(studies)
    verify_package()
    write(HERE/(args.mode+"_environment.json"),dict(environment=ENVIRONMENT,
        scientific_compiles=[dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p)) for p in sorted(SCIENTIFIC_COMPILES)],
        frozen_files_unchanged=True,original_workspace_fallback_observed=False,cuda_initialized=torch.cuda.is_initialized()))


if __name__=="__main__":
    main()
