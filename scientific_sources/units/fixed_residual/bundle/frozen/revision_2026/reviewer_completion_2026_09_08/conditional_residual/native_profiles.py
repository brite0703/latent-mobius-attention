"""All selected residual cost outcomes, using original fitting inputs only."""
from pathlib import Path
import shutil
import subprocess
import sys

import torch

import campaign_control as control
import campaign_lifecycle as life
import costs
import execution
from models import PairResidual
import native_costs
import native_inputs
import selection


def require_idle_gpu():
    program=shutil.which("nvidia-smi")
    if not program:
        raise RuntimeError("GPU activity could not be checked before retained timing")
    command=[program,"--query-compute-apps=pid,process_name","--format=csv,noheader,nounits"]
    flags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0
    result=subprocess.run(command,capture_output=True,text=True,timeout=15,creationflags=flags,check=True)
    if result.stdout.strip():
        raise RuntimeError("A GPU compute process is still active; scientific timings must wait")
    return dict(program=program,query="compute-apps pid/process_name",active_compute_processes=0)


def allocation():
    rows=[]
    for domain,seed in control.REPLICATES:
        for workload in native_costs.workload_plan():
            if workload["domain"]==domain:
                rows.append(dict(family="complete",domain=domain,seed=seed,procedure=workload["procedure"],workload=workload))
        for workload in costs.workload_plan():
            if workload["domain"]==domain:
                rows.append(dict(family="cached",domain=domain,seed=seed,procedure=workload["arm"],workload=workload))
    if len(rows)!=435 or sum(r["family"]=="complete" for r in rows)!=180:
        raise ValueError("The prescribed cost allocation changed")
    return rows


def cost_id(row):
    work=row["workload"]
    return f"{row['domain']}_seed{row['seed']}_{row['procedure']}_{row['family']}_{work['mode']}_b{work['batch_size']}"


def load_residual(root,choice,dimension):
    if choice["selected_id"] is None:
        return None,None,None
    spec=next(r for r in selection.candidate_plan() if r["id"]==choice["selected_id"])
    checkpoint=life.artifact(life.candidate_folder(root,spec)/"selected_checkpoint.pt")
    model=PairResidual(spec["arm"],dimension,seed=spec["seed"]).eval()
    pairs=model.pairs.clone()
    model.load_state_dict(torch.load(checkpoint["path"],map_location="cpu",weights_only=True),strict=True)
    if not torch.equal(model.pairs,pairs) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("The cost checkpoint changes the declared residual or contains nonfinite parameters")
    return model,spec,checkpoint


def profile_all(root):
    root=Path(root).resolve()
    context=control.verify_context(root)
    if context["scope"]!=control.RETAINED or torch.get_num_threads()!=1 or torch.cuda.is_initialized():
        raise ValueError("Scientific profiling requires a retained one-thread CPU context")
    locked=control.verify_selection(root)
    bundle=control.validate_parents(context["parent_bundle"]["path"],control.RETAINED)
    evaluation_path=root/"evaluation.json"
    evaluation=life.read(evaluation_path)
    if evaluation.get("complete") is not True or evaluation.get("scientific_evaluation") is not True:
        raise ValueError("Complete selected-predictor evaluation is required before scientific cost reporting")
    for source in evaluation["outcome_files"]+list(evaluation["dependencies"].values()):
        life.verify_artifact(source)
    dependencies=dict(context=life.artifact(root/"study_context.json"),selection=life.artifact(root/"selection_lock.json"),
                      parent_bundle=context["parent_bundle"],evaluation=life.artifact(evaluation_path))
    destination=root/"profiles.json"
    if destination.exists():
        prior=life.read(destination)
        if prior.get("complete") is not True or prior["dependencies"]!=dependencies or prior["allocation"]!=allocation():
            raise ValueError("Existing scientific cost records have different dependencies or allocation")
        for artifact in prior["artifacts"]:
            life.verify_artifact(artifact)
        return destination
    gpu_check=require_idle_gpu()
    choices={(r["domain"],r["seed"],r["arm"]):r for r in locked["choices"]}
    rows=[];artifacts=[]
    with life.exclusive(root/"profile_process"):
        for parent_row in bundle["parents"]:
            domain,seed=parent_row["domain"],parent_row["seed"]
            jobs=[r for r in allocation() if (r["domain"],r["seed"])==(domain,seed)]
            parent=payload=cache=None
            if parent_row["status"]=="available":
                parent,layer_path,payload=native_inputs.parent_and_inputs(parent_row,"train")
                cache=life.load_cache(parent_row["caches"]["train"],"train",parent_row["parent_checkpoint"]["sha256"])
            for job in jobs:
                record_path=root/"cost_outcomes"/(cost_id(job)+".json")
                if record_path.exists():
                    recorded=life.read(record_path)
                    if recorded["dependencies"]!=dependencies or recorded["allocation"]!=job:
                        raise ValueError("A preserved cost outcome differs from its fixed allocation")
                    for source in recorded["sources"]:
                        life.verify_artifact(source)
                else:
                    sources=[];result=None;residual=None;spec=None;reason=None
                    if parent_row["status"]!="available":
                        status="missing_parent";reason=parent_row["reason"]
                    else:
                        sources=list(payload["sources"])+[parent_row["caches"]["train"]]
                        if job["procedure"]!="unchanged":
                            residual,spec,checkpoint=load_residual(root,choices[domain,seed,job["procedure"]],cache.z.shape[-1])
                            if checkpoint is not None:
                                sources.append(checkpoint)
                        if job["procedure"]!="unchanged" and residual is None:
                            status="no_valid_residual";reason="Both prescribed residual rates failed"
                        else:
                            try:
                                if job["family"]=="complete":
                                    result=native_costs.profile(parent,layer_path,residual,cache,payload,job["workload"])
                                else:
                                    result=costs.profile(residual,cache,job["workload"],rate=spec["rate"])
                                status="profiled"
                            except execution.NumericalFailure as error:
                                status="failed_numerical";reason=str(error)
                        for source in sources:
                            life.verify_artifact(source)
                    recorded=dict(dependencies=dependencies,allocation=job,status=status,reason=reason,
                                  sources=sources,result=result)
                    life.atomic_json(record_path,recorded,immutable=True)
                rows.append(dict(id=cost_id(job),status=recorded["status"],allocation=job,reason=recorded["reason"]))
                artifacts.append(life.artifact(record_path))
        control.verify_context(root)
        for source in dependencies.values():
            life.verify_artifact(source)
        require_idle_gpu()
        life.atomic_json(destination,dict(complete=True,dependencies=dependencies,allocation=allocation(),
            prescribed_outcomes=435,complete_predictor_outcomes=180,cached_residual_outcomes=255,
            rows=rows,artifacts=artifacts,profiled=sum(r["status"]=="profiled" for r in rows),
            gpu_idle_check=gpu_check,scope="All selected procedures, original fitting inputs, one CPU thread; missing and failed cost outcomes retained"),immutable=True)
    return destination
