"""Native integration tests with explicit manufactured controller boundaries.

The actual 90/45 controller, native equations and timings have separate audits.
Here the controller is stubbed for positive wiring cases so no real retained
context or scientific test authorization is created.
"""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import campaign_control as control
import campaign_lifecycle as life
import costs
import execution
from models import ARMS,PairResidual
import native_cache
import native_costs
import native_inputs
import native_parent_bundle as parents
import native_pipeline
import native_profiles
import native_test_producer
import selection

HERE=Path(__file__).resolve().parent
SCRATCH=HERE/"discarded_native_delivery_20260909"


def rejected(call,kind=ValueError):
    try:
        call()
    except kind as error:
        return dict(type=type(error).__name__,message=str(error))
    raise AssertionError("Intentionally invalid native integration was accepted")


def validation_checks():
    folder=SCRATCH/"validation_check"
    domain,seed="ligand_contact",42
    identifier="ligand_contact_lma1_seed42_lr1"
    f0=torch.tensor([.1,-.2,.3],dtype=torch.float32)
    truth=torch.tensor([6.1,6.6,6.4],dtype=torch.float64)
    cache=execution.make_cache("val",["fixture:a","fixture:b","fixture:c"],f0,torch.zeros(3,8,8),truth,6.2,1.4,"a"*64)
    row=dict(domain=domain,seed=seed,selected_id=identifier)
    original=cache.f0.double().numpy()*1.4+6.2+np.array([1e-6,-1e-6,0.])
    archive=folder/"validation_predictions"/(identifier+".npz")
    archive.parent.mkdir(parents=True)
    np.savez(archive,ids=np.asarray(cache.ids),truth=cache.truth.numpy(),prediction=original)
    candidate=dict(artifacts=[dict(path=str(archive.relative_to(folder)),sha256=life.sha(archive))],
                   best_validation_rmse=math.sqrt(math.fsum(float(x)**2 for x in original-cache.truth.numpy())/3))
    with patch.object(parents,"MATCHED",folder):
        value,artifact=parents.validation_comparison(row,cache,candidate)
        assert 0.<value["archived_gpu_max_prediction_delta"]<2e-6
        bad=deepcopy(candidate);bad["best_validation_rmse"]+=.1
        errors=[rejected(lambda:parents.validation_comparison(row,cache,bad))]
        shifted=execution.make_cache("val",cache.ids,f0+.1,cache.z,truth,6.2,1.4,"a"*64)
        errors.append(rejected(lambda:parents.validation_comparison(row,shifted,candidate)))
        bad_identity=execution.make_cache("val",tuple(reversed(cache.ids)),f0,cache.z,truth,6.2,1.4,"a"*64)
        errors.append(rejected(lambda:parents.validation_comparison(row,bad_identity,candidate),AssertionError))
    return dict(valid_original_unit_comparison=True,rejections=errors,artifact=artifact)


def fixtures():
    result={}
    generator=torch.Generator().manual_seed(2026090848)
    for domain,seed in (("cubic",100),("ligand_contact",42)):
        module,_=native_inputs.modules(domain)
        with torch.random.fork_rng(devices=[]):
            model=module.build(seed,"lma1") if domain=="cubic" else module.build_model(seed,"ligand_contact","lma1")
        nodes,features=(12,12) if domain=="cubic" else (57,53)
        inputs=dict(x=torch.randn(7,nodes,features,generator=generator),mask=torch.ones(7,nodes,dtype=torch.bool))
        if domain=="ligand_contact":
            inputs.update(adj=torch.eye(nodes).expand(7,nodes,nodes).clone(),contact=torch.rand(7,nodes,216,generator=generator))
        checkpoint=SCRATCH/"manufactured_parents"/(domain+".pt")
        life.atomic_tensor(checkpoint,model.state_dict(),immutable=True)
        result[domain]=dict(model=model,inputs=inputs,checkpoint=life.artifact(checkpoint),
                            layer_path="head.layers.0" if domain=="cubic" else "pool.layers.0")
    return result


def producer_check(models):
    root=SCRATCH/"test_producer"
    rows=[dict(domain=d,seed=s,status="missing_parent" if (d,s)==("ligand_contact",46) else "available",
        reason="Injected two-rate parent failure" if (d,s)==("ligand_contact",46) else None,
        parent_checkpoint=None if (d,s)==("ligand_contact",46) else models[d]["checkpoint"])
        for d,s in control.REPLICATES]
    bundle_path=root/"manufactured_bundle.json"
    bundle=dict(parents=rows,producer_sources=[life.artifact(HERE/"native_test_producer.py")],manufactured_fixture=True)
    life.atomic_json(bundle_path,bundle,immutable=True)
    bundle_record=life.artifact(bundle_path)
    access_path=root/"test_access.json"
    life.atomic_json(access_path,dict(manufactured_controller_boundary=True),immutable=True)
    access=life.artifact(access_path)
    context=dict(scope=control.RETAINED,parent_bundle=bundle_record,manufactured_fixture=True)
    events=[];ready=False
    def gate(path):
        assert Path(path).resolve()==root.resolve()
        events.append("gate")
        if not ready:
            raise ValueError("Manufactured boundary: common selection is incomplete")
        return dict(manufactured_fixture=True)
    def read_payload(row,split,authorization=None):
        assert ready and "gate" in events and split=="test"
        assert authorization==dict(root=str(root.resolve()),test_access=access,parent_bundle=bundle_record)
        events.append("read_fixture")
        fixture=models[row["domain"]]
        payload=dict(split="test",ids=[f"manufactured:{row['domain']}:{row['seed']}:{i}" for i in range(7)],
            inputs=deepcopy(fixture["inputs"]),truth=torch.linspace(-1,1,7,dtype=torch.float64),
            target_mean=0.,target_sd=1.,sources=[fixture["checkpoint"]])
        return fixture["model"],fixture["layer_path"],payload
    def validate(path,manifest_path):
        assert Path(path).resolve()==root.resolve()
        manifest=life.read(manifest_path)
        assert manifest["parent_bundle"]==bundle_record and manifest["test_access"]==access
        assert len(manifest["parents"])==15 and manifest["producer_sources"]==bundle["producer_sources"]
        for row in manifest["parents"]:
            if row["status"]=="available":
                cache=life.load_cache(row["cache"],"test",row["parent_checkpoint"]["sha256"])
                assert len(cache.ids)==7 and row["native_prediction_and_bucket_equal"] is True
            else:
                assert row["cache"] is None and (row["domain"],row["seed"])==("ligand_contact",46)
        for source in manifest["sources"]:
            life.verify_artifact(source)
    with ExitStack() as stack:
        stack.enter_context(patch.object(control,"verify_context",lambda path:deepcopy(context)))
        stack.enter_context(patch.object(control,"test_access",gate))
        stack.enter_context(patch.object(control,"validate_test_manifest",validate))
        stack.enter_context(patch.object(native_inputs,"parent_and_inputs",read_payload))
        blocked=rejected(lambda:native_test_producer.produce(root,access,bundle_record))
        assert "read_fixture" not in events and not (root/"native_test_caches").exists()
        ready=True
        manifest_path=native_test_producer.produce(root,access,bundle_record)
        assert events.count("read_fixture")==14
        before=life.sha(manifest_path)
        assert native_test_producer.produce(root,access,bundle_record)==manifest_path
        assert events.count("read_fixture")==14 and before==life.sha(manifest_path)
        wrong=deepcopy(access);wrong["sha256"]="0"*64
        rejected(lambda:native_test_producer.produce(root,wrong,bundle_record))
    return dict(parent_outcomes=15,available=14,missing=1,manufactured_native_rows=98,
        early_gate_rejection=blocked,source_bound_manifest_and_idempotent_reuse=True,
        real_controller_boundary_stubbed=True,scientific_data_readers_called=False,
        artifact=life.artifact(manifest_path))


def profile_check(models):
    root=SCRATCH/"cost_delivery"
    parents_rows=[];choices=[]
    for domain,seed in control.REPLICATES:
        missing=(domain,seed)==("ligand_contact",46)
        cache=execution.make_cache("train",[f"manufactured:{domain}:{seed}:{i}" for i in range(256)],
            torch.zeros(256),torch.zeros(256,8,12 if domain=="cubic" else 8),torch.zeros(256,dtype=torch.float64),
            0.,1.,models[domain]["checkpoint"]["sha256"])
        cache_record=native_cache.save(cache,root/"manufactured_fitting"/f"{domain}_{seed}.pt")
        parents_rows.append(dict(domain=domain,seed=seed,status="missing_parent" if missing else "available",
            reason="Injected two-rate parent failure" if missing else None,parent_checkpoint=models[domain]["checkpoint"],caches={"train":cache_record}))
        for arm in ARMS:
            unavailable=missing or (domain,seed,arm)==("cubic",100,"pair_mlp")
            spec=next(r for r in selection.candidate_plan() if (r["domain"],r["seed"],r["arm"],r["rate_index"])==(domain,seed,arm,1))
            choices.append(dict(domain=domain,seed=seed,arm=arm,selected_id=None if unavailable else spec["id"]))
            if not unavailable:
                model=PairResidual(arm,cache.z.shape[-1],seed=seed)
                life.atomic_tensor(life.candidate_folder(root,spec)/"selected_checkpoint.pt",model.state_dict(),immutable=True)
    bundle_path=root/"parent_bundle.json"
    bundle=dict(parents=parents_rows,manufactured_fixture=True)
    life.atomic_json(bundle_path,bundle,immutable=True)
    context=dict(scope=control.RETAINED,parent_bundle=life.artifact(bundle_path),manufactured_fixture=True)
    life.atomic_json(root/"study_context.json",context,immutable=True)
    locked=dict(choices=choices,manufactured_controller_boundary=True)
    life.atomic_json(root/"selection_lock.json",locked,immutable=True)
    life.atomic_json(root/"evaluation.json",dict(complete=True,scientific_evaluation=True,
        manufactured_metadata_only=True,outcome_files=[],dependencies={}),immutable=True)
    calls=[]
    def load_payload(row,split):
        assert split=="train"
        fixture=models[row["domain"]]
        payload=dict(split="train",ids=[],inputs={},sources=[fixture["checkpoint"]])
        return fixture["model"],fixture["layer_path"],payload
    def complete(parent,layer_path,residual,cache,payload,workload,**kwargs):
        assert workload in native_costs.workload_plan()
        assert (residual is None)==(workload["procedure"]=="unchanged")
        if residual is not None:
            assert residual.arm==workload["procedure"] and residual.d==cache.z.shape[-1]
        calls.append("complete")
        return dict(manufactured_boundary_stub=True,no_timing_performed=True,workload=workload)
    def cached(model,cache,workload,rate,**kwargs):
        assert workload in costs.workload_plan() and model.arm==workload["arm"] and model.d==cache.z.shape[-1] and rate==.001
        calls.append("cached")
        return dict(manufactured_boundary_stub=True,no_timing_performed=True,workload=workload)
    with ExitStack() as stack:
        stack.enter_context(patch.object(control,"verify_context",lambda path:deepcopy(context)))
        stack.enter_context(patch.object(control,"verify_selection",lambda path:deepcopy(locked)))
        stack.enter_context(patch.object(control,"validate_parents",lambda path,scope:deepcopy(bundle)))
        stack.enter_context(patch.object(native_inputs,"parent_and_inputs",load_payload))
        stack.enter_context(patch.object(native_profiles,"require_idle_gpu",lambda:dict(manufactured_boundary_stub=True)))
        stack.enter_context(patch.object(native_costs,"profile",complete))
        stack.enter_context(patch.object(costs,"profile",cached))
        path=native_profiles.profile_all(root)
        receipt=life.read(path)
        assert len(receipt["rows"])==435 and len(receipt["artifacts"])==435
        counts={status:sum(row["status"]==status for row in receipt["rows"]) for status in ("profiled","missing_parent","no_valid_residual")}
        assert counts==dict(profiled=394,missing_parent=33,no_valid_residual=8)
        assert calls.count("complete")==165 and calls.count("cached")==229
        before=life.sha(path)
        assert native_profiles.profile_all(root)==path and before==life.sha(path) and len(calls)==394
        original=path.read_bytes()
        changed=deepcopy(receipt);changed["allocation"][0]["workload"]["batch_size"]=99
        try:
            life.atomic_json(path,changed)
            rejected(lambda:native_profiles.profile_all(root))
        finally:
            path.write_bytes(original)
    return dict(prescribed_outcomes=435,complete_outcomes=180,cached_outcomes=255,status_counts=counts,
        profile_dispatches=394,complete_dispatches=165,cached_dispatches=229,
        idempotent_reuse_without_measurement=True,changed_allocation_rejected=True,
        real_controller_and_timing_boundaries_stubbed=True,actual_timings_performed=0,artifact=life.artifact(path))


def main():
    destination=HERE/"native_delivery_cpu_audit.json"
    if destination.exists() or SCRATCH.exists():
        raise FileExistsError("Preserve existing native delivery audit outputs")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    for name in ("campaign_control_cpu_audit_v2.json","native_inputs_cpu_audit.json","native_costs_cpu_audit.json"):
        parents.verify_receipt(HERE/name)
    names=("native_parent_bundle.py","native_test_producer.py","native_profiles.py","native_pipeline.py","audit_native_delivery.py",
        "native_inputs.py","native_cache.py","native_costs.py","models.py","execution.py","selection.py","costs.py",
        "campaign_control.py","campaign_lifecycle.py","native_input_specification.md","native_cost_specification.md",
        "native_inputs_cpu_audit.json","native_costs_cpu_audit.json","campaign_control_cpu_audit_v2.json")
    sources=[life.artifact(HERE/name) for name in names]
    had_bundle=parents.DESTINATION.exists();had_context=native_pipeline.ROOT.exists()
    unavailable=rejected(parents.build)
    missing_decision=rejected(native_pipeline.activate)
    assert parents.DESTINATION.exists()==had_bundle and native_pipeline.ROOT.exists()==had_context
    check=validation_checks()
    models=fixtures()
    producer=producer_check(models)
    print(json.dumps(dict(stage="native test producer wiring",manufactured_parent_outcomes=15,real_test_access=False)),flush=True)
    profiles=profile_check(models)
    for source in sources:
        life.verify_artifact(source)
    receipt=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope="Native delivery wiring with explicitly stubbed controller/timing boundaries and manufactured data; separate real-kernel/source audits are bound",
        preceding_stage_rejection=unavailable,missing_activation_decision_rejection=missing_decision,
        validation_comparison=check,test_producer=producer,profiles=profiles,
        retained_context_created=False,actual_fifteen_parent_bundle_created=False,
        scientific_test_readers_called=False,retained_fits=0,scientific_timing_results=0,
        cuda_initialized=torch.cuda.is_initialized(),sources=sources,
        artifacts=[check["artifact"],producer["artifact"],profiles["artifact"]],
        remaining="Execute actual parent preparation, validation-equivalence and native source binding after both receptor studies; record any activation and then complete all scientific stages")
    life.atomic_json(destination,receipt,immutable=True)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("sources","artifacts","validation_comparison","test_producer","profiles")}),flush=True)


if __name__=="__main__":
    main()
