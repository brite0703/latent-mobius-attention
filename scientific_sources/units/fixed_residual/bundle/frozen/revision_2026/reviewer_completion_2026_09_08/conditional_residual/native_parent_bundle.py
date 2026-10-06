"""Prepare the actual fifteen-parent bundle after both receptor studies end."""
import math
from pathlib import Path
import sys

import numpy as np
import torch

import campaign_control as control
import campaign_lifecycle as life
import native_cache
import native_inputs

HERE=Path(__file__).resolve().parent
RECEPTOR=native_inputs.POCKET.parent
MATCHED=native_inputs.POCKET/"matched_study"
DESTINATION=HERE/"native_parent_bundle_v1"


def verify_receipt(path):
    value=life.read(path)
    if value.get("passed") is not True:
        raise ValueError("Required adapter audit is incomplete: "+str(path))
    for source in value.get("sources",[])+value.get("artifacts",[]):
        life.verify_artifact(source)
    return value


def preceding_studies():
    paths=dict(sequence_final_audit=RECEPTOR/"sequence_final_audit.json",sequence_profiles=RECEPTOR/"sequence_profiles.json",
        matched_final_audit=MATCHED/"final_audit.json",matched_profiles=MATCHED/"profiles.json")
    # Read no parent or scientific input until every preceding terminal stage exists.
    if not all(p.exists() for p in paths.values()):
        raise ValueError("The full-sequence and matched evaluation, final audits and profiles must finish first")
    for role,path in paths.items():
        record=life.read(path)
        if record.get("complete" if role.endswith("profiles") else "passed") is not True:
            raise ValueError("A preceding receptor stage is incomplete: "+role)
    for prefix,folder,stem in (("sequence",RECEPTOR,"sequence_"),("matched",MATCHED,"")):
        final,profile=life.read(paths[prefix+"_final_audit"]),life.read(paths[prefix+"_profiles"])
        if final["evaluation_sha256"]!=life.sha(folder/(stem+"evaluation.json")) or final["selection_lock_sha256"]!=life.sha(folder/(stem+"selection_lock.json")):
            raise ValueError("A preceding final audit no longer binds its evaluation and selection")
        if profile["selection_lock_sha256"]!=final["selection_lock_sha256"]:
            raise ValueError("Preceding profiles and final audit use different selections")
    sys.path.insert(0,str(native_inputs.POCKET))
    sys.path.insert(0,str(RECEPTOR))
    import sequence_scoring
    import matched_scoring
    sequence_scoring.selection_context()
    matched_scoring.selection_context()
    return {role:life.artifact(path) for role,path in paths.items()}


def validation_comparison(row,cache,candidate):
    path=MATCHED/"validation_predictions"/(row["selected_id"]+".npz")
    expected=next(s for s in candidate["artifacts"] if (MATCHED/s["path"]).resolve()==path.resolve())
    if life.sha(path)!=expected["sha256"]:
        raise ValueError("The original matched validation vector changed")
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["ids"],np.asarray(cache.ids,dtype=str))
        np.testing.assert_array_equal(archive["truth"],cache.truth.numpy())
        original=archive["prediction"].copy()
    old_rmse=math.sqrt(math.fsum(float(x)**2 for x in original-cache.truth.numpy())/len(cache.ids))
    if abs(old_rmse-candidate["best_validation_rmse"])>1e-12:
        raise ValueError("Archived original validation predictions do not reproduce the selected score")
    current=cache.f0.double().numpy()*cache.target_sd+cache.target_mean
    delta=np.abs(current-original)
    if not np.all(delta<=.00003+.00003*np.abs(original)):
        raise ValueError("Native molecular cache exceeds the fixed CPU/GPU validation tolerance")
    new_rmse=math.sqrt(math.fsum(float(x)**2 for x in current-cache.truth.numpy())/len(cache.ids))
    return dict(archived_gpu_max_prediction_delta=float(delta.max()),archived_gpu_rmse=old_rmse,
                native_cpu_rmse=new_rmse,rmse_delta=new_rmse-old_rmse,parent_reselection=False),life.artifact(path)


def matched_row(seed,destination):
    selection_path=MATCHED/"selection_lock.json"
    choice=next(r for r in life.read(selection_path)["selections"] if (r["setting"],r["head"],r["seed"])==("ligand_contact","lma1",seed))
    selected_id=choice["selected_id"]
    row=dict(domain="ligand_contact",seed=seed,status="available" if selected_id else "missing_parent",
        reason=None if selected_id else "Both prescribed original ligand/contact first-order rates failed",
        parent_selection=life.artifact(selection_path),
        parent_candidates=[life.artifact(MATCHED/"candidates"/f"ligand_contact_lma1_seed{seed}_lr{i}.json") for i in range(2)],
        selected_id=selected_id,parent_checkpoint=life.artifact(MATCHED/"checkpoints"/(selected_id+".pt")) if selected_id else None,
        caches={},native_cache_audit=None)
    if control.original_parent_choice(row)!=selected_id:
        raise ValueError("Matched parent selection changed")
    if selected_id is None:
        return row
    candidate=life.read(MATCHED/"candidates"/(selected_id+".json"))
    sources={str(Path(__file__).resolve()):life.artifact(__file__)}
    for name in ("native_input_specification.md","native_inputs_cpu_audit.json","native_inputs.py","native_cache.py"):
        path=HERE/name;sources[str(path.resolve())]=life.artifact(path)
    checks=[]
    for split in ("train","val"):
        parent,layer_path,payload=native_inputs.parent_and_inputs(row,split)
        cache,check=native_cache.extract(parent,layer_path,payload,row["parent_checkpoint"]["sha256"])
        row["caches"][split]=native_cache.save(cache,destination/f"seed{seed}_{split}.pt")
        for bound in payload["sources"]+[row["caches"][split]]:
            sources[str(Path(bound["path"]).resolve())]=bound
        if split=="val":
            comparison,archive=validation_comparison(row,cache,candidate)
            check.update(comparison)
            sources[archive["path"]]=archive
        checks.append(dict(split=split,**check))
    for source in sources.values():
        life.verify_artifact(source)
    audit_path=destination/f"seed{seed}_native_cache_audit.json"
    life.atomic_json(audit_path,dict(passed=True,domain=row["domain"],seed=seed,
        parent_checkpoint_sha256=row["parent_checkpoint"]["sha256"],checks=checks,
        cache_content_digests={split:entry["content_digest"] for split,entry in row["caches"].items()},
        sources=list(sources.values()),new_test_predictions=False,residual_updates=0),immutable=True)
    row["native_cache_audit"]=life.artifact(audit_path)
    return row


def final_parent_audit(bundle_path,fragment_path,rows,preceding):
    sources=list(preceding.values())+[life.artifact(bundle_path),life.artifact(fragment_path),life.artifact(__file__),life.artifact(HERE/"native_delivery_cpu_audit.json")]
    sources += [r["native_cache_audit"] for r in rows if r["status"]=="available"]
    audit=dict(passed=True,parent_bundle=life.artifact(bundle_path),prescribed_parents=15,
        available_parents=sum(r["status"]=="available" for r in rows),
        matched_available_parents=sum(r["status"]=="available" for r in rows if r["domain"]=="ligand_contact"),
        exact_native_and_archived_validation_checks=True,sources=sources,new_test_predictions=False,residual_updates=0)
    life.atomic_json(DESTINATION/"parent_cache_audit.json",audit,immutable=True)


def build():
    if torch.get_num_threads()!=1 or torch.cuda.is_initialized():
        raise ValueError("Actual residual parent preparation requires one CPU thread without CUDA")
    preceding=preceding_studies()
    for name in ("campaign_control_cpu_audit_v2.json","native_inputs_cpu_audit.json","native_costs_cpu_audit.json","native_delivery_cpu_audit.json"):
        verify_receipt(HERE/name)
    bundle_path=DESTINATION/"parent_bundle.json"
    fragment_path=HERE/"parent_fragments/cubic_v2/fragment.json"
    if bundle_path.exists():
        bundle=control.validate_parents(bundle_path,control.RETAINED)
        if bundle["preceding_studies"]!=preceding:
            raise ValueError("The original parent bundle cites different preceding study results")
        final_parent_audit(bundle_path,fragment_path,bundle["parents"],preceding)
        verify_receipt(DESTINATION/"parent_cache_audit.json")
        return bundle_path
    fragment=life.read(fragment_path)
    for source in fragment["sources"]:
        life.verify_artifact(source)
    rows=list(fragment["parents"])
    for seed in range(42,47):
        record_path=DESTINATION/f"seed{seed}_parent.json"
        if record_path.exists():
            row=life.read(record_path)
            control.original_parent_choice(row)
            if row["status"]=="available":
                control.checked(row["native_cache_audit"])
                for split,record in row["caches"].items():
                    life.load_cache(record,split,row["parent_checkpoint"]["sha256"])
        else:
            row=matched_row(seed,DESTINATION)
            life.atomic_json(record_path,row,immutable=True)
        rows.append(row)
    producer_paths=[HERE/name for name in ("native_test_producer.py","native_inputs.py","native_cache.py","native_costs.py","native_profiles.py","native_parent_bundle.py","native_pipeline.py")]
    producer_paths += [Path(s["path"]) for s in fragment["sources"] if Path(s["path"]).suffix==".py"]
    producer_paths += [Path(s["path"]) for s in life.read(MATCHED/"implementation_lock.json")["sources"] if Path(s["path"]).suffix==".py"]
    bundle=dict(schema=1,scope=control.RETAINED,parents=rows,producer_sources=[life.artifact(p) for p in sorted(set(producer_paths))],
                preceding_studies=preceding,preparation_scope="Actual fitting/validation parents only; no retained residual activation or test prediction")
    life.atomic_json(bundle_path,bundle,immutable=True)
    control.validate_parents(bundle_path,control.RETAINED)
    final_parent_audit(bundle_path,fragment_path,rows,preceding)
    return bundle_path
