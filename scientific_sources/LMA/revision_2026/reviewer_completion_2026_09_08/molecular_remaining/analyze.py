"""Independent reconciliation and complete reporting for the remaining molecular block."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import campaign as c
import molecular_data as d
import molecular_models as m

HERE=m.HERE
b=d.base
FIELDS=("block","depth","head","width")
HEAD_LABELS=dict(mean_count="Mean/count",deepsets_raw70="Plain-sum Deep Sets (width 70)",
    cp_pool="CP",lma1="LMA k=1",lma2="LMA k=2",additive2="Additive k=2",ppgn="PPGN-style")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    return tuple(row[k] for k in FIELDS)


def identifier(config):
    return "_".join(map(str,key(config)))


def label(config):
    if config["block"]=="ppgn":
        return f'PPGN-style, width {config["width"]}'
    prefix={"exact_labels":"Annotation-filter GCN", "gat":"GAT", "original_gcn":"Archived full-label GCN"}[config["block"]]
    return f'{prefix}, depth {config["depth"]}: {HEAD_LABELS[config["head"]]}'


def independent_metrics(y,p):
    y,p=np.asarray(y,dtype=np.float64),np.asarray(p,dtype=np.float64)
    assert y.ndim==p.ndim==1 and y.shape==p.shape and len(y)>0
    assert np.isfinite(y).all() and np.isfinite(p).all()
    n=len(y)
    ym,pm=math.fsum(y)/n,math.fsum(p)/n
    a,bv=y-ym,p-pm
    sy,sp=math.fsum(a*a),math.fsum(bv*bv)
    return dict(rmse=math.sqrt(math.fsum((y-p)**2)/n),mae=math.fsum(abs(y-p))/n,
        pearson=math.fsum(a*bv)/math.sqrt(sy*sp) if sy>0 and sp>0 else None)


def describe(values):
    a=list(map(float,values))
    return dict(n=len(a),mean=statistics.mean(a) if a else None,
        sample_sd=statistics.stdev(a) if len(a)>1 else None,
        minimum=min(a) if a else None,maximum=max(a) if a else None)


def csv_write(name,rows,fields):
    with (HERE/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def config(block,head,depth=1,width=None):
    return dict(block=block,depth=depth,head=head,width=width)


def comparisons():
    pairs=[]
    def add(left,right,scope):
        pairs.append(dict(left=left,right=right,scope=scope))
    for head in m.EXACT_HEADS:
        add(config("exact_labels",head),config("original_gcn",head),"Annotation-filter sensitivity; archived comparator reused")
    for depth in (1,3):
        for other in ("lma1","cp_pool","deepsets_raw70"):
            add(config("gat","lma2",depth),config("gat",other,depth),"Readout procedures within the same GAT convention")
    for head in m.GAT_HEADS:
        add(config("gat",head,3),config("gat",head,1),"GAT depth sensitivity")
    for depth in (1,3):
        for head in m.GAT_HEADS:
            if depth==3 and head=="deepsets_raw70":
                continue  # No such archived GCN counterpart was fitted.
            add(config("gat",head,depth),config("original_gcn",head,depth),"Backbone/aggregation procedure contrast; archived comparator reused")
    add(config("ppgn","ppgn",3,32),config("ppgn","ppgn",3,16),"PPGN width sensitivity")
    for width in (16,32):
        for head in ("lma2","cp_pool"):
            add(config("ppgn","ppgn",3,width),config("original_gcn",head),"Different graph representation and readout; archived comparator reused")
    assert len(pairs)==31
    return pairs


def baseline_rows():
    needed={key(p["right"]) for p in comparisons() if p["right"]["block"]=="original_gcn"}
    rows,sources=[],[]
    for name,folder in [("core",m.CORE),("cardinality",m.CORE/"cardinality_controls")]:
        assert read(folder/"final_audit.json")["passed"]
        evaluation=read(folder/"evaluation.json")
        sources.append(dict(campaign=name,evaluation_sha256=b.sha(folder/"evaluation.json"),
            final_audit_sha256=b.sha(folder/"final_audit.json")))
        for old in evaluation["rows"]:
            cfg=config("original_gcn",old["head"],old["depth"])
            if key(cfg) not in needed:
                continue
            new=dict(**cfg,seed=old["seed"],status=old["status"],selected_id=old["selected_id"],
                test=old["test"],source_campaign=name)
            if old["status"]=="valid":
                path=folder/old["prediction_file"]
                assert b.sha(path)==old["prediction_sha256"]
                z=np.load(path)
                actual=independent_metrics(z["truth"],z["prediction"])
                for metric,value in actual.items():
                    assert old["test"][metric] is None if value is None else abs(value-old["test"][metric])<1e-12
                new.update(prediction_file=str(path),prediction_sha256=old["prediction_sha256"])
            rows.append(new)
    assert len(rows)==len(needed)*5 and len({(key(r),r["seed"]) for r in rows})==len(rows)
    return rows,sources


def audit():
    lock=c.verify()
    selection=c.verify_selection()
    evaluation=read(HERE/"evaluation.json")
    assert evaluation["selection_lock_sha256"]==b.sha(HERE/"selection_lock.json")
    assert datetime.fromisoformat(evaluation["evaluated_utc"])>=datetime.fromisoformat(selection["locked_utc"])
    assert len(evaluation["rows"])==90
    train={name:d.load("train",name) for name in ("full","exact_labels")}
    statistics_by_block={}
    for name,data in train.items():
        values=data["y"].double().numpy()
        mean=math.fsum(values)/len(values)
        sd=math.sqrt(math.fsum((values-mean)**2)/len(values))
        statistics_by_block[name]=(mean,sd,len(values))
    blobs={name:d.load(name,device="cuda") for name in ("val","test")}
    ids={name:np.asarray(data["ids"]) for name,data in blobs.items()}
    truths={name:data["y"].double().cpu().numpy() for name,data in blobs.items()}
    assert not set(train["full"]["ids"])&set(ids["val"])
    assert not set(train["full"]["ids"])&set(ids["test"])
    assert not set(ids["val"])&set(ids["test"])
    records={}
    val_delta=test_delta=scaling_delta=0.
    for item in selection["candidate_records"]:
        row=read(HERE/item["path"])
        spec=row["spec"]
        assert spec in lock["candidates"] and spec["id"] not in records
        records[spec["id"]]=row
        assert row["status"] in ("valid","failed")
        expected=statistics_by_block["exact_labels" if spec["block"]=="exact_labels" else "full"]
        scaling_delta=max(scaling_delta,abs(row["target_mean"]-expected[0]),abs(row["target_sd"]-expected[1]))
        assert scaling_delta<1e-13 and row["training_records"]==expected[2]
        assert row["effective_batch_size"]==256 and row["microbatch_size"]==(32 if spec["block"]=="ppgn" else 256)
        assert row["implementation_lock_sha256"]==b.sha(HERE/"implementation_lock.json")
        assert datetime.fromisoformat(lock["locked_utc"])<=datetime.fromisoformat(row["started_utc"])
        assert datetime.fromisoformat(row["finished_utc"])<=datetime.fromisoformat(selection["locked_utc"])
        assert [h["epoch"] for h in row["history"]]==list(range(1,row["epochs_run"]+1))
        if row["status"]!="valid":
            continue
        assert b.sha(HERE/row["checkpoint"])==row["checkpoint_sha256"]
        assert b.sha(HERE/row["validation_predictions"])==row["validation_prediction_sha256"]
        z=np.load(HERE/row["validation_predictions"])
        np.testing.assert_array_equal(z["ids"],ids["val"])
        np.testing.assert_array_equal(z["truth"],truths["val"])
        value=independent_metrics(z["truth"],z["prediction"])["rmse"]
        val_delta=max(val_delta,abs(value-row["best_validation_rmse"]))
        best=min((h for h in row["history"] if "validation_rmse" in h),key=lambda h:(h["validation_rmse"],h["epoch"]))
        assert (best["epoch"],best["validation_rmse"])==(row["best_epoch"],row["best_validation_rmse"])
    assert len(records)==180
    reloads=[]
    for choice,row in zip(selection["selections"],evaluation["rows"]):
        eligible=[records[i] for i in choice["candidate_ids"] if records[i]["status"]=="valid"]
        assert len(choice["candidate_ids"])==2
        assert all(key(records[i]["spec"])==key(choice) and records[i]["spec"]["seed"]==choice["seed"] for i in choice["candidate_ids"])
        best=min(eligible,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if eligible else None
        assert choice["selected_id"]==(best["spec"]["id"] if best else None)
        assert all(choice[k]==row[k] for k in choice)
        if best is None:
            assert row["status"]=="all_candidates_failed" and row["test"] is None
            continue
        assert row["status"]=="valid"
        assert b.sha(HERE/row["prediction_file"])==row["prediction_sha256"]
        z=np.load(HERE/row["prediction_file"])
        np.testing.assert_array_equal(z["ids"],ids["test"])
        np.testing.assert_array_equal(z["truth"],truths["test"])
        scores=independent_metrics(z["truth"],z["prediction"])
        for metric,value in scores.items():
            if value is None:
                assert row["test"][metric] is None
            else:
                test_delta=max(test_delta,abs(value-row["test"][metric]))
        checkpoint=torch.load(HERE/best["checkpoint"],map_location="cpu",weights_only=True)
        assert checkpoint["target_mean"]==best["target_mean"] and checkpoint["target_sd"]==best["target_sd"]
        model,candidate=c.restore(choice)
        assert candidate==best and m.count(model)==best["total_parameters"]==row["total_parameters"]
        saved_v=np.load(HERE/best["validation_predictions"])["prediction"]
        vp=d.predict(model,blobs["val"],best["target_mean"],best["target_sd"])
        tp=d.predict(model,blobs["test"],best["target_mean"],best["target_sd"])
        vd,td=float(np.max(abs(vp-saved_v))),float(np.max(abs(tp-z["prediction"])))
        assert max(vd,td)<1e-10
        reloads.append(dict(id=choice["selected_id"],validation_max_delta=vd,test_max_delta=td))
        del model
        torch.cuda.empty_cache()
    assert val_delta<1e-12 and test_delta<1e-12
    references,reference_sources=baseline_rows()
    for row in references:
        if row["status"]=="valid":
            z=np.load(row["prediction_file"])
            np.testing.assert_array_equal(z["ids"],ids["test"])
            np.testing.assert_array_equal(z["truth"],truths["test"])
    manifest=read(m.CORE/"study_manifest.json")
    first=Path(manifest["files"][0]["path"])
    root=next((r for r in (m.CORE,m.REVISION.parent) if (r/first).is_file() and b.sha(r/first)==manifest["files"][0]["sha256"]),None)
    assert root is not None
    for item in manifest["files"]:
        assert b.sha(root/item["path"])==item["sha256"],item["path"]
    output=dict(passed=True,audited_utc=b.now(),candidates=len(records),selections=90,
        valid_candidates=sum(r["status"]=="valid" for r in records.values()),
        failed_candidates=sum(r["status"]=="failed" for r in records.values()),valid_selected=len(reloads),
        max_independent_validation_metric_delta=val_delta,max_independent_test_metric_delta=test_delta,
        maximum_target_scaling_delta=scaling_delta,target_scaling_tolerance=1e-13,
        checkpoint_reloads=reloads,archived_reference_predictors=len(references),reference_sources=reference_sources,
        previous_neural_manifest_entries_unchanged=len(manifest["files"]),analysis_source_sha256=b.sha(__file__),
        qualification="Numerical/bookkeeping reconciliation, not a historical reproduction or global optimization certificate")
    b.write_json(HERE/"final_audit.json",output)
    print(json.dumps({k:v for k,v in output.items() if k!="checkpoint_reloads"}),flush=True)


def overlap():
    c.verify_selection()
    assert read(HERE/"final_audit.json")["passed"]
    old=read(HERE/"overlap_sensitivity.json")
    assert old["passed"] and old["manifest_sha256"]==b.sha(d.DATA/"sample_manifest.csv")
    _,entries=d.exact_ids()
    by_id={row["pdbid"]:row for row in entries}
    fitting={r["canonical_smiles"] for r in entries if r["split"] in ("train","val")}
    groups=dict(full={r["pdbid"] for r in entries if r["split"]=="test"},
        absent_from_train_and_val={r["pdbid"] for r in entries if r["split"]=="test" and r["canonical_smiles"] not in fitting})
    groups["shared_with_train_or_val"]=groups["full"]-groups["absent_from_train_and_val"]
    for name,members in groups.items():
        assert members==set(old["subsets"][name])
    rows=[]
    for selected in read(HERE/"evaluation.json")["rows"]:
        result={k:selected[k] for k in (*FIELDS,"seed","status","selected_id")}
        if selected["status"]!="valid":
            result["groups"]=None
        else:
            path=HERE/selected["prediction_file"]
            assert b.sha(path)==selected["prediction_sha256"]
            z=np.load(path)
            assert len(z["ids"])==len(groups["full"]) and set(z["ids"].tolist())==groups["full"]
            expected=np.asarray([float(by_id[str(i)]["y"]) for i in z["ids"]],dtype=np.float32).astype(float)
            np.testing.assert_array_equal(z["truth"],expected)
            scores={}
            for name,members in groups.items():
                mask=np.asarray([str(i) in members for i in z["ids"]])
                scores[name]=dict(records=int(mask.sum()),**independent_metrics(z["truth"][mask],z["prediction"][mask]))
            assert abs(scores["full"]["rmse"]-selected["test"]["rmse"])<1e-12
            result.update(groups=scores,prediction_sha256=selected["prediction_sha256"])
        rows.append(result)
    assert len(rows)==90
    summaries=[]
    for cfg in m.CONFIGS:
        for name,members in groups.items():
            values=[r["groups"][name]["rmse"] for r in rows if key(r)==key(cfg) and r["status"]=="valid"]
            desc=describe(values)
            summaries.append(dict(**cfg,subset=name,test_records=len(members),valid_seeds=len(values),
                mean_rmse=desc["mean"],sample_sd_rmse=desc["sample_sd"]))
    output=dict(passed=True,created_utc=b.now(),rows=rows,summaries=summaries,
        subsets={name:sorted(members) for name,members in groups.items()},
        manifest_sha256=b.sha(d.DATA/"sample_manifest.csv"),analysis_source_sha256=b.sha(__file__),
        prior_240_predictor_artifact_sha256=b.sha(HERE/"overlap_sensitivity.json"),
        qualification="Metadata-only sensitivity on the reused test; no refitting, reselection or protein-level decontamination")
    b.write_json(HERE/"remaining_overlap_sensitivity.json",output)
    csv_write("remaining_overlap_sensitivity.csv",summaries,list(summaries[0]))
    print(json.dumps(dict(stage="remaining_overlap_complete",selections=90,subset_sizes={k:len(v) for k,v in groups.items()})),flush=True)


def measure(model,state,inputs,target,mode,lr,microbatch):
    times,baselines,peaks,increments=[],[],[],[]
    for index in range(5):
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4) if mode=="train" else None
        torch.manual_seed(95000+index)
        if mode=="train":
            model.train()
            def operation():
                optimizer.zero_grad(set_to_none=True)
                c.accumulated_backward(model,inputs,target,microbatch)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step()
        else:
            model.eval()
            def operation():
                with torch.inference_mode():
                    return model(*inputs)
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for _ in range(20):
            operation()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000/20)
        peak=torch.cuda.max_memory_allocated()
        baselines.append(baseline/2**20)
        peaks.append(peak/2**20)
        increments.append((peak-baseline)/2**20)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite profiling update")
        del optimizer
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    model.eval()
    return dict(block_average_ms=times,median_block_average_ms=statistics.median(times),
        minimum_block_average_ms=min(times),maximum_block_average_ms=max(times),
        baseline_allocated_mib=baselines,peak_allocated_mib=peaks,incremental_peak_mib=increments)


def profile():
    choices=c.verify_selection()
    assert read(HERE/"final_audit.json")["passed"]
    blob=d.load("train")
    n=int(blob["mask"][:64].sum(1).max())
    inputs=(blob["X"][:64,:n].cuda(),blob["mask"][:64,:n].cuda(),blob["adj"][:64,:n,:n].cuda())
    rows=[]
    for choice in choices["selections"]:
        if choice["seed"]!=42:
            continue
        if choice["selected_id"] is None:
            rows.append(dict(**choice,status="no_selected_checkpoint",timings=[]))
            continue
        model,candidate=c.restore(choice)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            reference=model(*inputs).cpu()
        target=((blob["y"][:64].double()-candidate["target_mean"])/candidate["target_sd"]).float().cuda()
        microbatch=32 if choice["block"]=="ppgn" else 256
        timings=[]
        for batch in (1,64):
            for mode in ("eval","train"):
                try:
                    result=measure(model,state,tuple(x[:batch] for x in inputs),target[:batch],mode,candidate["spec"]["lr"],microbatch)
                    result["status"]="profiled"
                except (FloatingPointError,torch.OutOfMemoryError) as exc:
                    result=dict(status="failed",error_type=type(exc).__name__,error=str(exc))
                    model.load_state_dict(state)
                    model.zero_grad(set_to_none=True)
                    model.eval()
                    torch.cuda.empty_cache()
                timings.append(dict(batch_size=batch,padded_atoms=n,
                    mode="full_optimizer_step" if mode=="train" else "evaluation_forward",
                    microbatch_size=min(batch,microbatch) if mode=="train" else batch,**result))
        with torch.no_grad():
            delta=float((reference-model(*inputs).cpu()).abs().max())
        assert delta==0. and b.sha(HERE/candidate["checkpoint"])==choice["checkpoint_sha256"]
        rows.append(dict(**choice,status="profiled" if all(t["status"]=="profiled" for t in timings) else "partial_profile_failure",
            total_parameters=m.count(model),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            state_tensor_bytes=sum(t.numel()*t.element_size() for t in state.values()),
            checkpoint_bytes=(HERE/candidate["checkpoint"]).stat().st_size,
            restored_prediction_max_delta=delta,timings=timings))
        b.write_json(HERE/"profiles.json",dict(complete=False,rows=rows))
        del model,target
        torch.cuda.empty_cache()
    assert len(rows)==18
    output=dict(complete=True,profiled_utc=b.now(),rows=rows,analysis_source_sha256=b.sha(__file__),
        hardware=torch.cuda.get_device_name(),dtype="float32",tf32=False,cpu_threads=4,
        workload_ids=blob["ids"][:64],valid_atom_counts=blob["mask"][:64].sum(1).tolist(),padded_atoms=n,
        blocks=5,warmups_per_block=10,repetitions_per_block=20,
        boundary="Preloaded GPU inputs; whole model. Training includes zero_grad, standardized MSE, accumulated backward, clipping and AdamW. Excludes loading, transfers, target de-standardization, validation and the once-per-epoch scheduler. Batch64 PPGN training uses two 32-example microbatches.",
        interpretation="Medians of block-average times on one GPU; not per-request latency quantiles or device-independent costs")
    b.write_json(HERE/"profiles.json",output)
    print(json.dumps(dict(stage="remaining_molecular_profiles_complete",procedures=len(rows),
        workloads=sum(len(r["timings"]) for r in rows),failed_workloads=sum(t["status"]!="profiled" for r in rows for t in r["timings"]))),flush=True)


def report():
    c.verify_selection()
    assert read(HERE/"final_audit.json")["passed"]
    profiles=read(HERE/"profiles.json")
    assert profiles["complete"] and read(HERE/"remaining_overlap_sensitivity.json")["passed"]
    allrows=read(HERE/"evaluation.json")["rows"]
    references,reference_sources=baseline_rows()
    rows=[r for r in allrows if r["status"]=="valid"]
    groups=[]
    for cfg in m.CONFIGS:
        values=[r for r in rows if key(r)==key(cfg)]
        groups.append(dict(**cfg,label=label(cfg),planned_seeds=5,valid_seeds=len(values),
            metrics={metric:describe(r["test"][metric] for r in values if r["test"][metric] is not None) for metric in ("rmse","mae","pearson")},
            total_parameters=values[0]["total_parameters"] if values else None))
    index={(key(r),r["seed"]):r for r in allrows+references}
    assert len(index)==len(allrows)+len(references)
    contrasts=[]
    for pair in comparisons():
        differences=[]
        unavailable=[]
        for seed in c.SEEDS:
            left,right=index[(key(pair["left"]),seed)],index[(key(pair["right"]),seed)]
            if left["status"]==right["status"]=="valid":
                differences.append(dict(seed=seed,left_rmse=left["test"]["rmse"],right_rmse=right["test"]["rmse"],
                    difference=left["test"]["rmse"]-right["test"]["rmse"]))
            else:
                unavailable.append(dict(seed=seed,left_status=left["status"],right_status=right["status"]))
        contrasts.append(dict(**pair,rows=differences,unavailable_pairs=unavailable,
            summary=describe(v["difference"] for v in differences),
            left_lower=sum(v["difference"]<0 for v in differences),left_higher=sum(v["difference"]>0 for v in differences)))
    candidates=[read(HERE/item["path"]) for item in c.verify_selection()["candidate_records"]]
    output=dict(generated_utc=b.now(),groups=groups,paired_contrasts=contrasts,
        candidate_count=len(candidates),valid_candidates=sum(r["status"]=="valid" for r in candidates),
        failed_candidates=sum(r["status"]=="failed" for r in candidates),selection_count=90,valid_selected=len(rows),
        aggregate_fit_seconds=sum(r["wall_seconds"] for r in candidates),epoch_cap_candidates=sum(r["reached_epoch_cap"] for r in candidates),
        reference_sources=reference_sources,analysis_source_sha256=b.sha(__file__),
        interpretation="Descriptive optimizer/selection repetitions on one reused ligand-only split; procedural differences, no population equivalence or causal attribution")
    b.write_json(HERE/"summary.json",output)
    seeds=[]
    for row in allrows:
        seeds.append({k:row[k] for k in (*FIELDS,"seed","status","selected_id")}|dict(
            selected_lr=row.get("selected_lr"),best_epoch=row.get("best_epoch"),epochs_run=row.get("epochs_run"),
            training_records=row.get("training_records"),rmse=row["test"]["rmse"] if row["test"] else None,
            mae=row["test"]["mae"] if row["test"] else None,pearson=row["test"]["pearson"] if row["test"] else None))
    csv_write("selected_seeds.csv",seeds,list(seeds[0]))
    diffs=[dict(left=identifier(p["left"]),right=identifier(p["right"]),**v) for p in contrasts for v in p["rows"]]
    csv_write("paired_differences.csv",diffs,["left","right","seed","left_rmse","right_rmse","difference"])
    failures=[dict(id=r["spec"]["id"],status=r["status"],error_type=r.get("error_type"),error=r.get("error")) for r in candidates if r["status"]!="valid"]
    csv_write("failed_candidates.csv",failures,["id","status","error_type","error"])
    def format_summary(desc):
        return "unavailable" if not desc["n"] else f'{desc["mean"]:.6f}'+(f' ± {desc["sample_sd"]:.6f}' if desc["sample_sd"] is not None else " (one valid seed)")
    table=[f'| {g["label"]} | {format_summary(g["metrics"]["rmse"])} | {g["total_parameters"]} | {g["valid_seeds"]}/5 |' for g in groups]
    paired=[f'| {label(p["left"])} − {label(p["right"])} | {format_summary(p["summary"])} | {p["left_lower"]}/{p["summary"]["n"]} |' for p in contrasts]
    cost=[]
    for row in profiles["rows"]:
        by_mode={t["mode"]:t for t in row["timings"] if t["batch_size"]==64}
        times=[]
        for mode in ("evaluation_forward","full_optimizer_step"):
            t=by_mode.get(mode)
            times.append(f'{t["median_block_average_ms"]:.3f}' if t and t["status"]=="profiled" else "unavailable")
        cost.append(f'| {label(row)} | {times[0]} | {times[1]} |')
    text="\n".join([
        "# Remaining molecular comparisons", "",
        f'{output["valid_candidates"]}/180 candidates were numerically valid and {len(rows)}/90 selected predictors were scored. Every candidate record and all 90 validation choices were locked before scoring this block. Aggregate fitting time was {output["aggregate_fit_seconds"]/60:.2f} minutes. Candidate failures and missing pairs remain in the exported tables.', "",
        "| Procedure | Test RMSE, mean ± sample SD | Parameters | Valid seeds |", "|---|---:|---:|---:|",*table,"",
        "All procedures use the same ligand-only molecular representation and reused validation/test partition. The annotation-filter block retains 7,143 fitting records passing the declared rule and recomputes training-only scaling. GAT and PPGN use the full 7,384 fitting rows. Every method has five prescribed seeds and two candidate learning rates. Validation determines both the checkpoint and learning rate; no cross-family winner is selected for reporting.","",
        "| Prespecified paired contrast | RMSE difference, mean ± sample SD | Left lower |", "|---|---:|---:|",*paired,"",
        "Negative differences favor the left procedure. These are descriptive paired optimizer/selection repetitions on a fixed reused split. The archived full-label GCN controls are reused fits, with no archived depth-three plain-sum width70 counterpart. Changes in backbone, representation, training sample and capacity are stated in the comparison scope; these differences do not identify a causal mechanism or population equivalence.","",
        "The separate overlap table retains all 90 new selections on the full 2,171 records, the 2,021 with no canonical ligand in train or validation, and the complementary 150. This metadata-only evaluation does not refit or reselect checkpoints and does not repair protein-level dependence. The earlier 240-predictor sensitivity artifact remains unchanged.","",
        "| Complete model, batch 64 | Evaluation ms | AdamW step ms |", "|---|---:|---:|",*cost,"",
        "Costs use the same first 64 full-training records and common padding for every procedure. Each of five blocks restores the selected checkpoint, runs ten warmups and 20 timed operations, and records actual allocated-memory baselines and peaks. PPGN's batch 64 optimizer step comprises two accumulated 32-record microbatches. Inputs reside on the RTX 3060 Laptop GPU; float32 operations disable TF32. The timed training step includes gradient reset, standardized MSE, backward, clipping and AdamW. Loading, transfers, target de-standardization, validation and the once-per-epoch scheduler are excluded. Results are block-average times, not per-request quantiles. Full batch-one results and storage counts are in profiles.json.","",
        "Independent audits reconcile all valid candidate validation scores, choices, train-only scaling, selected test metrics and regenerated validation/test predictions. The prior neural manifest is checked unchanged. These checks establish numerical consistency within the reconstruction; they do not restore the unavailable historical experiment or establish global optimization.","",
        "The GAT adaptation normalizes over neighborhood support, with four concatenated heads and explicit handling of padded queries. PPGN uses pair tensors, masked pointwise maps, channelwise matrix products and valid diagonal/off-diagonal max pooling. Its finite width carries no claim of attaining an existential WL guarantee. The separate loop-free C6/two-triangles trace check is not a learned molecular result or a calculation on normalized, self-looped molecular adjacency.","",
        "Original source provenance and protein/pocket inputs remain unresolved. The annotation filter certifies a syntax rule, not error-free affinities. These results must be integrated alongside earlier unfavorable results and the synthetic studies; they do not establish a general higher-order advantage.",""
    ])
    (HERE/"report.md").write_text(text,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,8.5))
    for i,g in enumerate(groups):
        values=[r["test"]["rmse"] for r in rows if key(r)==key(g)]
        if values:
            ax.scatter(values,np.full(len(values),i),s=20,color="#64748b",alpha=.7)
            desc=g["metrics"]["rmse"]
            ax.errorbar(desc["mean"],i,xerr=desc["sample_sd"] if desc["sample_sd"] is not None else 0,fmt="o",color="#155e75",capsize=3)
    ax.set_yticks(range(len(groups)),[f'{g["label"]} [{g["valid_seeds"]}/5]' for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (lower is better); points = seeds, bars = sample SD")
    ax.set_title("Remaining molecular comparisons on the reused ligand-only split")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/"all_procedures.png",dpi=180)
    fig.savefig(HERE/"all_procedures.svg")
    plt.close(fig)
    print(json.dumps(dict(stage="remaining_molecular_report_complete",procedures=len(groups),contrasts=len(contrasts),valid_selections=len(rows))),flush=True)


def preflight():
    pairs=comparisons()
    references,sources=baseline_rows()
    expected=independent_metrics([0.,1.,2.],[1.,1.,1.])
    assert expected==dict(rmse=math.sqrt(2/3),mae=2/3,pearson=None)
    new={key(cfg) for cfg in m.CONFIGS}
    old={key(r) for r in references}
    for pair in pairs:
        assert key(pair["left"]) in new and key(pair["right"]) in new|old
    payload=dict(passed=True,checked_utc=b.now(),planned_configurations=len(new),planned_contrasts=len(pairs),
        reused_reference_predictors=len(references),reference_sources=sources,analysis_source_sha256=b.sha(__file__),
        qualification="Checks analysis configuration, archived-reference integrity and an independent metric fixture; post-fit audits have not run")
    b.write_json(HERE/"analysis_prefit_audit.json",payload)
    print(json.dumps(payload),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["preflight","audit","overlap","profile","report"])
    args=parser.parse_args()
    if args.stage!="preflight":
        b.configure()
    globals()[args.stage]()
