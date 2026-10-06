"""Independent outcome reconciliation, complete-model costs and full reporting."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import numpy as np
import torch
import gate_models as gm
import campaign
import study as base
from final_audit import independent_metrics

HERE=gm.HERE
LABELS={"legacy1":"Original k1", "legacy2_product":"Original k2 product",
        "legacy2_additive":"Original k2 additive", "separated2_product_zero":"Separate product, alpha init 0",
        "separated2_additive_zero":"Separate additive, alpha init 0",
        "separated2_product_one":"Separate product, alpha init 1",
        "separated2_additive_one":"Separate additive, alpha init 1"}
PAIRS=[("legacy2_product","legacy1"),
       ("separated2_product_zero","legacy2_product"),
       ("separated2_product_one","legacy2_product"),
       ("separated2_product_zero","separated2_product_one"),
       ("legacy2_product","legacy2_additive"),
       ("separated2_product_zero","separated2_additive_zero"),
       ("separated2_product_one","separated2_additive_one")]


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def audit():
    lock=campaign.verify()
    selections=campaign.verify_selection()
    evaluation=read("evaluation.json")
    assert evaluation["selection_lock_sha256"]==base.sha(HERE/"selection_lock.json")
    assert datetime.fromisoformat(evaluation["evaluated_utc"])>=datetime.fromisoformat(selections["locked_utc"])
    assert len(evaluation["rows"])==35
    train=torch.load(base.DATA/"pdbbind_train.pt",weights_only=True,map_location="cpu")
    val=base.load_split("val")
    test=base.load_split("test")
    mean=float(train["y"].double().mean())
    sd=float(train["y"].double().std(unbiased=False))
    ids={s:np.asarray(b["ids"]) for s,b in (("val",val),("test",test))}
    truth={s:b["y"].double().cpu().numpy() for s,b in (("val",val),("test",test))}
    assert not set(train["ids"])&set(val["ids"])
    assert not set(train["ids"])&set(test["ids"])
    assert not set(val["ids"])&set(test["ids"])
    records={}
    val_delta=test_delta=scaling_delta=0.
    for item in selections["candidate_records"]:
        record=read(item["path"])
        spec=record["spec"]
        records[spec["id"]]=record
        assert spec in lock["candidates"]
        scaling_delta=max(scaling_delta,abs(record["target_mean"]-mean),abs(record["target_sd"]-sd))
        assert scaling_delta<1e-13
        assert record["implementation_lock_sha256"]==base.sha(HERE/"implementation_lock.json")
        assert datetime.fromisoformat(lock["locked_utc"])<=datetime.fromisoformat(record["started_utc"])
        assert datetime.fromisoformat(record["finished_utc"])<=datetime.fromisoformat(selections["locked_utc"])
        if record["status"]!="valid":
            continue
        assert base.sha(HERE/record["checkpoint"])==record["checkpoint_sha256"]
        assert base.sha(HERE/record["validation_predictions"])==record["validation_prediction_sha256"]
        z=np.load(HERE/record["validation_predictions"])
        assert np.array_equal(z["ids"],ids["val"]) and np.array_equal(z["truth"],truth["val"])
        score=independent_metrics(z["truth"],z["prediction"])["rmse"]
        val_delta=max(val_delta,abs(score-record["best_validation_rmse"]))
        best=min((r for r in record["history"] if "validation_rmse" in r),
                 key=lambda r:(r["validation_rmse"],r["epoch"]))
        assert (best["epoch"],best["validation_rmse"])==(record["best_epoch"],record["best_validation_rmse"])
        assert [r["epoch"] for r in record["history"]]==list(range(1,record["epochs_run"]+1))
    reloads=[]
    for choice,row in zip(selections["selections"],evaluation["rows"]):
        valid=[records[i] for i in choice["candidate_ids"] if records[i]["status"]=="valid"]
        best=min(valid,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if valid else None
        assert choice["selected_id"]==(best["spec"]["id"] if best else None)
        assert all(choice[k]==row[k] for k in choice)
        if best is None:
            assert row["status"]=="all_candidates_failed"
            continue
        assert base.sha(HERE/row["prediction_file"])==row["prediction_sha256"]
        z=np.load(HERE/row["prediction_file"])
        assert np.array_equal(z["ids"],ids["test"]) and np.array_equal(z["truth"],truth["test"])
        scores=independent_metrics(z["truth"],z["prediction"])
        for key,value in scores.items():
            if value is None:
                assert row["test"][key] is None
            else:
                test_delta=max(test_delta,abs(value-row["test"][key]))
        checkpoint=torch.load(HERE/best["checkpoint"],weights_only=True,map_location="cpu")
        assert checkpoint["spec"]==best["spec"]
        model=gm.build(choice["seed"],choice["head"],1).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        vp=base.predict(model,val,checkpoint["target_mean"],checkpoint["target_sd"])
        tp=base.predict(model,test,checkpoint["target_mean"],checkpoint["target_sd"])
        saved_v=np.load(HERE/best["validation_predictions"])["prediction"]
        vd=float(np.max(np.abs(vp-saved_v)))
        td=float(np.max(np.abs(tp-z["prediction"])))
        assert max(vd,td)<1e-10
        gates=model.head.layers[0].order_gates.detach().cpu().tolist()
        assert gates==row["selected_gate_coefficients"]
        reloads.append(dict(id=choice["selected_id"],validation_max_delta=vd,test_max_delta=td))
        del model
    assert val_delta<1e-12 and test_delta<1e-12
    previous=json.loads((gm.CORE/"study_manifest.json").read_text(encoding="utf-8"))
    files=previous["files"]
    # Determine and validate the declared manifest base using its path format.
    first=Path(files[0]["path"])
    roots=[gm.CORE,gm.REVISION.parent]
    root=next((r for r in roots if (r/first).is_file() and base.sha(r/first)==files[0]["sha256"]),None)
    assert root is not None
    for item in files:
        assert base.sha(root/item["path"])==item["sha256"],item["path"]
    output=dict(audited_utc=base.now(),passed=True,candidates=len(records),
        valid_candidates=sum(r["status"]=="valid" for r in records.values()),
        failed_candidates=sum(r["status"]!="valid" for r in records.values()),selections=35,
        valid_selected=len(reloads),max_independent_validation_metric_delta=val_delta,
        max_independent_test_metric_delta=test_delta,checkpoint_reloads=reloads,
        maximum_cpu_gpu_target_scaling_delta=scaling_delta,target_scaling_tolerance=1e-13,
        previous_neural_manifest_entries_unchanged=len(files),
        audited_analysis_source_sha256=base.sha(__file__),
        qualification="Numerical and bookkeeping audit; not statistical confirmation or a global optimization certificate")
    base.write_json(HERE/"final_audit.json",output)
    print(json.dumps({k:v for k,v in output.items() if k!="checkpoint_reloads"}),flush=True)


def profile():
    from profile_models import measure_blocks
    choices=campaign.verify_selection()
    assert read("final_audit.json")["passed"]
    blob=torch.load(base.DATA/"pdbbind_train.pt",weights_only=True,map_location="cpu")
    n=int(blob["mask"][:64].sum(1).max())
    x=blob["X"][:64,:n].cuda()
    mask=blob["mask"][:64,:n].cuda()
    adj=blob["adj"][:64,:n,:n].cuda()
    rows=[]
    for choice in choices["selections"]:
        if choice["seed"]!=42:
            continue
        if choice["selected_id"] is None:
            rows.append(dict(**choice,status="no_selected_checkpoint"))
            continue
        candidate=read("candidates/"+choice["selected_id"]+".json")
        path=HERE/candidate["checkpoint"]
        checkpoint=torch.load(path,weights_only=True,map_location="cpu")
        state=checkpoint["state_dict"]
        model=gm.build(42,choice["head"],1).cuda().eval()
        model.load_state_dict(state)
        with torch.no_grad():
            reference=model(x,mask,adj).clone()
        target=((blob["y"][:64].double()-checkpoint["target_mean"])/checkpoint["target_sd"]).float().cuda()
        timings=[]
        for batch in (1,64):
            for mode in ("eval","train"):
                result=measure_blocks(model,state,x[:batch],mask[:batch],adj[:batch],target[:batch],mode,candidate["spec"]["lr"])
                # There is no sampled branch in this campaign.
                result["mode"]="full_optimizer_step" if mode=="train" else "evaluation_forward"
                timings.append(dict(batch_size=batch,padded_atoms=n,**result))
        with torch.no_grad():
            delta=float((reference-model(x,mask,adj)).abs().max())
        assert delta==0. and base.sha(path)==choice["checkpoint_sha256"]
        rows.append(dict(**choice,status="profiled",restored_prediction_max_delta=delta,timings=timings))
        base.write_json(HERE/"profiles.json",dict(complete=False,rows=rows))
        del model
        torch.cuda.empty_cache()
    base.write_json(HERE/"profiles.json",dict(complete=True,profiled_utc=base.now(),rows=rows,
        analysis_source_sha256=base.sha(__file__),measurement_source_sha256=base.sha(gm.CORE/"profile_models.py"),
        hardware=torch.cuda.get_device_name(),dtype="float32",tf32=False,cpu_threads=4,
        workload_ids=blob["ids"][:64],valid_atom_counts=blob["mask"][:64].sum(1).tolist(),padded_atoms=n,
        blocks=5,repetitions_per_block=20,warmups_per_block=10,
        boundary="Preloaded GPU inputs, full model; train includes zero_grad, MSE, backward, clipping and AdamW. Transfers and target de-standardization excluded.",
        interpretation="Median of block-average times; not per-request latency quantiles. Both branches execute regardless of gate."))
    print(json.dumps(dict(stage="gate_profiles_complete",procedures=len(rows),workloads=sum(len(r.get("timings",[])) for r in rows))),flush=True)


def describe(values):
    a=[float(v) for v in values]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,
                sample_sd=statistics.stdev(a) if len(a)>1 else None,
                minimum=min(a) if a else None,maximum=max(a) if a else None)


def write_csv(name,rows):
    with (HERE/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report():
    assert read("final_audit.json")["passed"] and read("profiles.json")["complete"]
    evaluation=read("evaluation.json")
    allrows=evaluation["rows"]
    rows=[r for r in allrows if r["status"]=="valid"]
    groups=[]
    for head in gm.HEADS:
        subset=[r for r in rows if r["head"]==head]
        groups.append(dict(head=head,label=LABELS[head],planned_seeds=5,
            rmse=describe([r["test"]["rmse"] for r in subset]),
            mae=describe([r["test"]["mae"] for r in subset]),
            pearson=describe([r["test"]["pearson"] for r in subset if r["test"]["pearson"] is not None]),
            total_parameters=subset[0]["total_parameters"] if subset else None))
    pairs=[]
    for a,b in PAIRS:
        ar={r["seed"]:r for r in rows if r["head"]==a}
        br={r["seed"]:r for r in rows if r["head"]==b}
        values=[dict(seed=s,difference=ar[s]["test"]["rmse"]-br[s]["test"]["rmse"]) for s in base.SEEDS if s in ar and s in br]
        pairs.append(dict(left=a,right=b,rows=values,summary=describe([r["difference"] for r in values]),
            left_lower=sum(r["difference"]<0 for r in values),left_higher=sum(r["difference"]>0 for r in values)))
    candidates=[read("candidates/"+s["id"]+".json") for s in campaign.verify()["candidates"]]
    output=dict(generated_utc=base.now(),groups=groups,paired_contrasts=pairs,
        valid_candidates=sum(r["status"]=="valid" for r in candidates),candidate_count=len(candidates),
        valid_selected=len(rows),selection_count=len(allrows),
        aggregate_fit_seconds=sum(r["wall_seconds"] for r in candidates),
        epoch_cap_candidates=sum(r["reached_epoch_cap"] for r in candidates),
        interpretation="Descriptive paired optimizer/validation-selection repetitions on a repeatedly used split; no population ranking or causal attribution")
    base.write_json(HERE/"summary.json",output)
    seeds=[dict(head=r["head"],seed=r["seed"],selected_id=r["selected_id"],selected_lr=r["selected_lr"],
        best_epoch=r["best_epoch"],epochs_run=r["selected_epochs"],rmse=r["test"]["rmse"],mae=r["test"]["mae"],
        pearson=r["test"]["pearson"],gate1=r["selected_gate_coefficients"][0],
        gate2=r["selected_gate_coefficients"][1] if len(r["selected_gate_coefficients"])>1 else None) for r in rows]
    write_csv("selected_seeds.csv",seeds)
    write_csv("paired_differences.csv",[dict(left=r["left"],right=r["right"],**v) for r in pairs for v in r["rows"]])
    table=[]
    for g in groups:
        stats=g["rmse"]
        value=f'{stats["mean"]:.6f} ± {stats["sample_sd"]:.6f}' if stats["n"]==5 else str(stats)
        table.append(f'| {g["label"]} | {value} | {g["total_parameters"]} | {stats["n"]}/5 |')
    contrasts=[f'| {LABELS[r["left"]]} − {LABELS[r["right"]]} | {r["summary"]["mean"]:+.6f} | {r["left_lower"]}/{r["summary"]["n"]} |'
               for r in pairs if r["summary"]["n"]]
    costs=[]
    for p in read("profiles.json")["rows"]:
        if p["status"]!="profiled":
            continue
        times={t["mode"]:t for t in p["timings"] if t["batch_size"]==64}
        costs.append(f'| {LABELS[p["head"]]} | {times["evaluation_forward"]["median_block_average_ms"]:.3f} | {times["full_optimizer_step"]["median_block_average_ms"]:.3f} |')
    text="\n".join([
        "# Gate and readout experiment", "",
        f'{output["valid_candidates"]}/{len(candidates)} candidates were numerically valid; {len(rows)}/35 selected predictors were scored. All 70 candidate records and 35 validation-based choices were locked before this campaign\'s test scoring. Total candidate fitting time was {output["aggregate_fit_seconds"]/60:.2f} minutes.', "",
        "| Procedure | Test RMSE, mean ± sample SD | Parameters | Valid seeds |", "|---|---:|---:|---:|",*table,"",
        "| Prespecified paired contrast | Mean RMSE difference | Left procedure lower |", "|---|---:|---:|",*contrasts,"",
        "Negative differences favor the left procedure. The CSV files retain every selected seed, gate coefficient and prespecified paired difference. Candidate histories and both learning rates are retained, including any failures. No favorable initialization is selected across procedure families.","",
        "The comparison uses one-layer GCN, M=8 and the same ligand-only 7,384/958/2,171 partition. For each seed, every retained first-order, encoder and regressor tensor starts from the same untrained k1 model. This differs from the earlier k2 initialization protocol; results are not pooled across campaigns. The two separated initializations fit the same architecture. The original-versus-separated contrast jointly changes cross-order normalization, gate placement and readout scale. It does not isolate denominator dilution or establish the cause of the historical molecular error.","",
        "The pre-fit audit checks exact off-state recovery within dtype tolerances, paired tensors, independent equations and gradients, invariance and masking. Alpha is an unconstrained trained scalar; its selected value is scale-dependent and is not a scale-independent importance measure. Initial functional equality does not imply matching training trajectories. There is no new theorem in the separated gate construction.","",
        "| Complete model, batch 64 | Evaluation ms | AdamW step ms |", "|---|---:|---:|",*costs,"",
        "Costs are medians of five block-average times on the common first-64-training-graph workload, with 10 warmups and 20 measured operations per block. Inputs reside on the RTX 3060 Laptop GPU; float32 operations disable TF32. A training operation includes loss, backward, global clipping and AdamW. Both branches execute even when alpha is zero. Profiles also retain batch-one workloads, allocated-memory baselines and peaks, and verify restoration of the selected checkpoints.","",
        "Independent calculations recheck every valid candidate's validation metric and all selected test metrics. Both validation and test predictions are regenerated from every selected checkpoint. The audit also verifies that the prior neural archive's manifest entries remain unchanged.","",
        "These are development-informed optimizer/selection repetitions on a reused split. The 150 validation–test shared canonical ligands, 241 annotation-qualified training labels and absence of protein/pocket features remain limitations. This block addresses part of R6.4, R8.5 and R10.2; it does not complete routing, balancing, bucket sensitivity, ten-seed synthetic reliability or the remaining comparators.",""
    ])
    (HERE/"report.md").write_text(text,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for i,g in enumerate(groups):
        data=[r["test"]["rmse"] for r in rows if r["head"]==g["head"]]
        if data:
            ax.scatter(data,np.full(len(data),i),color="#64748b",s=22,alpha=.7)
            if len(data)>1:
                ax.errorbar(g["rmse"]["mean"],i,xerr=g["rmse"]["sample_sd"],fmt="o",color="#155e75",capsize=3)
    ax.set_yticks(range(len(groups)),[g["label"] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (lower is better); points = seeds, bars = sample SD")
    ax.set_title("Gate/readout comparison on the reused ligand-only split")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/"all_procedures.png",dpi=180)
    fig.savefig(HERE/"all_procedures.svg")
    plt.close(fig)
    print(json.dumps(output),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","profile","report"])
    args=parser.parse_args()
    base.configure()
    {"audit":audit,"profile":profile,"report":report}[args.stage]()
