"""Independent outcome reconciliation, complete-model costs and full reporting."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import component_models as gm
import campaign
import study as base
from final_audit import independent_metrics

HERE=gm.HERE
LABELS={name:f'M{c["M"]} k{c["k"]}: '+("query-free" if not c["query"] else
        f'learned, penalty {c["penalty"]:g}' if c["routing"]=="learned" else c["routing"]+" routing")
        for name,c in gm.CONFIGS.items()}
PAIRS=[(f"m8_k{k}_{p}",f"m8_k{k}_learned0") for k in (1,2)
       for p in ("learned001","learned01","fixed","uniform","noquery")]
PAIRS += [(f"m{M}_k{k}_learned0",f"m8_k{k}_learned0") for M in (4,16) for k in (1,2,3)]
PAIRS += [(f"m{M}_k{k}_learned0",f"m{M}_k1_learned0") for M in (4,8,16) for k in (2,3)]


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def audit():
    lock=campaign.verify()
    selections=campaign.verify_selection()
    evaluation=read("evaluation.json")
    assert evaluation["selection_lock_sha256"]==base.sha(HERE/"selection_lock.json")
    assert datetime.fromisoformat(evaluation["evaluated_utc"])>=datetime.fromisoformat(selections["locked_utc"])
    assert len(evaluation["rows"])==95
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
        model=gm.build_model(choice["seed"],choice["head"],1).cuda()
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
        failed_candidates=sum(r["status"]!="valid" for r in records.values()),selections=95,
        valid_selected=len(reloads),max_independent_validation_metric_delta=val_delta,
        max_independent_test_metric_delta=test_delta,checkpoint_reloads=reloads,
        maximum_cpu_gpu_target_scaling_delta=scaling_delta,target_scaling_tolerance=1e-13,
        previous_neural_manifest_entries_unchanged=len(files),
        audited_analysis_source_sha256=base.sha(__file__),
        qualification="Numerical and bookkeeping audit; not statistical confirmation or a global optimization certificate")
    base.write_json(HERE/"final_audit.json",output)
    print(json.dumps({k:v for k,v in output.items() if k!="checkpoint_reloads"}),flush=True)


def measure_blocks(model,state,x,mask,adj,target,mode,lr,penalty_weight):
    wall,baselines,peaks,increments=[],[],[],[]
    for block in range(5):
        model.load_state_dict(state)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4) if mode=="train" else None
        torch.manual_seed(94000+block)
        if mode=="train":
            model.train()
            def operation():
                optimizer.zero_grad(set_to_none=True)
                pred=model(x,mask,adj)
                mse=(pred-target).square().mean()
                penalty=model.balance_loss()
                loss=mse+penalty_weight*penalty if penalty_weight else mse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,10.)
                optimizer.step()
        else:
            model.eval()
            def operation():
                with torch.inference_mode():
                    return model(x,mask,adj)
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(20):
            operation()
        torch.cuda.synchronize()
        wall.append((time.perf_counter()-start)*1000/20)
        peak=torch.cuda.max_memory_allocated()
        baselines.append(baseline/2**20)
        peaks.append(peak/2**20)
        increments.append((peak-baseline)/2**20)
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        del optimizer
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    model.eval()
    return dict(block_average_ms=wall,median_block_average_ms=statistics.median(wall),
        minimum_block_average_ms=min(wall),maximum_block_average_ms=max(wall),
        baseline_allocated_mib=baselines,peak_allocated_mib=peaks,incremental_peak_mib=increments)


def profile():
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
        model=gm.build_model(42,choice["head"],1).cuda().eval()
        model.load_state_dict(state)
        with torch.no_grad():
            reference=model(x,mask,adj).clone()
        target=((blob["y"][:64].double()-checkpoint["target_mean"])/checkpoint["target_sd"]).float().cuda()
        timings=[]
        for batch in (1,64):
            for mode in ("eval","train"):
                result=measure_blocks(model,state,x[:batch],mask[:batch],adj[:batch],target[:batch],mode,candidate["spec"]["lr"],gm.CONFIGS[choice["head"]]["penalty"])
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
        analysis_source_sha256=base.sha(__file__),
        hardware=torch.cuda.get_device_name(),dtype="float32",tf32=False,cpu_threads=4,
        workload_ids=blob["ids"][:64],valid_atom_counts=blob["mask"][:64].sum(1).tolist(),padded_atoms=n,
        blocks=5,repetitions_per_block=20,warmups_per_block=10,
        boundary="Preloaded GPU inputs, full model including frozen route encoder; train includes zero_grad, MSE, raw balance computation, weighted penalty when nonzero, backward, clipping and AdamW. Transfers and target de-standardization excluded.",
        interpretation="Median of block-average times; not per-request latency quantiles."))
    print(json.dumps(dict(stage="component_profiles_complete",procedures=len(rows),workloads=sum(len(r.get("timings",[])) for r in rows))),flush=True)


def describe(values):
    a=[float(v) for v in values]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,
                sample_sd=statistics.stdev(a) if len(a)>1 else None,
                minimum=min(a) if a else None,maximum=max(a) if a else None)


def diagnostics():
    selections=campaign.verify_selection()
    assert read("final_audit.json")["passed"]
    blobs={s:base.load_split(s) for s in ("train","val","test")}
    directory=HERE/"routing_diagnostics"
    directory.mkdir(exist_ok=True)
    output=[]
    for choice in selections["selections"]:
        if choice["selected_id"] is None:
            output.append(dict(**choice,status="no_selected_checkpoint"))
            continue
        candidate=read("candidates/"+choice["selected_id"]+".json")
        checkpoint=torch.load(HERE/candidate["checkpoint"],weights_only=True,map_location="cpu")
        model=gm.build_model(choice["seed"],choice["head"],1).cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        M=gm.CONFIGS[choice["head"]]["M"]
        summaries=[]
        for split,blob in blobs.items():
            pieces=[]
            mass=torch.zeros(M,device="cuda",dtype=torch.float64)
            entropy=torch.zeros((),device="cuda",dtype=torch.float64)
            atoms=0
            with torch.no_grad():
                for start in range(0,len(blob["y"]),64):
                    idx=torch.arange(start,min(start+64,len(blob["y"])),device="cuda")
                    x,mask,adj=base.get_batch(blob,idx)
                    model(x,mask,adj)
                    pi=model.head.layers[0].last_routing.double()
                    n=mask.sum(1)
                    assert bool(torch.isfinite(pi).all())
                    assert float((pi.sum(-1)-1).abs().max())<2e-6
                    slot=(pi*mask[:,:,None]).sum(1)
                    mass += slot.sum(0)
                    count=int(n.sum())
                    atoms += count
                    ent=-(torch.special.xlogy(pi,pi)*mask[:,:,None]).sum((1,2))
                    entropy += ent.sum()
                    graph_mass=slot/n[:,None]
                    graph_penalty=M*((graph_mass-1/M)**2).sum(-1)
                    square_terms=(pi.square()*mask[:,:,None]).sum((1,2))
                    soft_numerator=slot.square().sum(-1)-square_terms
                    hard_counts=(torch.nn.functional.one_hot(pi.argmax(-1),M)*mask[:,:,None]).sum(1)
                    hard_numerator=(hard_counts*(hard_counts-1)).sum(-1)
                    eligible=n>=2
                    denominator=(n*(n-1)).double()
                    soft=torch.full_like(denominator,float("nan"))
                    hard=torch.full_like(denominator,float("nan"))
                    soft[eligible]=soft_numerator[eligible]/denominator[eligible]
                    hard[eligible]=hard_numerator[eligible]/denominator[eligible]
                    assert bool((soft[eligible]>=-1e-12).all()) and bool((soft[eligible]<=1+2e-6).all())
                    assert bool((hard[eligible]>=0).all()) and bool((hard[eligible]<=1).all())
                    pieces.append(torch.stack([n.double(),graph_penalty,ent/n,soft,hard],1).cpu().numpy())
            array=np.concatenate(pieces)
            eligible=array[:,0]>=2
            soft_values=array[eligible,3]
            hard_values=array[eligible,4]
            global_mass=(mass/atoms).cpu().tolist()
            raw_entropy=float(entropy)/atoms
            path=directory/f'{choice["selected_id"]}_{split}.npz'
            np.savez_compressed(path,ids=np.asarray(blob["ids"]),atom_counts=array[:,0].astype(np.int64),
                graph_load_penalty=array[:,1],graph_mean_entropy_nats=array[:,2],
                soft_distinct_atom_coassignment=array[:,3],hard_argmax_distinct_atom_coassignment=array[:,4],
                pair_statistic_eligible=eligible)
            summaries.append(dict(split=split,graphs=len(array),valid_atoms=atoms,eligible_pair_graphs=int(eligible.sum()),
                atom_weighted_bucket_mass=global_mass,
                penalty_at_global_atom_weighted_load=float(M*((mass/atoms-1/M)**2).sum()),
                mean_graph_load_penalty=float(array[:,1].mean()),
                atom_weighted_entropy_nats=raw_entropy,atom_weighted_entropy_over_log_M=raw_entropy/np.log(M),
                graph_weighted_soft_coassignment_mean=float(soft_values.mean()) if len(soft_values) else None,
                uniform_soft_reference=1/M,
                graph_weighted_soft_coassignment_minus_uniform=float(soft_values.mean()-1/M) if len(soft_values) else None,
                graph_weighted_hard_argmax_coassignment_mean=float(hard_values.mean()) if len(hard_values) else None,
                per_graph_file=str(path.relative_to(HERE)),per_graph_sha256=base.sha(path)))
        output.append(dict(**choice,status="computed",configuration=gm.CONFIGS[choice["head"]],splits=summaries))
        del model
        base.write_json(HERE/"routing_diagnostics.json",dict(complete=False,rows=output))
    base.write_json(HERE/"routing_diagnostics.json",dict(complete=True,computed_utc=base.now(),rows=output,
        analysis_source_sha256=base.sha(__file__),selection_lock_sha256=base.sha(HERE/"selection_lock.json"),
        weighting="Global mass and entropy weight valid atoms; within-graph penalties/coassignment means weight eligible graphs equally",
        qualification="Assignment concentration, not unknown active-interaction collision probability or molecular Bayes risk; uniform hard argmax degenerates under first-index ties"))
    print(json.dumps(dict(stage="routing_diagnostics_complete",selected=len(output))),flush=True)


def write_csv(name,rows):
    with (HERE/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report():
    assert read("final_audit.json")["passed"] and read("profiles.json")["complete"]
    assert read("routing_diagnostics.json")["complete"]
    evaluation=read("evaluation.json")
    allrows=evaluation["rows"]
    rows=[r for r in allrows if r["status"]=="valid"]
    groups=[]
    for head in gm.HEADS:
        subset=[r for r in rows if r["head"]==head]
        groups.append(dict(head=head,label=LABELS[head],configuration=gm.CONFIGS[head],planned_seeds=5,
            rmse=describe([r["test"]["rmse"] for r in subset]),
            mae=describe([r["test"]["mae"] for r in subset]),
            pearson=describe([r["test"]["pearson"] for r in subset if r["test"]["pearson"] is not None]),
            total_parameters=subset[0]["total_parameters"] if subset else None,
            trainable_parameters=subset[0]["trainable_parameters"] if subset else None,
            frozen_parameters=subset[0]["frozen_parameters"] if subset else None))
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
        gate2=r["selected_gate_coefficients"][1] if len(r["selected_gate_coefficients"])>1 else None,
        gate3=r["selected_gate_coefficients"][2] if len(r["selected_gate_coefficients"])>2 else None) for r in rows]
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
        "# Routing, balancing, query and bucket-width experiment", "",
        f'{output["valid_candidates"]}/{len(candidates)} candidates were numerically valid; {len(rows)}/95 selected predictors were scored. All 190 candidate records and 95 validation-based choices were locked before this campaign\'s test scoring. Total candidate fitting time was {output["aggregate_fit_seconds"]/60:.2f} minutes.', "",
        "| Procedure | Test RMSE, mean ± sample SD | Parameters | Valid seeds |", "|---|---:|---:|---:|",*table,"",
        "| Prespecified paired contrast | Mean RMSE difference | Left procedure lower |", "|---|---:|---:|",*contrasts,"",
        "Negative differences favor the left procedure. CSV files retain every selected seed, gate coefficient and prespecified paired difference. Candidate histories retain both learning rates and any failures. No favorable width, order or penalty is selected across procedure families.","",
        "The comparison uses one-layer GCN and the same ligand-only 7,384/958/2,171 partition. A seed-specific untrained M16 k3 template supplies retained tensors, first-k gates and first-M routing rows. This differs from earlier initialization protocols; means are not pooled across campaigns. Prefix pairing does not preserve initial functions across widths or orders. Both routing normalization and memory competition change, alongside parameter count and compute.","",
        "Routing and balancing comparisons cross M8 with k1/k2. The fixed control computes assignments from a separate frozen untrained GCN and routing maps, while prediction representations continue to learn. This is a fixed random-feature branch, not merely a frozen routing matrix on changing features. Uniform routing removes unused routing parameters. Query-free readout averages all concatenated memory tokens, weighting orders by multiplicity; it preserves residual information paths. Trainable/frozen counts appear in the machine-readable table and every selected record.","",
        "The pre-fit audit checks original-equation equivalence, paired retained tensors, initial assignment equality, fixed-assignment preservation after prediction updates, gradient paths, masking and permutation/batch invariance. The uniform routing penalty and query-free mean are checked explicitly. These controls do not supply a novel theorem or a molecular Bayes-risk estimate.","",
        "| Complete model, batch 64 | Evaluation ms | AdamW step ms |", "|---|---:|---:|",*costs,"",
        "Costs are medians of five block-average times on the common first-64-training-graph workload, with 10 warmups and 20 measured operations per block. Inputs reside on the RTX 3060 Laptop GPU; float32 operations disable TF32. A training operation includes the prediction loss and declared balancing penalty, backward, global clipping and AdamW. The frozen routing encoder is included. Profiles also retain batch-one workloads, allocated-memory baselines and peaks, and verify restoration of selected checkpoints.","",
        "Independent calculations recheck every valid candidate's validation metric and all selected test metrics. Both validation and test predictions are regenerated from every selected checkpoint. The audit also verifies that the prior neural archive's manifest entries remain unchanged.","",
        "Routing diagnostics separately report global load, graph-specific load, normalized entropy and distinct-atom hard/soft coassignment. Mean minibatch penalties differ from penalties at pooled split loads. Uniform hard-argmax routing is degenerate because of tie handling; the soft uniform reference is 1/M. No such statistic measures unknown target-interaction collisions or molecular irreducible error. The separate exact finite-domain diagnostic illustrates why balanced load and input-collision rates alone do not identify retained target information.","",
        "These are development-informed optimizer/selection repetitions on a reused split. The 150 validation–test shared canonical ligands, 241 annotation-qualified training labels and absence of protein/pocket features remain limitations. This block addresses component and M-sensitivity requests within the reconstructed setting; it does not complete ten-seed synthetic reliability, deeper hierarchical tasks, GAT or the unavailable historical protocol.",""
    ])
    (HERE/"report.md").write_text(text,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,8))
    for i,g in enumerate(groups):
        data=[r["test"]["rmse"] for r in rows if r["head"]==g["head"]]
        if data:
            ax.scatter(data,np.full(len(data),i),color="#64748b",s=22,alpha=.7)
            if len(data)>1:
                ax.errorbar(g["rmse"]["mean"],i,xerr=g["rmse"]["sample_sd"],fmt="o",color="#155e75",capsize=3)
    ax.set_yticks(range(len(groups)),[g["label"] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (lower is better); points = seeds, bars = sample SD")
    ax.set_title("Component/width comparison on the reused ligand-only split")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/"all_procedures.png",dpi=180)
    fig.savefig(HERE/"all_procedures.svg")
    plt.close(fig)
    print(json.dumps(output),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","profile","diagnostics","report"])
    args=parser.parse_args()
    base.configure()
    {"audit":audit,"profile":profile,"diagnostics":diagnostics,"report":report}[args.stage]()

