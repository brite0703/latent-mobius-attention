"""Audit, profile and report every supplemental normalization/count control."""
import csv
import json
import math
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import controls as c
from final_audit import independent_metrics
from profile_models import measure_blocks
from summarize import describe, csv_write

HERE, PRIMARY, base = c.HERE, c.PRIMARY, c.base
LABELS = dict(deepsets_raw35="Plain sum Deep Sets, w=35", deepsets_raw70="Plain sum Deep Sets, w=70",
              deepsets_count35="LN-sum Deep Sets + count, w=35", deepsets_count70="LN-sum Deep Sets + count, w=70",
              pma_count="PMA + additive log-count")


def main():
    base.configure()
    lock = c.verify()
    choices = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    evaluation = json.loads((HERE/"evaluation.json").read_text(encoding="utf-8"))
    primary_eval = json.loads((PRIMARY/"evaluation.json").read_text(encoding="utf-8"))
    combined = json.loads((PRIMARY/"combined_selection_gate.json").read_text(encoding="utf-8"))
    assert combined["primary_selection_sha256"] == base.sha(PRIMARY/"selection_lock.json")
    assert combined["supplement_selection_sha256"] == base.sha(HERE/"selection_lock.json")
    assert datetime.fromisoformat(combined["locked_utc"]) < datetime.fromisoformat(evaluation["evaluated_utc"])
    assert datetime.fromisoformat(combined["locked_utc"]) < datetime.fromisoformat(primary_eval["evaluated_utc"])
    assert len(evaluation["rows"]) == len(choices["selections"]) == 25
    train = torch.load(base.DATA/"pdbbind_train.pt",weights_only=True,map_location="cpu")
    val = torch.load(base.DATA/"pdbbind_val.pt",weights_only=True,map_location="cpu")
    mean = float(train["y"].double().mean())
    sd = float(train["y"].double().std(unbiased=False))
    assert evaluation["selection_lock_sha256"] == base.sha(HERE/"selection_lock.json")
    assert evaluation["combined_gate_sha256"] == base.sha(PRIMARY/"combined_selection_gate.json")
    test = base.load_split("test")
    y = test["y"].double().cpu().numpy()
    records = {}
    validation_delta = 0.
    for item in choices["candidate_records"]:
        assert base.sha(HERE/item["path"]) == item["sha256"]
        row = json.loads((HERE/item["path"]).read_text(encoding="utf-8"))
        records[row["spec"]["id"]] = row
        assert abs(row["target_mean"]-mean) < 1e-13 and abs(row["target_sd"]-sd) < 1e-13
        assert row["implementation_lock_sha256"] == base.sha(HERE/"implementation_lock.json")
        assert datetime.fromisoformat(row["started_utc"]) >= datetime.fromisoformat(lock["locked_utc"])
        assert datetime.fromisoformat(row["finished_utc"]) <= datetime.fromisoformat(choices["locked_utc"])
        if row["status"] != "valid":
            continue
        assert base.sha(HERE/row["checkpoint"]) == row["checkpoint_sha256"]
        assert base.sha(HERE/row["validation_predictions"]) == row["validation_prediction_sha256"]
        archive = np.load(HERE/row["validation_predictions"])
        assert np.array_equal(archive["ids"], np.asarray(val["ids"]))
        assert np.array_equal(archive["truth"], val["y"].double().numpy())
        metric = independent_metrics(archive["truth"],archive["prediction"])["rmse"]
        validation_delta = max(validation_delta,abs(metric-row["best_validation_rmse"]))
        best = min((h for h in row["history"] if "validation_rmse" in h),key=lambda h:(h["validation_rmse"],h["epoch"]))
        assert best["epoch"] == row["best_epoch"] and best["validation_rmse"] == row["best_validation_rmse"]
        assert row["epochs_run"] == len(row["history"])
        assert [h["epoch"] for h in row["history"]] == list(range(1,row["epochs_run"]+1))
    assert len(records) == 50
    reload_rows, test_delta, timings, profile_specs = [], 0., [], []
    for choice, row in zip(choices["selections"],evaluation["rows"]):
        valid = [records[id_] for id_ in choice["candidate_ids"] if records[id_]["status"]=="valid"]
        selected = min(valid,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if valid else None
        assert choice["selected_id"] == (selected["spec"]["id"] if selected else None)
        assert row["selected_id"] == choice["selected_id"]
        if selected is None:
            continue
        path = HERE/row["prediction_file"]
        assert base.sha(path) == row["prediction_sha256"]
        archive = np.load(path)
        assert np.array_equal(archive["ids"],np.asarray(test["ids"])) and np.array_equal(archive["truth"],y)
        metric = independent_metrics(y,archive["prediction"])
        for key in metric:
            if metric[key] is not None:
                test_delta = max(test_delta,abs(metric[key]-row["test"][key]))
            else:
                assert row["test"][key] is None
        checkpoint = torch.load(HERE/selected["checkpoint"],weights_only=True,map_location="cpu")
        model = c.build(choice["seed"],choice["head"],1).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        p = base.predict(model,test,checkpoint["target_mean"],checkpoint["target_sd"])
        error = float(np.max(np.abs(p-archive["prediction"])))
        assert error < 1e-10
        reload_rows.append(dict(id=choice["selected_id"],maximum_prediction_delta=error))
        if choice["seed"]==42:
            profile_specs.append((choice, selected, checkpoint))
        del model
    assert validation_delta < 1e-12 and test_delta < 1e-12
    # Release the audit's test tensors before cost measurements so the GPU
    # allocation baseline contains the same workload as the core profiler.
    del test
    torch.cuda.empty_cache()
    n = int(train["mask"][:64].sum(1).max())
    workload = {k:train[k][:64].cuda() for k in ("X","mask","adj","y")}
    workload["X"],workload["mask"],workload["adj"] = workload["X"][:,:n],workload["mask"][:,:n],workload["adj"][:,:n,:n]
    for choice, selected, checkpoint in profile_specs:
        model = c.build(42,choice["head"],1).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            reference = model(workload["X"],workload["mask"],workload["adj"]).clone()
        target = ((workload["y"].double()-checkpoint["target_mean"])/checkpoint["target_sd"]).float()
        profile = []
        for batch in (1,64):
            for mode in ("eval","train"):
                profile.append(dict(batch_size=batch,padded_n=n,**measure_blocks(model,checkpoint["state_dict"],
                    workload["X"][:batch],workload["mask"][:batch],workload["adj"][:batch],target[:batch],mode,selected["spec"]["lr"])))
        with torch.no_grad():
            restored = model(workload["X"],workload["mask"],workload["adj"])
        assert float((restored-reference).abs().max()) == 0.
        timings.append(dict(head=choice["head"],depth=1,seed=42,timings=profile))
        del model
        torch.cuda.empty_cache()
    audit = dict(passed=True,audited_utc=base.now(),candidates=50,valid_candidates=sum(r["status"]=="valid" for r in records.values()),
                 failed_candidates=sum(r["status"]!="valid" for r in records.values()),selected=25,valid_selected=len(reload_rows),
                 independent_validation_metric_max_delta=validation_delta,independent_test_metric_max_delta=test_delta,
                 checkpoint_reload=reload_rows,combined_gate_verified=True,source_sha256=base.sha(__file__))
    base.write_json(HERE/"final_audit.json",audit)
    base.write_json(HERE/"profiles.json",dict(profiled_utc=base.now(),complete=True,rows=timings,
         workload_ids=train["ids"][:64],padded_n=n,measurement_boundary="Same five blocks/twenty calls/ten warmups and complete-model boundary as the core protocol."))
    summaries = []
    for name in c.HEADS:
        rows = [r for r in evaluation["rows"] if r["head"]==name and r["status"]=="valid"]
        summaries.append(dict(head=name,parameters=rows[0]["total_parameters"] if rows else None,
                             rmse=describe([r["test"]["rmse"] for r in rows]),
                             mae=describe([r["test"]["mae"] for r in rows])))
    all_rows = {(r["head"],r["seed"]):r for r in primary_eval["rows"] if r["depth"]==1 and r["status"]=="valid"}
    all_rows.update({(r["head"],r["seed"]):r for r in evaluation["rows"] if r["status"]=="valid"})
    pairs = [("deepsets_raw35","deepsets"),("deepsets_raw70","deepsets_wide"),
             ("deepsets_count35","deepsets"),("deepsets_count70","deepsets_wide"),("pma_count","pma"),
             ("deepsets_raw70","deepsets_raw35")]
    # Comparisons with proposed k2 are displayed for every supplemental baseline.
    pairs += [(name,"lma2") for name in c.HEADS]
    contrasts = []
    for left,right in pairs:
        values = [all_rows[left,s]["test"]["rmse"]-all_rows[right,s]["test"]["rmse"]
                  for s in base.SEEDS if (left,s) in all_rows and (right,s) in all_rows]
        contrasts.append(dict(left=left,right=right,expected_pairs=5,left_better=sum(v<0 for v in values),**describe(values)))
    result = dict(generated_utc=base.now(),summaries=summaries,contrasts=contrasts,
        aggregate_fit_seconds=sum(r["wall_seconds"] for r in records.values()),
        candidates_reaching_epoch_cap=sum(r["reached_epoch_cap"] for r in records.values()),
        selected_reaching_epoch_cap=sum(r.get("selected_epochs")==150 for r in evaluation["rows"]),
        valid_candidates=audit["valid_candidates"],valid_selected=audit["valid_selected"])
    base.write_json(HERE/"results_summary.json",result)
    selected_rows = [dict(head=r["head"],seed=r["seed"],status=r["status"],selected_id=r["selected_id"],
                         validation_rmse=r["validation_rmse"],test_rmse=r["test"]["rmse"] if r["test"] else None,
                         test_mae=r["test"]["mae"] if r["test"] else None,parameters=r.get("total_parameters"),
                         selected_lr=r.get("selected_lr"),best_epoch=r.get("best_epoch"),epochs_run=r.get("selected_epochs"))
                     for r in evaluation["rows"]]
    csv_write(HERE/"selected_predictors.csv",selected_rows)
    csv_write(HERE/"all_candidates.csv",[dict(**r["spec"],status=r["status"],validation_rmse=r["best_validation_rmse"],
        best_epoch=r["best_epoch"],epochs_run=r["epochs_run"],wall_seconds=r["wall_seconds"],
        peak_allocated_mib=r["peak_allocated_mib"],error=r.get("error","")) for r in records.values()])
    csv_write(HERE/"paired_differences.csv",[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in contrasts])
    text = ["# Baseline normalization and cardinality results","",
        "All five controls were locked before their fits and before either block's new test evaluation. The 170-candidate core lock remains unchanged. These are exploratory comparisons on the same reused ligand-only split.","",
        "| Supplemental head | Valid/expected | Parameters | Test RMSE, mean ± SD |",
        "|---|---:|---:|---:|"]
    for s in summaries:
        statistic = f'{s["rmse"]["mean"]:.5f} ± {s["rmse"]["sd"]:.5f}' if s["rmse"]["n"]>1 else str(s["rmse"]["mean"])
        text.append(f'| {LABELS[s["head"]]} | {s["rmse"]["n"]}/5 | {s["parameters"]} | {statistic} |')
    text += ["","| Paired difference (left minus right) | Pairs | Mean ΔRMSE | Left lower |",
             "|---|---:|---:|---:|"]
    for r in contrasts:
        if r["n"]:
            text.append(f'| {r["left"]} − {r["right"]} | {r["n"]}/5 | {r["mean"]:+.5f} | {r["left_better"]}/{r["n"]} |')
    text += ["",f'Aggregate supplemental candidate fitting time: {result["aggregate_fit_seconds"]/60:.2f} minutes. Valid candidates {audit["valid_candidates"]}/50; valid selected outcomes {audit["valid_selected"]}/25. Candidates reaching 150 epochs: {result["candidates_reaching_epoch_cap"]}/50; selected candidates reaching that cap: {result["selected_reaching_epoch_cap"]}/25.',"",
        "Removing LayerNorm changes both the representation and its optimization scales. Adding count does not restore all unnormalized sum information. PMA uses an additive log-count term; Deep Sets can use count nonlinearly. These results do not identify a single cause of an error difference or validate a theorem. All original and supplemental baselines must remain visible.","",
        "| Head | Seed | LR | Selected epoch / epochs run | Validation RMSE | Test RMSE |","|---|---:|---:|---:|---:|---:|"]
    for r in selected_rows:
        if r["status"]=="valid":
            text.append(f'| {r["head"]} | {r["seed"]} | {r["selected_lr"]} | {r["best_epoch"]} / {r["epochs_run"]} | {r["validation_rmse"]:.6f} | {r["test_rmse"]:.6f} |')
        else:
            text.append(f'| {r["head"]} | {r["seed"]} | — | — | missing | missing |')
    (HERE/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    fig,ax = plt.subplots(figsize=(10,5),layout="constrained")
    for j,s in enumerate(summaries):
        v = s["rmse"]
        ax.scatter(v["values"],j+np.linspace(-.10,.10,v["n"]),color="#176786",s=28)
        if v["n"]>1:
            ax.errorbar(v["mean"],j,xerr=v["sd"],fmt="D",color="#182a35",capsize=3)
    ax.set_yticks(range(5),[LABELS[s["head"]] for s in summaries])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (pK units; lower is better)")
    ax.set_title("Supplemental GCN1 baselines\nFive seeds; diamonds and bars show mean ± one SD")
    ax.grid(axis="x",alpha=.2)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(HERE/"all_controls.png",dpi=180,bbox_inches="tight")
    fig.savefig(HERE/"all_controls.svg",bbox_inches="tight")
    print(json.dumps({k:v for k,v in result.items() if k not in ("summaries","contrasts")}),flush=True)


if __name__=="__main__":
    main()
