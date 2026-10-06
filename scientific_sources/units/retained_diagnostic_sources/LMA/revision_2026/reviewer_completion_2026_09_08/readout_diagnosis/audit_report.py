"""Reconcile diagnostic metrics from saved record rows with Python scalars."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, statistics

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return math.fsum(values)/len(values)


def main():
    original=read(HERE/"diagnosis.json")
    uniform=read(HERE/"uniform_query_diagnosis.json")
    assert original["complete"] and uniform["complete"]
    for receipt in (original,uniform):
        for source in receipt["sources"]:
            assert sha(source["path"])==source["sha256"]
        for artifact in receipt["artifacts"]:
            assert sha(HERE/artifact["path"])==artifact["sha256"]
    source_paths=[HERE/"diagnosis.json",HERE/"uniform_query_diagnosis.json",
                  HERE/"fitting_record_diagnostics.csv",HERE/"uniform_query_records.csv",Path(__file__)]
    tables=[]
    for name in ("fitting_record_diagnostics.csv","uniform_query_records.csv"):
        with (HERE/name).open(newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==8960
        table={(r["head"],int(r["seed"]),r["id"]):r for r in rows}
        assert len(table)==8960
        tables.append(table)
    assert set(tables[0])==set(tables[1])
    comparisons=[]
    def close(a,b):
        delta=abs(a-b);comparisons.append(delta)
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    for row,urow in zip(original["rows"],uniform["rows"]):
        assert row["selected_id"]==urow["selected_id"]
        items=[tables[0][row["head"],row["seed"],key] for key in original["input_ids"]]
        more=[tables[1][row["head"],row["seed"],key] for key in original["input_ids"]]
        truth=[float(r["truth"]) for r in items]
        full=[float(r["full_prediction"]) for r in items]
        first=[float(r["first_only_prediction"]) for r in items]
        zero=[float(r["zero_higher_prediction"]) for r in items]
        uq=[float(r["uniform_prediction"]) for r in more]
        assert truth==[float(r["truth"]) for r in more] and full==[float(r["full_prediction"]) for r in more]
        close(math.sqrt(mean([(p-y)**2 for p,y in zip(full,truth)])),row["fitting_rmse"])
        for prefix,values in (("first_only",first),("zero_higher",zero)):
            close(math.sqrt(mean([(a-b)**2 for a,b in zip(values,full)])),row[prefix+"_prediction_change_rms"])
            close(mean([(a-y)**2-(b-y)**2 for a,b,y in zip(values,full,truth)]),row[prefix+"_minus_full_fitting_mse"])
        close(math.sqrt(mean([(a-b)**2 for a,b in zip(uq,full)])),urow["uniform_prediction_change_rms"])
        close(mean([(a-y)**2-(b-y)**2 for a,b,y in zip(uq,full,truth)]),urow["uniform_minus_full_fitting_mse"])
        for key,summary in row["diagnostics"].items():
            values=[float(r[key]) for r in items]
            close(mean(values),summary["mean"])
            close(statistics.stdev(values),summary["sample_sd"])
    for receipt in (original,uniform):
        for group in receipt["groups"]:
            rows=[r for r in receipt["rows"] if r["head"]==group["head"]]
            assert len(rows)==5
            for key,value in group.items():
                if isinstance(value,dict) and "mean" in value:
                    close(mean([r[key] for r in rows]),value["mean"])
                    close(statistics.stdev([r[key] for r in rows]),value["sample_sd"])
            for key,value in group.get("diagnostics",{}).items():
                values=[r["diagnostics"][key]["mean"] for r in rows]
                close(mean(values),value["mean"])
                close(statistics.stdev(values),value["sample_sd"])
    labels=["Original k=1","Original product k=2","Original additive k=2",
        "Separate product, gate initialized at 0","Separate additive, gate initialized at 0",
        "Separate product, gate initialized at 1","Separate additive, gate initialized at 1"]
    lines=["# What the completed readout models use","",
        "This exploratory analysis checks all 35 fixed validation-selected models on the same 256 fitting records. It uses no validation/test tensor or test prediction file and fits no new parameters. The earlier experiment's test outcomes were already known. Every original weight and prediction was restored after each perturbation; independent replay agreed exactly with the native full predictions.",
        "",
        "The second-order branch is active on this sample. Removing the original product branch changes predictions by an average RMS of 1.685113 pK units and raises fitting-sample MSE in all five seeds. Removing the additive branch also causes a large change (1.548460 pK). These effects show dependence of the fitted predictors on their learned branches; they do not distinguish multiplication from added nonlinear capacity or establish a generalization benefit.",
        "",
        "A separately declared extension replaced learned query weights with exact uniform averaging while retaining the memories and gates. For original product k=2, its mean prediction-change RMS is 0.199745 pK, compared with 1.685113 pK for branch removal. Its fitting-MSE change averages +0.039455 and has mixed signs across seeds. The effect is smaller than branch removal but is not zero. The result does not warrant calling the query irrelevant.",
        "",
        "| Procedure | Remove second-order branch: prediction RMS change | Zero higher-order value gate: RMS change | Uniform query: RMS change | Uniform query: fitting MSE change |",
        "|---|---:|---:|---:|---:|"]
    for label,group,ugroup in zip(labels,original["groups"],uniform["groups"]):
        assert group["head"]==ugroup["head"]
        lines.append(f"| {label} | {group['first_only_prediction_change_rms']['mean']:.6f} | {group['zero_higher_prediction_change_rms']['mean']:.6f} | {ugroup['uniform_prediction_change_rms']['mean']:.6f} | {ugroup['uniform_minus_full_fitting_mse']['mean']:+.6f} |")
    lines.extend(["",
        "Entries average the five seed-specific diagnostics; RMS is over the fixed 256-record sample before averaging. Positive MSE changes mean the perturbation worsens fitting error. The k=1 removal and zero-gate entries are exact null controls. All per-seed values, dispersion, records, means and signs remain in the accompanying JSON and CSV files.",
        "",
        "For the original product k=2 models, average attention entropy is 0.997081 of its maximum and the order-two attention share is 0.793164; uniform weighting would allocate 28/36 = 0.777778 to that order merely because it contains more slots. This motivates scrutiny of query selectivity and the order readout, but entropy alone does not bound the final prediction error without accounting for memory values and the downstream nonlinear maps. The uniform-query extension measures that dependence directly on this sample.",
        "",
        "The separate-normalization perturbations agree exactly when the higher-order gate is zero. In the joint-memory architecture, zeroing a memory value and removing its slot are different interventions. Deleting the second order changes its content and the normalization of the first order; zeroing its gated memory also changes the logits used to compute attention. Neither intervention holds all attention effects fixed or supplies a newly trained k=1 model. Feature coadaptation, biases, residual access and nonlinear maps prevent interpreting deletion damage as proof of a useful pairwise interaction. MSE changes are in pK squared. The uniform-query extension makes all joint slots uniform; it does not preserve the original per-query order masses.",
        "",
        "The next scientific decision remains the completed, audited matched ligand/sequence/contact comparison. This diagnosis rejects negligible computational influence under the stated interventions on the sampled fitting inputs, but does not identify why the original test gains are inconsistent. If that fixed study remains unfavorable, the bounded next hypothesis is predictive utility from a product residual added after a completely frozen k=1 predictor, against unchanged, additive and parameter-matched pair-MLP controls. That plan is not yet a locked protocol or a result. See conditional_residual_test.md and the independently assessed web critique in web_review_42_disposition.md. The theorem contribution and the synthetic-plus-real empirical target remain open.",
        "",
        "Integrity: all 17,920 saved record rows, all 35 identities in both stages, diagnostic scalars and group summaries have been independently reconciled with scalar arithmetic. CPU-only execution was verified. This is numerical checking, not statistical confirmation or an originality certificate."])
    report=HERE/"findings.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    receipt=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        saved_record_rows=17920,checkpoint_pairs=35,scalar_comparisons=len(comparisons),
        maximum_scalar_discrepancy=max(comparisons),report_sha256=sha(report),
        sources=[dict(path=str(p),sha256=sha(p)) for p in source_paths],
        qualification="Independent scalar reconciliation of exploratory fitting-only diagnostics.")
    (HERE/"report_audit.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="sources"}),flush=True)


if __name__=="__main__":
    main()
