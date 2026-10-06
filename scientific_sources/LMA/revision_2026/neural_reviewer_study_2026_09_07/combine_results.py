"""Assemble both independently locked blocks without cross-family selection."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study
from summarize import LABELS, csv_write

HERE = study.HERE
SUPP = HERE/"cardinality_controls"
LABELS = dict(LABELS, deepsets="LN-sum Deep Sets, w=35", deepsets_wide="LN-sum Deep Sets, w=70",
              deepsets_raw35="Plain sum Deep Sets, w=35", deepsets_raw70="Plain sum Deep Sets, w=70",
              deepsets_count35="LN-sum Deep Sets + count, w=35",
              deepsets_count70="LN-sum Deep Sets + count, w=70", pma_count="PMA + log-count")
ORDER1 = ["mean_count","pma","pma_count","deepsets","deepsets_wide",
          "deepsets_raw35","deepsets_raw70","deepsets_count35","deepsets_count70",
          "janossy2","dcnv2","cp_pool","lma1","lma2","lma3","additive2"]
ORDER3 = ["mean_count","deepsets_wide","cp_pool","lma1","lma2","additive2"]


def main():
    core = json.loads((HERE/"results_summary.json").read_text(encoding="utf-8"))
    supp = json.loads((SUPP/"results_summary.json").read_text(encoding="utf-8"))
    core_audit = json.loads((HERE/"final_audit.json").read_text(encoding="utf-8"))
    supp_audit = json.loads((SUPP/"final_audit.json").read_text(encoding="utf-8"))
    assert core_audit["passed"] and supp_audit["passed"]
    by_key = {(r["depth"],r["head"]):r for r in core["summaries"]}
    by_key.update({(1,r["head"]):dict(r,depth=1,observed_seeds=r["rmse"]["n"],expected_seeds=5) for r in supp["summaries"]})
    rows = [by_key[depth,name] for depth,names in [(1,ORDER1),(3,ORDER3)] for name in names]
    summary = dict(generated_utc=study.now(),candidates=220,selections=110,configurations=22,
        valid_candidates=core_audit["valid_candidates"]+supp_audit["valid_candidates"],
        valid_selected=core_audit["valid_selected"]+supp_audit["valid_selected"],
        failed_candidates=core_audit["failed_candidates"]+supp_audit["failed_candidates"],
        aggregate_fitting_seconds=core["aggregate_fitting_seconds"]+supp["aggregate_fit_seconds"],
        candidates_reaching_epoch_cap=core["candidates_reaching_epoch_cap"]+supp["candidates_reaching_epoch_cap"],
        selected_reaching_epoch_cap=core["selected_reaching_epoch_cap"]+supp["selected_reaching_epoch_cap"],
        gate_sha256=study.sha(HERE/"combined_selection_gate.json"),rows=rows,
        core_contrasts=core["contrasts"],supplement_contrasts=supp["contrasts"],
        scope="Two development-informed blocks, jointly fixed before new scoring on an already reused ligand-only split.")
    study.write_json(HERE/"combined_results_summary.json",summary)
    flat = [dict(depth=r["depth"],head=r["head"],label=LABELS[r["head"]],parameters=r["parameters"],
            valid_seeds=r["rmse"]["n"],expected_seeds=5,mean_rmse=r["rmse"]["mean"],sd_rmse=r["rmse"]["sd"],
            minimum_rmse=r["rmse"]["minimum"],maximum_rmse=r["rmse"]["maximum"],
            mean_mae=r["mae"]["mean"]) for r in rows]
    csv_write(HERE/"tables"/"all_22_configurations.csv",flat)
    history, cp = [], []
    for directory in (HERE,SUPP):
        for path in sorted((directory/"candidates").glob("*.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            maximum = max((h["maximum_preclip_gradient_norm"] for h in r["history"]),default=0.)
            above = sum(h["maximum_preclip_gradient_norm"]>10 for h in r["history"])
            history.append(dict(id=r["spec"]["id"],head=r["spec"]["head"],depth=r["spec"]["depth"],seed=r["spec"]["seed"],
                status=r["status"],epochs_run=r["epochs_run"],largest_recorded_preclip_norm=maximum,
                epochs_with_a_batch_norm_above_10=above,cap=r["reached_epoch_cap"]))
            vals = [h["cp_numerics"] for h in r["history"] if "cp_numerics" in h]
            if vals:
                cp.append(dict(id=r["spec"]["id"],depth=r["spec"]["depth"],seed=r["spec"]["seed"],
                    maximum_product=max(v["max_abs_pretanh_product"] for v in vals if v["max_abs_pretanh_product"] is not None),
                    mean_epoch_saturation_fraction=float(np.mean([v["fraction_abs_product_at_least_10"] for v in vals])),
                    maximum_epoch_zero_product_fraction=max(v["fraction_zero_products"] for v in vals),
                    maximum_factor_gradient=max(v["maximum_factor_weight_gradient_norm"] for v in vals)))
    csv_write(HERE/"tables"/"all_220_stability_records.csv",history)
    csv_write(HERE/"tables"/"cp_product_diagnostics.csv",cp)
    profiles = []
    for directory in (HERE,SUPP):
        record = json.loads((directory/"profiles.json").read_text(encoding="utf-8"))
        for r in record["rows"]:
            for t in r.get("timings",[]):
                profiles.append(dict(depth=r["depth"],head=r["head"],seed=42,batch_size=t["batch_size"],padded_n=t["padded_n"],
                    mode=t["mode"],median_ms=t["median_block_average_ms"],minimum_ms=t["minimum_block_average_ms"],
                    maximum_ms=t["maximum_block_average_ms"],maximum_peak_allocated_mib=max(t["peak_allocated_mib"]),
                    maximum_incremental_peak_mib=max(t["incremental_peak_mib"])))
    csv_write(HERE/"tables"/"all_22_model_profiles.csv",profiles)
    text = ["# All neural results: core comparison and pre-test supplement","",
        "All 22 configurations and 110 prescribed selected outcomes are represented. Separate source locks preserve the 170-candidate core and 50-candidate supplement. The combined gate fixed both blocks before either new test evaluation. The pre-existing development/test reuse remains.","",
        "| GCN depth | Head | Parameters | Valid/expected seeds | Test RMSE, mean ± SD |",
        "|---:|---|---:|---:|---:|"]
    for r in rows:
        rmse = f'{r["rmse"]["mean"]:.5f} ± {r["rmse"]["sd"]:.5f}' if r["rmse"]["n"]>1 else str(r["rmse"]["mean"])
        text.append(f'| {r["depth"]} | {LABELS[r["head"]]} | {r["parameters"]} | {r["rmse"]["n"]}/5 | {rmse} |')
    text += ["",f'Valid candidates {summary["valid_candidates"]}/220; valid selected outcomes {summary["valid_selected"]}/110. Aggregate fitting time {summary["aggregate_fitting_seconds"]/60:.2f} minutes. Epoch-cap candidates: {summary["candidates_reaching_epoch_cap"]}/220; selected candidates reaching the cap: {summary["selected_reaching_epoch_cap"]}/110.',"",
        "SD and individual seed differences are descriptive, conditional on the reused split and prescribed selection procedure. They are not confidence intervals over molecular populations. Plain-sum and count variants remain distinct configurations; no favorable head is used to replace an unfavorable original baseline.","",
        "Core individual outcomes and paired contrasts: complete_tables.md. Supplemental individual outcomes and comparisons: cardinality_controls/report.md. Profiles for all 22 complete-model configurations (88 measurements): tables/all_22_model_profiles.csv. All candidate stability summaries: tables/all_220_stability_records.csv."]
    (HERE/"all_results.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    fig,axes = plt.subplots(2,1,figsize=(11.5,13.5),gridspec_kw={"height_ratios":[16,6]},layout="constrained")
    all_values = [v for r in rows for v in r["rmse"]["values"]]
    shared_limits = (min(all_values)-.01,max(all_values)+.01)
    for ax,depth,names in zip(axes,(1,3),(ORDER1,ORDER3)):
        for j,name in enumerate(names):
            r = by_key[depth,name]["rmse"]
            color = "#12698d" if name.startswith("lma") else "#b57333" if "count" in name or "raw" in name else "#587264"
            ax.scatter(r["values"],j+np.linspace(-.12,.12,r["n"]),color=color,s=23,alpha=.75)
            if r["n"]>1:
                ax.errorbar(r["mean"],j,xerr=r["sd"],fmt="D",color=color,capsize=3,markersize=4.8)
        ax.set_yticks(range(len(names)),[LABELS[name] for name in names])
        ax.invert_yaxis()
        ax.set_title(f'GCN depth {depth}')
        ax.set_xlabel("Test RMSE (pK units; lower is better)")
        ax.set_xlim(*shared_limits)
        ax.grid(axis="x",alpha=.16)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Complete ligand-only neural comparison\nPoints: five selected seeds · diamonds and bars: mean ± one SD",fontsize=15)
    fig.savefig(HERE/"figures"/"all_22_neural_results.png",dpi=180,bbox_inches="tight")
    fig.savefig(HERE/"figures"/"all_22_neural_results.svg",bbox_inches="tight")
    plt.close(fig)
    tex = [
        r"\section{Neural readout comparisons within a ligand-only reconstruction}",
        r"\label{sec:neural-reviewer-study}",
        r"The nonlinear head remains outside the scalar theorem assumptions. To address the requested architectural comparisons, we jointly trained graph encoders and readouts on the documented ligand-only LP-PDBBind reconstruction. The split contains 7,384 training, 958 validation and 2,171 test records, with 53 atom features and normalized bond adjacency. It omits protein and pocket inputs. The historical test split had already influenced development, so the new comparisons remain exploratory.",
        r"The core block fixes eleven readouts at GCN depth one and six at depth three, each with five paired seeds and two learning-rate candidates. A separately locked source-inspection supplement adds plain-sum and count-preserving Deep Sets and a PMA count correction. It preserves every core baseline. All 220 candidate records and 110 selections were fixed before either block's new test evaluation.",
        r"Every run uses AdamW, batch 256, weight decay $10^{-4}$, gradient clipping at 10, cosine decay over at most 150 epochs, and training-only target standardization. The learning rates are $3\times10^{-4}$ and $10^{-3}$. Validation RMSE selects the checkpoint and rate under the recorded stopping/tie rules. Shared encoder initialization denotes equal initial tensors within each depth and seed; all encoders subsequently train jointly with their readouts.",
        r"\href{https://arxiv.org/abs/1811.01900}{Janossy-2} uses 64 sampled ordered distinct pairs per training forward and exact pair averaging for validation and test. \href{https://arxiv.org/abs/2008.13535}{DCN-V2} uses two full-matrix cross layers on a normalized mean/count summary. \href{https://arxiv.org/abs/2205.11691}{CP global pooling} applies tanh after multiplying affine factors and includes a low-order sum branch; its product/tanh computation uses float64. These are explicitly defined readout adaptations, not reproductions of the original papers' full benchmark architectures.\par",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{All new neural configurations. Test RMSE is mean $\pm$ sample SD over five prescribed optimizer/selection seeds on one reused split. Missing depth-three entries denote unperformed configurations. P denotes total trainable parameters. The two blocks and all individual outcomes are preserved separately.}",
        r"\label{tab:all-neural-reviewer-results}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Head & P, GCN1 & RMSE, GCN1 & P, GCN3 & RMSE, GCN3\\\midrule"
    ]
    for name in ORDER1:
        a=by_key[1,name]
        b=by_key.get((3,name))
        def cell(r):
            if r is None:
                return "--"
            return "$"+f'{r["rmse"]["mean"]:.5f}\\pm{r["rmse"]["sd"]:.5f}'+"$" if r["rmse"]["n"]>1 else "missing"
        label=LABELS[name]
        tex.append(f'{label} & {a["parameters"]} & {cell(a)} & {b["parameters"] if b else "--"} & {cell(b)}'+r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}",
        r"Removing pooled LayerNorm changes activation and gradient scales as well as sensitivity to sum magnitude. The Deep Sets count input enters its nonlinear readout, whereas PMA's added coefficient gives an additive log-count correction. These controls therefore delimit the baseline interpretation without identifying a single cause of a prediction difference. The product/additive pair has identical shapes, initial tensors and memory-token counts.",
        f'All {summary["valid_candidates"]} numerically valid candidates and {summary["valid_selected"]} valid selections are retained. {summary["candidates_reaching_epoch_cap"]} candidates reached the epoch cap, including {summary["selected_reaching_epoch_cap"]} selected candidates. Numerical validity and validation selection do not establish convergence to a global optimum.',
        r"Whole-model forward and optimizer-step costs are measured on common inputs at batch sizes one and 64 using an RTX 3060 Laptop GPU. The supplement records precision, warm-ups, repeated block timings, baseline and incremental allocation, all learning curves and selected predictions. Routing, gate and balancing ablations, bucket-width sensitivity and ten-seed synthetic neural comparisons remain outside this completed block; no completion of those requests is implied."
    ]
    rendered = "\n".join(t+(r"\par" if not t.startswith("\\") and "&" not in t else "") for t in tex)
    (HERE/"manuscript_insert.tex").write_text(rendered+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","core_contrasts","supplement_contrasts")}),flush=True)


if __name__=="__main__":
    main()
