"""Complete descriptive tables and figures; no outcome-dependent omissions."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import models
import study

HERE = study.HERE
LABELS = dict(mean_count="Mean + count", pma="PMA", deepsets="LN-sum Deep Sets (w=35)",
              deepsets_wide="LN-sum Deep Sets (w=70)", janossy2="Janossy-2 readout",
              dcnv2="DCN-V2 readout", cp_pool="CP readout", lma1="LMA k=1",
              lma2="LMA k=2", lma3="LMA k=3", additive2="Additive k=2")


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def describe(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                sd=float(a.std(ddof=1)) if len(a)>1 else None,
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None,
                values=a.tolist())


def main():
    evaluation = json.loads((HERE/"evaluation.json").read_text(encoding="utf-8"))
    profiles = json.loads((HERE/"profiles.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE/"final_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and profiles["complete"]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE/"candidates").glob("*.json"))]
    tables, figures = HERE/"tables", HERE/"figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    chosen_rows = []
    for row in evaluation["rows"]:
        chosen_rows.append(dict(depth=row["depth"], head=row["head"], seed=row["seed"], status=row["status"],
            selected_id=row["selected_id"], validation_rmse=row["validation_rmse"],
            **{k:row.get(k) for k in ["total_parameters", "selected_epochs", "best_epoch", "selected_lr"]},
            **{"test_"+k:row["test"][k] if row["test"] else None for k in ("rmse", "mae", "pearson")}))
    csv_write(tables/"selected_predictors.csv", chosen_rows)
    candidate_rows, history_rows = [], []
    for row in records:
        candidate_rows.append(dict(**row["spec"], status=row["status"], best_validation_rmse=row["best_validation_rmse"],
            best_epoch=row["best_epoch"], epochs_run=row["epochs_run"], reached_epoch_cap=row["reached_epoch_cap"],
            total_parameters=row["total_parameters"], head_parameters=row["head_parameters"],
            wall_seconds=row["wall_seconds"], peak_allocated_mib=row["peak_allocated_mib"],
            error=row.get("error", "")))
        for h in row["history"]:
            history_rows.append(dict(id=row["spec"]["id"], **{k:v for k,v in h.items() if k!="cp_numerics"},
                                     **h.get("cp_numerics", {})))
    csv_write(tables/"all_candidates.csv", candidate_rows)
    csv_write(tables/"all_epoch_histories.csv", history_rows)
    summarized = []
    for depth, names in [(1, models.HEADS), (3, models.DEPTH3_HEADS)]:
        for name in names:
            rows = [r for r in chosen_rows if r["depth"]==depth and r["head"]==name and r["status"]=="valid"]
            summary = dict(depth=depth, head=name, expected_seeds=5, observed_seeds=len(rows),
                           parameters=rows[0]["total_parameters"] if rows else None)
            for metric in ("rmse", "mae", "pearson"):
                summary[metric] = describe([r["test_"+metric] for r in rows if r["test_"+metric] is not None])
            summary["epochs"] = [r["selected_epochs"] for r in rows]
            summary["best_epochs"] = [r["best_epoch"] for r in rows]
            summary["learning_rates"] = [r["selected_lr"] for r in rows]
            summarized.append(summary)
    contrast_definitions = []
    for depth in (1, 3):
        contrast_definitions += [
            (f"k=2 minus k=1 (GCN {depth})", (depth, "lma2"), (depth, "lma1")),
            (f"k=2 minus additive (GCN {depth})", (depth, "lma2"), (depth, "additive2"))]
    contrast_definitions.append(("Wider minus matched LN-sum Deep Sets", (1, "deepsets_wide"), (1, "deepsets")))
    contrast_definitions += [(LABELS[name]+": depth 3 minus 1", (3, name), (1, name)) for name in models.DEPTH3_HEADS]
    contrast_rows = []
    by_key = {(r["depth"], r["head"], r["seed"]):r for r in chosen_rows if r["status"]=="valid"}
    for title, left, right in contrast_definitions:
        pairs = [(seed, by_key[left+(seed,)]["test_rmse"]-by_key[right+(seed,)]["test_rmse"])
                 for seed in study.SEEDS if left+(seed,) in by_key and right+(seed,) in by_key]
        delta = describe([v for _,v in pairs])
        contrast_rows.append(dict(comparison=title, left_depth=left[0], left_head=left[1],
            right_depth=right[0], right_head=right[1], expected_pairs=5, observed_pairs=len(pairs),
            left_better=sum(v<0 for _,v in pairs), **delta, seeds=[s for s,_ in pairs]))
    csv_write(tables/"paired_differences.csv", [{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in contrast_rows])
    profile_rows = []
    for r in profiles["rows"]:
        for t in r.get("timings", []):
            profile_rows.append(dict(depth=r["depth"], head=r["head"], seed=42, batch_size=t["batch_size"],
                padded_n=t["padded_n"], mode=t["mode"], median_ms=t["median_block_average_ms"],
                minimum_ms=t["minimum_block_average_ms"], maximum_ms=t["maximum_block_average_ms"],
                maximum_peak_allocated_mib=max(t["peak_allocated_mib"]),
                maximum_incremental_peak_mib=max(t["incremental_peak_mib"])))
    csv_write(tables/"whole_model_profiles.csv", profile_rows)
    summary = dict(generated_utc=study.now(), evidence="Exploratory conditional five-seed comparisons on one reused ligand-only split",
        valid_candidates=audit["valid_candidates"], failed_candidates=audit["failed_candidates"],
        valid_selected=audit["valid_selected"], selected_total=85,
        aggregate_fitting_seconds=sum(r["wall_seconds"] for r in records),
        candidates_reaching_epoch_cap=sum(r["reached_epoch_cap"] for r in records),
        selected_reaching_epoch_cap=sum(r["selected_epochs"]==study.EPOCHS for r in chosen_rows if r["status"]=="valid"),
        training_mean_test_baseline=evaluation["training_mean_baseline"], summaries=summarized, contrasts=contrast_rows)
    study.write_json(HERE/"results_summary.json", summary)
    lines = ["# Complete neural comparison tables", "",
             "Exploratory results on one previously inspected ligand-only split. Each row expects five paired optimizer/selection seeds. SD is descriptive; it is not a confidence interval. All model families remain in the table.", "",
             "| GCN depth | Head | Valid/expected seeds | Parameters | Test RMSE, mean ± SD | Test MAE, mean | Pearson, mean |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for s in summarized:
        rmse = f'{s["rmse"]["mean"]:.5f} ± {s["rmse"]["sd"]:.5f}' if s["rmse"]["n"]>1 else str(s["rmse"]["mean"])
        mae = f'{s["mae"]["mean"]:.5f}' if s["mae"]["n"] else "missing"
        pearson = f'{s["pearson"]["mean"]:.5f}' if s["pearson"]["n"] else "undefined"
        lines.append(f'| {s["depth"]} | {LABELS[s["head"]]} | {s["observed_seeds"]}/5 | {s["parameters"]} | {rmse} | {mae} | {pearson} |')
    lines += ["", f'Training-mean baseline test RMSE: {evaluation["training_mean_baseline"]["rmse"]:.5f}.', "",
              "| Paired difference (left minus right) | Observed/expected | Mean ΔRMSE | SD | Range | Left lower RMSE |",
              "|---|---:|---:|---:|---|---:|"]
    for r in contrast_rows:
        if r["n"]:
            sd_label = f'{r["sd"]:.5f}' if r["sd"] is not None else "undefined"
            lines.append(f'| {r["comparison"]} | {r["n"]}/5 | {r["mean"]:+.5f} | {sd_label} | [{r["minimum"]:+.5f}, {r["maximum"]:+.5f}] | {r["left_better"]}/{r["n"]} |')
        else:
            lines.append(f'| {r["comparison"]} | 0/5 | missing | missing | missing | 0/0 |')
    lines += ["", "Negative differences favor the named left-hand method; positive differences favor the reference. Five seeds share the same data split and selection rule. No confirmatory p-value, equivalence or independent-dataset claim is made.", "",
              f'Candidate fits: {len(records)}; valid: {audit["valid_candidates"]}; failed: {audit["failed_candidates"]}. Aggregate candidate fitting time: {summary["aggregate_fitting_seconds"]/60:.2f} minutes, excluding preparation, audits, final test evaluation and profiling. Candidates reaching the 150-epoch cap: {summary["candidates_reaching_epoch_cap"]}/170; selected predictors whose candidate reached that cap: {summary["selected_reaching_epoch_cap"]}/{audit["valid_selected"]}.', "",
              "## Individual selected predictors", "",
              "| Depth | Head | Seed | LR | Selected epoch / epochs run | Validation RMSE | Test RMSE |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for r in chosen_rows:
        if r["status"]=="valid":
            lines.append(f'| {r["depth"]} | {LABELS[r["head"]]} | {r["seed"]} | {r["selected_lr"]} | {r["best_epoch"]} / {r["selected_epochs"]} | {r["validation_rmse"]:.6f} | {r["test_rmse"]:.6f} |')
        else:
            lines.append(f'| {r["depth"]} | {LABELS[r["head"]]} | {r["seed"]} | — | — | missing | missing |')
    lines += ["", "All 170 candidates and every recorded epoch are available in tables/all_candidates.csv and tables/all_epoch_histories.csv; the corresponding JSON records retain complete failure information.", "",
              "## Whole-model resource measurements", "",
              "RTX 3060 Laptop GPU. Values below use batch 64 and the same first 64 training graphs, with common padding. Times are medians of five block-average measurements (20 repetitions each), after ten warm-ups per block. Host-to-device transfers and target de-standardization are excluded. CP products/tanh use float64. Janossy evaluation is exact; training is sampled.", "",
              "| Depth | Head | Evaluation forward, ms/batch | Complete optimizer step, ms/batch | Eval incremental peak, MiB | Train incremental peak, MiB |",
              "|---:|---|---:|---:|---:|---:|"]
    for s in summarized:
        rows = [r for r in profile_rows if r["depth"]==s["depth"] and r["head"]==s["head"] and r["batch_size"]==64]
        if len(rows) != 2:
            continue
        e = next(r for r in rows if r["mode"]=="deterministic_eval_forward")
        t = next(r for r in rows if r["mode"]=="sampled_train_optimizer_step")
        lines.append(f'| {s["depth"]} | {LABELS[s["head"]]} | {e["median_ms"]:.4f} | {t["median_ms"]:.4f} | {e["maximum_incremental_peak_mib"]:.3f} | {t["maximum_incremental_peak_mib"]:.3f} |')
    (HERE/"complete_tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot_results(summarized, chosen_rows, contrast_rows, profile_rows, records, evaluation, figures)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("summaries", "contrasts")}), flush=True)


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory/(name+".svg"), bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)


def plot_results(summaries, selected, contrasts, profiles, records, evaluation, figures):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":12})
    fig, axs = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={"height_ratios":[11,6]}, layout="constrained")
    colors = {"lma2":"#12668b", "lma1":"#676767", "additive2":"#be6d25"}
    full_values = [v for s in summaries for v in s["rmse"]["values"]]
    for ax, depth in zip(axs, (1,3)):
        group = [s for s in summaries if s["depth"]==depth]
        for j, s in enumerate(group):
            values = s["rmse"]["values"]
            color = colors.get(s["head"], "#527766")
            ax.scatter(values, j+np.linspace(-.12,.12,len(values)), s=24, color=color, alpha=.7)
            if s["rmse"]["n"] > 1:
                ax.errorbar(s["rmse"]["mean"], j, xerr=s["rmse"]["sd"], fmt="D", color=color, markersize=5, capsize=3)
        ax.set_yticks(range(len(group)), [LABELS[s["head"]] for s in group])
        ax.invert_yaxis()
        ax.set_title(f'GCN depth {depth}: all five selected seeds')
        ax.set_xlabel("Test RMSE (pK units; lower is better)")
        ax.set_xlim(min(full_values)-.01,max(full_values)+.01)
        style_axis(ax)
    fig.suptitle("Ligand-only molecular comparison\nPoints: seeds · diamonds and bars: mean ± one SD", fontsize=15)
    save_figure(fig, figures, "all_neural_results")
    fig, axs = plt.subplots(2, 1, figsize=(11, 8.5), layout="constrained")
    for ax, items, title in zip(axs, [contrasts[:5], contrasts[5:]], ["Head comparisons", "Backbone depth comparisons"]):
        for j, r in enumerate(items):
            vals = r["values"]
            ax.scatter(vals, j+np.linspace(-.10,.10,len(vals)), color="#12668b", s=26)
            if r["n"] > 1:
                ax.errorbar(r["mean"], j, xerr=r["sd"], fmt="D", color="#1a2830", markersize=5, capsize=3)
        ax.axvline(0, color="#666666", ls="--", lw=1)
        ax.set_yticks(range(len(items)), [r["comparison"] for r in items])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Paired ΔRMSE: negative favors the left-hand method")
        style_axis(ax)
    fig.suptitle("Paired outcomes on the same reused split\nPoints: five seeds · bars: one SD, not confidence intervals", fontsize=14)
    save_figure(fig, figures, "paired_neural_differences")
    fig, axs = plt.subplots(5, 4, figsize=(13, 13), layout="constrained")
    lookup = {r["spec"]["id"]:r for r in records}
    for ax, s in zip(axs.flat, summaries):
        ids = [r["selected_id"] for r in selected if r["depth"]==s["depth"] and r["head"]==s["head"] and r["status"]=="valid"]
        for i, id_ in enumerate(ids):
            h = [h for h in lookup[id_]["history"] if "validation_rmse" in h]
            ax.plot([v["epoch"] for v in h], [v["validation_rmse"] for v in h], lw=.9, alpha=.75, label=str(study.SEEDS[i]))
        ax.set_title(f'GCN{s["depth"]} · {LABELS[s["head"]]}', fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Validation RMSE", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
    for ax in list(axs.flat)[len(summaries):]:
        ax.set_visible(False)
    legend_axis = list(axs.flat)[len(summaries)]
    legend_axis.set_visible(True)
    legend_axis.set_axis_off()
    handles, labels = axs.flat[0].get_legend_handles_labels()
    legend_axis.legend(handles,labels,title="Paired seed",loc="center",frameon=False)
    fig.suptitle("Selected candidates: all five validation trajectories per configuration\nAxes retain each panel's observed range; every candidate history is also exported", fontsize=13)
    save_figure(fig, figures, "validation_trajectories")


if __name__ == "__main__":
    main()
