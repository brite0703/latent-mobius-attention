"""Plot complete validation-selected histories without smoothing or censoring runs."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
LABELS = {"deepsets_ln":"Deep Sets + LN", "deepsets_plain":"Deep Sets, plain",
    "deepsets_wide":"Deep Sets, wide", "transformer":"Transformer", "janossy2":"Janossy, order two",
    "cp_pool":"CP", "lma1":"LMA, order one", "lma2":"LMA, order two", "lma3":"LMA, order three",
    "lma3_clip1":"LMA three, clip one", "additive2":"Additive, order two", "additive3":"Additive, order three"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    HERE.mkdir(exist_ok=True)
    traces, points, figures, sources = [], [], [], []
    output = HERE/"output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output/"synthetic_learning_curves.pdf"
    book = PdfPages(pdf_path, metadata={"Title":"Complete selected synthetic learning curves", "Author":"Jih-Jeng Huang"})
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.titlesize":8.5,
                         "axes.labelsize":8, "xtick.labelsize":7, "ytick.labelsize":7,
                         "svg.fonttype":"none", "savefig.facecolor":"white"})
    for block in ("synthetic_parity", "first_cubic", "hierarchical_sequence"):
        folder = COMPLETION/block
        final = read(folder/"final_audit.json")
        assert final["passed"]
        lock = read(folder/"selection_lock.json")
        sources.extend(dict(path=str(folder/name), sha256=sha(folder/name)) for name in ("selection_lock.json", "final_audit.json"))
        records = {}
        for artifact in lock["candidate_records"]:
            path = folder/artifact["path"]
            assert sha(path) == artifact["sha256"]
            records[path.stem] = read(path)
        condition_key = "n" if block == "synthetic_parity" else "task"
        conditions = list(dict.fromkeys(row[condition_key] for row in lock["selections"]))
        training_key = "training_cross_entropy" if block == "synthetic_parity" else "training_loss"
        validation_key = "validation_cross_entropy" if block == "synthetic_parity" else "validation_loss"
        best_key = "best_validation_cross_entropy" if block == "synthetic_parity" else "best_validation_loss"
        for condition in conditions:
            chosen = [row for row in lock["selections"] if row[condition_key] == condition]
            heads = list(dict.fromkeys(row["head"] for row in chosen))
            columns = 3 if len(heads) > 6 else 2
            nrows = math.ceil(len(heads)/columns)
            figure, axes = plt.subplots(nrows, columns, figsize=(7.0, 9.0 if nrows == 4 else 8.1), squeeze=False)
            figure.subplots_adjust(left=.115, right=.985, top=.91, bottom=.22, hspace=.55,
                                   wspace=.65 if columns == 3 else .35)
            seeds = sorted({row["seed"] for row in chosen})
            assert len(seeds) == 10
            color_for = {seed:plt.get_cmap("tab10")(i) for i, seed in enumerate(seeds)}
            title = (f"Parity: N = {condition}" if block == "synthetic_parity" else
                     "First-degree target" if condition == "first" else "One overlapping-cubic target" if condition == "cubic" else
                     "Position-aware hierarchy: tree depth "+str(condition)[-1])
            figure.suptitle(title, x=.105, y=.982, ha="left", fontsize=12)
            figure.text(.105, .948, "All ten validation-selected runs; traces end at their actual stopping epochs", fontsize=8.5)
            for axis, head in zip(axes.flat, heads):
                axis_values = []
                for choice in [row for row in chosen if row["head"] == head]:
                    identifier = choice["selected_id"]
                    assert identifier is not None
                    record = records[identifier]
                    assert record["status"] == "valid"
                    assert record["spec"]["seed"] == choice["seed"] and record["spec"]["head"] == head
                    history = record["history"]
                    assert [row["epoch"] for row in history] == list(range(1, record["epochs_run"]+1))
                    validation = [(row[validation_key], row["epoch"]) for row in history if validation_key in row]
                    assert min(validation) == (record[best_key], record["best_epoch"])
                    path = folder/"candidates"/(identifier+".json")
                    traces.append(dict(block=block, condition=condition, head=head, seed=choice["seed"],
                        selected_id=identifier, learning_rate=record["spec"]["lr"], epochs_run=record["epochs_run"],
                        best_epoch=record["best_epoch"], reached_epoch_cap=record["reached_epoch_cap"],
                        candidate_path=str(path), candidate_sha256=sha(path)))
                    for kind, field, style, alpha in (("training", training_key, "--", .32), ("validation", validation_key, "-", .9)):
                        entries = [(row["epoch"], row[field]) for row in history if field in row]
                        assert entries and all(math.isfinite(value) and value >= 0 for _, value in entries)
                        axis_values.extend(value for _, value in entries)
                        axis.plot([epoch for epoch, _ in entries], [value for _, value in entries],
                            color=color_for[choice["seed"]], linestyle=style, alpha=alpha, linewidth=.75)
                        points.extend(dict(block=block, condition=condition, head=head, seed=choice["seed"],
                            selected_id=identifier, series=kind, epoch=epoch, loss=value) for epoch, value in entries)
                    axis.scatter([record["best_epoch"]], [record[best_key]], s=10,
                        color=[color_for[choice["seed"]]], edgecolor="none", zorder=4)
                axis.set_title(LABELS[head], loc="left", pad=5)
                if min(axis_values) == 0:
                    axis.set_yscale("symlog", linthresh=1e-7)
                    scale_label = "symlog"
                elif max(axis_values)/min(axis_values) > 8:
                    axis.set_yscale("log")
                    scale_label = "log"
                else:
                    scale_label = "linear"
                axis.set_xlim(0, 102)
                axis.set_xticks([0, 50, 100])
                axis.set_xlabel("Epoch")
                axis.set_ylabel(("MSE" if block == "first_cubic" else "Cross-entropy")+" ("+scale_label+")")
                axis.grid(True, color="#dddddd", linewidth=.4, alpha=.7)
                axis.spines[["top", "right"]].set_visible(False)
            for axis in list(axes.flat)[len(heads):]:
                axis.set_visible(False)
            handles = [Line2D([0], [0], color=color_for[seed], linewidth=1.3, label=str(seed)) for seed in seeds]
            figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(.095,.075), ncol=5,
                          frameon=False, title="Data / initialization seed", fontsize=7.5, title_fontsize=8)
            figure.text(.105, .044, "Solid: scheduled validation loss. Faint dashed: stored within-epoch training loss. Dot: chosen checkpoint.", fontsize=7.1)
            figure.text(.105, .024, "No smoothing, average or confidence band. Axis scales and ranges vary by panel; symlog is linear below 1e-7.", fontsize=7.1)
            stem = block+"_"+str(condition)
            paths = [HERE/(stem+suffix) for suffix in (".png", ".svg")]
            figure.savefig(paths[0], dpi=170)
            figure.savefig(paths[1])
            book.savefig(figure)
            plt.close(figure)
            figures.append(dict(block=block, condition=condition, procedures=len(heads), selected_runs=len(chosen),
                files=[dict(path=str(path), sha256=sha(path)) for path in paths]))
    book.close()
    assert len(traces) == 780 and len(figures) == 9
    csv_path = HERE/"curve_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream))
    assert len(restored) == len(points)
    assert all(float(saved["loss"]) == point["loss"] and int(saved["epoch"]) == point["epoch"] for saved, point in zip(restored, points))
    manifest = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), selected_runs=780,
        point_count=len(points), figures=figures, traces=traces, source_locks=sources,
        plot_source_sha256=sha(Path(__file__)), points_csv_sha256=sha(csv_path),
        pdf=dict(path=str(pdf_path), sha256=sha(pdf_path), pages=9),
        qualifications=["All validation-selected histories from all three completed synthetic campaigns; no test-based curve selection.",
            "Training traces are the stored within-epoch objective; validation traces use scheduled fixed-checkpoint evaluation. Their difference is not a controlled generalization-gap estimate.",
            "Both learning-rate candidates remain in the original evidence archives; these figures display the selected candidate for every prescribed seed.",
            "No line is extended after its recorded stop and no incomplete-run mean or confidence interval is plotted.",
            "Positive-valued panels use a log axis when the largest/smallest plotted loss exceeds eight and a linear axis otherwise. Panels with zeros use symlog with linear threshold 1e-7. Each axis states its scale; ranges are panel-specific.",
            "These are visual diagnostics of finite-budget trajectories, not a convergence or causal-instability analysis."])
    (HERE/"curve_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=True, selected_runs=780, point_count=len(points), figures=9), indent=2), flush=True)


if __name__ == "__main__":
    main()
