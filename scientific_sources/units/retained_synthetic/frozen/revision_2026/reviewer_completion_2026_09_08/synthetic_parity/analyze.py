"""Post-selection audit, complete descriptive tables and ordinary-input costs."""
import argparse
import csv
import json
import math
import time
import numpy as np
import torch
from torch import nn
import parity_common as c
import parity_models as pm
import campaign

HERE = c.HERE


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def independent_metrics(y, logits, weights=None):
    y = np.asarray(y, dtype=int)
    score = np.asarray(logits, dtype=float)[:, 1]-np.asarray(logits, dtype=float)[:, 0]
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    total = math.fsum(float(v) for v in w)
    accuracy = math.fsum(float(v) for yy, ss, v in zip(y, score, w) if int(ss > 0) == yy)/total
    signed = (1-2*y)*score
    ce = math.fsum(float(v)*(max(float(s), 0.)+math.log1p(math.exp(-abs(float(s))))) for s, v in zip(signed, w))/total
    # Independent sorted score groups, not the pairwise outer product.
    negatives_below, numerator = 0., 0.
    for value in np.unique(score):
        p = math.fsum(float(v) for yy, ss, v in zip(y, score, w) if ss == value and yy == 1)
        q = math.fsum(float(v) for yy, ss, v in zip(y, score, w) if ss == value and yy == 0)
        numerator += p*(negatives_below+.5*q)
        negatives_below += q
    positive = math.fsum(float(v) for yy, v in zip(y, w) if yy == 1)
    negative = math.fsum(float(v) for yy, v in zip(y, w) if yy == 0)
    return dict(accuracy=accuracy, cross_entropy=ce, auc=numerator/(positive*negative) if positive*negative else None)


def audit():
    selected = campaign.verify_selection()
    evaluation = read("evaluation.json")
    assert evaluation["selection_lock_sha256"] == c.sha(HERE/"selection_lock.json")
    assert len(evaluation["rows"]) == 400 and len(evaluation["count_lookups"]) == 40
    candidates = {spec["id"]: read("candidates/"+spec["id"]+".json") for spec in campaign.specifications()}
    max_metric_delta, max_prediction_delta = 0., 0.
    choices = {(s["n"], s["head"], s["seed"]): s for s in selected["selections"]}
    for row in evaluation["rows"]:
        choice = choices[(row["n"], row["head"], row["seed"])]
        group = [candidates[key] for key in choice["candidate_ids"]]
        valid = [r for r in group if r["status"] == "valid"]
        expected = min(valid, key=lambda r: (r["best_validation_cross_entropy"], r["spec"]["lr"])) if valid else None
        assert choice["selected_id"] == (expected["spec"]["id"] if expected else None)
        if expected is None:
            assert row["status"] == "all_candidates_failed"
            continue
        check_epochs = [(h["validation_cross_entropy"], h["epoch"]) for h in expected["history"] if "validation_cross_entropy" in h]
        assert min(check_epochs) == (expected["best_validation_cross_entropy"], expected["best_epoch"])
        cp = HERE/expected["checkpoint"]
        assert c.sha(cp) == choice["checkpoint_sha256"]
        checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
        model = pm.build_model(row["seed"], row["head"], row["n"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        canonical = c.canonical_logits(model, row["n"])
        for split, prediction_file, expected_sha, expected_metrics in [
            ("val", expected["validation_prediction"], expected["validation_prediction_sha256"],
             dict(cross_entropy=expected["best_validation_cross_entropy"])),
            ("test", row["prediction_file"], row["prediction_sha256"], row["test"])]:
            assert c.sha(HERE/prediction_file) == expected_sha
            data = c.load_data(row["n"], row["seed"], split)
            with np.load(HERE/prediction_file) as saved:
                assert np.array_equal(saved["truth"], data["y"].numpy())
                assert np.array_equal(saved["count"], data["count"])
                delta = float(np.max(np.abs(canonical[data["count"]]-saved["logits"])))
                max_prediction_delta = max(delta, max_prediction_delta)
                assert delta <= 1e-10
                assert np.max(np.abs(canonical-saved["canonical_logits"])) <= 1e-10
                metrics = independent_metrics(saved["truth"], saved["logits"])
                for key, value in expected_metrics.items():
                    if value is None:
                        assert metrics[key] is None
                    else:
                        error = abs(metrics[key]-value)
                        max_metric_delta = max(max_metric_delta, error)
                        assert error < 1e-12
        weights = [math.comb(row["n"], j)/2**row["n"] for j in range(row["n"]+1)]
        pop = independent_metrics(np.arange(row["n"]+1) % 2, canonical, weights)
        for key, value in pop.items():
            assert abs(value-row["population"][key]) < 1e-12
        assert row["population"]["success_at_099"] == (pop["accuracy"] >= .99)
        del model
    for row in evaluation["count_lookups"]:
        train, test = c.load_data(row["n"], row["seed"], "train"), c.load_data(row["n"], row["seed"], "test")
        canonical, totals = c.count_lookup(row["n"], train)
        assert c.sha(HERE/row["prediction_file"]) == row["prediction_sha256"]
        with np.load(HERE/row["prediction_file"]) as saved:
            assert np.array_equal(saved["canonical_logits"], canonical)
            assert np.array_equal(saved["training_count_frequency"], totals)
            assert np.array_equal(saved["logits"], canonical[test["count"]])
        assert c.metrics(test["y"].numpy(), canonical[test["count"]]) == row["test"]
        assert c.population_metrics(row["n"], canonical) == row["population"]
    manifest = read("data_manifest.json")
    for row in manifest["datasets"]:
        with np.load(HERE/row["path"]) as saved:
            for key, value in c.generate(row["n"], row["seed"]).items():
                assert np.array_equal(saved[key], value)
    c.write_json(HERE/"final_audit.json", dict(completed_utc=c.now(), passed=True, candidates=len(candidates),
        selections=400, valid_selections=sum(r["status"]=="valid" for r in evaluation["rows"]),
        lookup_controls=40, maximum_metric_delta=max_metric_delta, maximum_checkpoint_prediction_delta=max_prediction_delta,
        implementation_lock_sha256=c.sha(HERE/"implementation_lock.json"), selection_lock_sha256=c.sha(HERE/"selection_lock.json"),
        evaluation_sha256=c.sha(HERE/"evaluation.json"), auditor_sha256=c.sha(__file__)))
    print(json.dumps(dict(stage="parity_final_audit_passed", max_metric_delta=max_metric_delta,
                          max_prediction_delta=max_prediction_delta)), flush=True)


def write_csv(path, rows):
    assert rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                sd=float(a.std(ddof=1)) if len(a)>1 else None,
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None)


def report():
    assert read("final_audit.json")["passed"]
    e = read("evaluation.json")
    rows, summary, pairs = [], [], []
    for r in e["rows"]+e["count_lookups"]:
        valid = r.get("status", "valid") == "valid"
        rows.append(dict(n=r["n"], head=r["head"], seed=r["seed"], status=r.get("status", "valid"),
            selected_id=r.get("selected_id"), parameters=r.get("total_parameters"), selected_lr=r.get("selected_lr"),
            best_epoch=r.get("best_epoch"), epochs_run=r.get("epochs_run"),
            test_accuracy=r["test"]["accuracy"] if valid else None, test_auc=r["test"]["auc"] if valid else None,
            test_cross_entropy=r["test"]["cross_entropy"] if valid else None,
            population_accuracy=r["population"]["accuracy"] if valid else None,
            population_auc=r["population"]["auc"] if valid else None,
            population_cross_entropy=r["population"]["cross_entropy"] if valid else None,
            success=bool(r["population"]["success_at_099"]) if valid else False))
    for n in c.LENGTHS:
        for head in pm.HEADS+["count_lookup"]:
            group = [r for r in rows if r["n"] == n and r["head"] == head]
            valid = [r for r in group if r["status"] == "valid"]
            assert len(group) == 10
            summary.append(dict(n=n, head=head, prescribed=10, valid=len(valid), failed=10-len(valid),
                successes=sum(r["success"] for r in group), success_denominator=10,
                test_accuracy=statistics([r["test_accuracy"] for r in valid]),
                population_accuracy=statistics([r["population_accuracy"] for r in valid]),
                test_auc=statistics([r["test_auc"] for r in valid]),
                population_auc=statistics([r["population_auc"] for r in valid]),
                population_cross_entropy=statistics([r["population_cross_entropy"] for r in valid])))
    contrasts = [("lma2", "lma1"), ("lma3", "lma1"), ("lma3_clip1", "lma3"),
                 ("deepsets_ln", "deepsets_plain"), ("deepsets_wide", "deepsets_plain")]
    index = {(r["n"], r["head"], r["seed"]): r for r in rows}
    for n in c.LENGTHS:
        for left, right in contrasts:
            for seed in c.SEEDS:
                a, b = index[n, left, seed], index[n, right, seed]
                valid = a["status"] == "valid" and b["status"] == "valid"
                pairs.append(dict(n=n, left=left, right=right, seed=seed, valid_pair=valid,
                    population_accuracy_difference=a["population_accuracy"]-b["population_accuracy"] if valid else None,
                    test_accuracy_difference=a["test_accuracy"]-b["test_accuracy"] if valid else None,
                    population_auc_difference=a["population_auc"]-b["population_auc"] if valid else None,
                    population_cross_entropy_difference=a["population_cross_entropy"]-b["population_cross_entropy"] if valid else None))
    candidates = [read("candidates/"+s["id"]+".json") for s in campaign.specifications()]
    resources = dict(candidate_count=len(candidates), valid=sum(r["status"]=="valid" for r in candidates),
        failed=sum(r["status"]=="failed" for r in candidates), aggregate_fit_seconds=sum(r["wall_seconds"] for r in candidates),
        epoch_cap_candidates=sum(r["reached_epoch_cap"] for r in candidates),
        largest_peak_allocated_mib=max(r["peak_allocated_mib"] for r in candidates))
    c.write_json(HERE/"summary.json", dict(groups=summary, resources=resources,
        qualification="All outcomes; independent data/optimization replicates under a development-informed architecture comparison. No test-driven winner or length selection."))
    write_csv(HERE/"all_selected_seeds.csv", rows)
    write_csv(HERE/"paired_differences.csv", pairs)
    lines = ["# Ten-seed parity reliability study", "", "New data draws and declared training procedures; historical results are not reproduced by assertion.", "",
             f"Completed {resources['candidate_count']} candidates: {resources['valid']} valid, {resources['failed']} numerical failures. All 400 neural choices were fixed before test scoring. Forty count lookups use training labels only.", "",
             "Mean ± sample SD across valid data/optimization replicates. Success denominators retain all ten prescribed seeds, including failed procedures. Population metrics average over every binary count with exact binomial weights; they are not uncertainty intervals. Success means at least 99% population accuracy, not correctness on every count.", "",
             "| N | Procedure | Valid / 10 | Test accuracy | Population accuracy | Population AUC | Success / 10 |", "|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(s):
        return "failed" if s["mean"] is None else f"{s['mean']:.6f} ± {s['sd']:.6f}" if s["sd"] is not None else f"{s['mean']:.6f}"
    for s in summary:
        lines.append(f"| {s['n']} | {s['head']} | {s['valid']} | {fmt(s['test_accuracy'])} | {fmt(s['population_accuracy'])} | {fmt(s['population_auc'])} | {s['successes']} |")
    lines += ["", "All 440 selected/lookup outcomes and the 200 prescribed paired comparisons are in the accompanying CSV files. Training curves and pre-clipping gradients remain in every candidate record, including failures. Model sizes, construction/minibatch seeds, exact Janossy computation, CP precision and failure rules are fixed in the protocol and implementation lock.", "",
              "For unpositioned binary tokens, count determines parity. The learned count lookup and the separate analytic zero-error rule expose that information sufficiency. A neural failure here does not demonstrate information loss from sum pooling, a limitation of bounded-degree polynomial features, or failure of the restricted scalar quadratic theorems. Independent models at four lengths measure learning at those lengths, not extrapolation. Product order is not Boolean Fourier degree after nonlinear memory processing.", "",
              f"Aggregate retained fitting time: {resources['aggregate_fit_seconds']:.3f} seconds; candidates reaching 100 epochs: {resources['epoch_cap_candidates']}. Source/selection/prediction reconciliation is in final_audit.json. Further interpretation must preserve all lengths and the distinct first-order/cubic and hierarchical reviewer obligations."]
    (HERE/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    names = pm.HEADS+["count_lookup"]
    labels = ["DS LN", "DS plain", "DS wide", "Transformer", "Janossy-2", "CP", "k1", "k2", "k3", "k3 clip1", "Count lookup"]
    for ax, n in zip(axes.flat, c.LENGTHS):
        for j, name in enumerate(names):
            group = [r for r in rows if r["n"] == n and r["head"] == name and r["status"] == "valid"]
            vals = [r["population_accuracy"] for r in group]
            ax.scatter(np.linspace(j-.15, j+.15, len(vals)), vals, s=20, alpha=.7)
            if vals:
                ax.plot([j-.25, j+.25], [np.mean(vals)]*2, color="black", lw=2)
        ax.axhline(.99, color="#8d2b27", ls="--", lw=1)
        ax.axhline(.5, color="#888888", ls=":", lw=1)
        ax.set_title(f"N = {n}")
        ax.set_xticks(range(len(names)), labels, rotation=55, ha="right", fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("Population accuracy")
    fig.suptitle("All prescribed parity procedures: individual data/optimization replicates", fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE/"all_procedures.png", dpi=180)
    fig.savefig(HERE/"all_procedures.svg")
    plt.close(fig)
    print(json.dumps(dict(stage="parity_report_complete", groups=len(summary), individual_rows=len(rows), pairs=len(pairs))), flush=True)


def profile():
    campaign.verify_selection()
    e = read("evaluation.json")
    rows = []
    for n in c.LENGTHS:
        train = c.load_data(n, 100, "train", "cuda")
        for head in pm.HEADS:
            choice = next(r for r in e["rows"] if (r["n"], r["head"], r["seed"]) == (n, head, 100))
            if choice["status"] != "valid":
                rows.append(dict(n=n, head=head, status="no_valid_prescribed_seed100_checkpoint"))
                continue
            candidate = read("candidates/"+choice["selected_id"]+".json")
            checkpoint = torch.load(HERE/candidate["checkpoint"], map_location="cpu", weights_only=True)
            for batch in (1, 64):
                for mode in ("forward", "adamw_step"):
                    model = pm.build_model(100, head, n).cuda()
                    model.load_state_dict(checkpoint["state_dict"])
                    model.train(mode == "adamw_step")
                    torch.manual_seed(51100)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=choice["selected_lr"], weight_decay=1e-4)
                    x, y = train["x"][:batch], train["y"][:batch]
                    def operation():
                        if mode == "forward":
                            with torch.no_grad():
                                value = model(x)
                        else:
                            optimizer.zero_grad(set_to_none=True)
                            value = nn.functional.cross_entropy(model(x), y)
                            value.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), pm.clip_threshold(head))
                            optimizer.step()
                        return value
                    for _ in range(10):
                        operation()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(5):
                        start = time.perf_counter()
                        for __ in range(20):
                            value = operation()
                        torch.cuda.synchronize()
                        times.append((time.perf_counter()-start)*1000/20)
                    assert bool(torch.isfinite(value).all())
                    peak = torch.cuda.max_memory_allocated()/2**20
                    model.load_state_dict(checkpoint["state_dict"])
                    assert all(torch.equal(v.cpu(), checkpoint["state_dict"][k]) for k, v in model.state_dict().items())
                    rows.append(dict(n=n, head=head, status="valid", batch=batch, mode=mode,
                        milliseconds_blocks=times, median_milliseconds=float(np.median(times)), peak_allocated_mib=peak,
                        total_parameters=pm.parameter_count(model), checkpoint_restored=True))
                    del model, optimizer
                    torch.cuda.empty_cache()
    c.write_json(HERE/"profiles.json", dict(completed_utc=c.now(), rows=rows,
        boundary="Complete preloaded model on actual first training rows, excluding data generation/I/O; 10 warmups then five blocks of 20 operations; disposable optimizer trajectories restored afterward.",
        qualification="Janossy uses the exact binary four-type computation. Canonical validation shortcut is not used here. No generic graph/sequence throughput claim."))
    print(json.dumps(dict(stage="parity_profiles_complete", workloads=len(rows))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "report", "profile"])
    args = parser.parse_args()
    c.configure()
    {"audit": audit, "report": report, "profile": profile}[args.stage]()
