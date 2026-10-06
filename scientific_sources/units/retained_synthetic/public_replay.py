"""Portable, read-only reconciliation and CPU replay of retained synthetic results."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path, PureWindowsPath
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import sys
import traceback
import contextlib
import io

HERE = Path(__file__).resolve().parent
REPORTS = None
CANDIDATE_ROOT = HERE.parents[1]
COMP_REL = "revision_2026/reviewer_completion_2026_09_08"
STUDIES = {"synthetic_parity": (800, 400), "first_cubic": (400, 200),
           "hierarchical_sequence": (360, 180)}
BLOCKED_READS = []
SCIENTIFIC_COMPILES = set()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def safe_relative(base, value):
    legacy = PureWindowsPath(value)
    if legacy.is_absolute() or legacy.drive or ".." in legacy.parts:
        raise ValueError("Unsafe relative artifact path: " + str(value))
    path = base.joinpath(*legacy.parts).resolve()
    if not within(path, base.resolve()):
        raise ValueError("Artifact escapes its containing directory")
    return path


def historical_path(value, manifest):
    relative = PureWindowsPath(value).relative_to(PureWindowsPath(manifest["historical_root"]))
    return safe_relative(HERE / "frozen", relative.as_posix())


def deny_original_workspace(manifest):
    legacy = PureWindowsPath(manifest["historical_root"])

    def hook(event, args):
        if event == "compile" and len(args) > 1 and isinstance(args[1], str):
            compiled = Path(args[1]).resolve()
            if within(compiled, HERE / "frozen"):
                SCIENTIFIC_COMPILES.add(compiled.relative_to(HERE).as_posix())
            return
        if event != "open" or not args or isinstance(args[0], int):
            return
        raw = os.fsdecode(args[0])
        local = Path(raw).resolve()
        if within(local, HERE):
            return
        if within(PureWindowsPath(str(local)), legacy) or within(PureWindowsPath(raw), legacy):
            BLOCKED_READS.append(raw)
            raise PermissionError("Original-workspace fallback rejected: " + raw)
    sys.addaudithook(hook)



EDITORIAL_SOURCE_OMISSIONS = {
    "frozen/revision_2026/reviewer_completion_2026_09_08/synthetic_parity/web_review_27_disposition.md":
        "40b5c80ba94e59f9f4c11c9d7d580c7c0365afd04e9dd3882b727813e875cab4",
}


def declared_editorial_omission(path, expected_sha256):
    relative = path.resolve().relative_to(HERE.resolve()).as_posix()
    if relative not in EDITORIAL_SOURCE_OMISSIONS:
        return False
    assert expected_sha256 == EDITORIAL_SOURCE_OMISSIONS[relative], "Editorial reference differs"
    assert not path.exists(), "The public source exception requires an absent editorial body"
    return True


def verify_bundle(manifest):
    assert not list((HERE / "frozen").rglob("*.pyc")), "Unmanifested scientific bytecode must not be used"
    seen, checked, omissions = set(), [], []
    for item in manifest["files"]:
        if item["path"] in seen:
            raise ValueError("Duplicate bundle entry")
        seen.add(item["path"])
        path = safe_relative(HERE, item["path"])
        if declared_editorial_omission(path, item["sha256"]):
            omissions.append(dict(path=item["path"], original_sha256=item["sha256"],
                                  status="Editorial body not distributed; its bytes are not verified"))
            continue
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Bundle integrity failure: " + item["path"])
        checked.append(item)
    assert {x["path"] for x in omissions} == set(EDITORIAL_SOURCE_OMISSIONS)
    return dict(files=len(checked), bytes=sum(item["bytes"] for item in checked),
                original_inventory_files=len(seen), declared_editorial_omissions=omissions,
                scope="Every distributed original bundle file is checked; one named editorial body is explicitly unavailable")


def classification(y, logits, weights=None):
    import numpy as np
    y, logits = np.asarray(y, dtype=np.int64), np.asarray(logits, dtype=np.float64)
    assert logits.shape == (len(y), 2) and np.isfinite(logits).all()
    assert set(np.unique(y)).issubset({0, 1})
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    assert w.shape == y.shape and np.isfinite(w).all() and (w >= 0).all()
    total = math.fsum(w)
    assert total > 0
    scores = logits[:, 1] - logits[:, 0]
    accuracy = math.fsum(w * ((scores > 0) == y)) / total
    ce = math.fsum(w * np.logaddexp(0., (1 - 2 * y) * scores)) / total
    pos, neg = math.fsum(w[y == 1]), math.fsum(w[y == 0])
    auc = None
    if pos > 0 and neg > 0:
        order = np.argsort(scores, kind="stable")
        negative_before, terms, start = 0., [], 0
        while start < len(order):
            end = start + 1
            while end < len(order) and scores[order[end]] == scores[order[start]]:
                end += 1
            group = order[start:end]
            positive_here = math.fsum(w[group][y[group] == 1])
            negative_here = math.fsum(w[group][y[group] == 0])
            terms.append(positive_here * (negative_before + negative_here / 2))
            negative_before += negative_here
            start = end
        auc = math.fsum(terms) / (pos * neg)
    return dict(accuracy=accuracy, auc=auc, cross_entropy=ce)


def regression(y, prediction):
    import numpy as np
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    errors = [float(p) - float(t) for p, t in zip(prediction, y)]
    return dict(mse=math.fsum(e * e for e in errors) / len(errors),
                mae=math.fsum(abs(e) for e in errors) / len(errors))


def measured(study, y, prediction):
    return regression(y, prediction) if study == "first_cubic" else classification(y, prediction)


def metric_difference(actual, recorded):
    maximum = 0.
    for key, expected in recorded.items():
        value = actual[key]
        if expected is None or isinstance(expected, bool):
            assert value == expected, (key, value, expected)
        else:
            difference = abs(float(value) - float(expected))
            assert difference <= 1e-11, (key, value, expected)
            maximum = max(maximum, difference)
    return maximum


class Inputs:
    def __init__(self):
        self.cache = {}
        self.files = 0
        self.rows = 0

    def full(self, study, spec):
        import numpy as np
        task = spec.get("n", spec.get("task"))
        key = study, task if study != "first_cubic" else "both", spec["seed"]
        if key in self.cache:
            return self.cache[key]
        folder = HERE / "frozen" / COMP_REL / study / "data"
        seed = spec["seed"]
        if study == "synthetic_parity":
            n = spec["n"]
            rng = np.random.default_rng(2026090803 + 1000 * n + seed)
            x = rng.integers(0, 2, (6000, n), dtype=np.uint8)
            count = x.sum(axis=1, dtype=np.int64)
            expected = dict(x=x, y=count % 2, count=count)
            path = folder / f"n{n}_seed{seed}.npz"
        elif study == "hierarchical_sequence":
            depth = int(spec["task"][-1])
            n = 3 ** depth
            rng = np.random.default_rng(2026090805 + 1000 * depth + seed)
            x = rng.integers(0, 2, (6000, n), dtype=np.uint8)
            signs = x.astype(np.int64) * 2 - 1
            while signs.shape[1] > 1:
                # Independent sign-polynomial recurrence, rather than the original
                # generator's majority-count operation.
                a, b, c = signs[:, 0::3], signs[:, 1::3], signs[:, 2::3]
                signs = (a + b + c - a * b * c) // 2
            expected = dict(x=x, y=(signs[:, 0] > 0).astype(np.int64),
                            count=x.sum(axis=1, dtype=np.int64))
            path = folder / f"{spec['task']}_seed{seed}.npz"
        else:
            bits = np.asarray([[(state // (2 ** j)) % 2 for j in range(12)]
                               for state in range(4096)], dtype=np.uint8)
            signs = bits.astype(np.int64) * 2 - 1
            weights = [1, -1, 1, 1, -1, 1, -1, 1]
            support = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 5, 6),
                       (0, 5, 7), (8, 9, 10), (8, 9, 11), (2, 6, 10)]
            first = sum(w * signs[:, j] for j, w in enumerate(weights)) / math.sqrt(8)
            cubic = sum(w * signs[:, a] * signs[:, b] * signs[:, c]
                        for w, (a, b, c) in zip(weights, support)) / math.sqrt(8)
            order = np.random.default_rng(2026090806 + seed).permutation(4096)
            expected = dict(bits=bits, first_target=first, cubic_target=cubic,
                            train_ids=order[:2048], val_ids=order[2048:3072], test_ids=order[3072:])
            path = folder / f"seed{seed}.npz"
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == set(expected)
            for field, value in expected.items():
                np.testing.assert_array_equal(saved[field], value, err_msg=str(path) + ": " + field)
        self.files += 1
        self.rows += len(expected.get("x", expected.get("bits")))
        self.cache[key] = expected
        return expected

    def split(self, study, spec, split):
        import numpy as np
        full = self.full(study, spec)
        if study == "first_cubic":
            ids = np.arange(4096) if split == "population" else full[split + "_ids"]
            return dict(ids=ids, truth=full[spec["task"] + "_target"][ids], mask=full["bits"][ids])
        slices = dict(train=slice(0, 4800), val=slice(4800, 5400), test=slice(5400, 6000))
        selected = slices[split]
        return dict(ids=np.arange(6000)[selected], truth=full["y"][selected],
                    x=full["x"][selected], count=full["count"][selected])


def assert_prediction_rows(study, saved, data):
    import numpy as np
    np.testing.assert_array_equal(saved["truth"], data["truth"])
    if study == "synthetic_parity":
        np.testing.assert_array_equal(saved["count"], data["count"])
        np.testing.assert_array_equal(saved["logits"], saved["canonical_logits"][data["count"]])
        return saved["logits"]
    np.testing.assert_array_equal(saved["ids"], data["ids"])
    return saved["prediction"]


def metric_row(study, spec, split, metrics, role="neural"):
    return dict(study=study, task=str(spec.get("task", spec.get("n"))),
                head=spec["head"], seed=spec["seed"], split=split, role=role, **metrics)


def expected_specifications(study, lock):
    import numpy as np
    if study == "synthetic_parity":
        heads = ["deepsets_ln", "deepsets_plain", "deepsets_wide", "transformer",
                 "janossy2", "cp_pool", "lma1", "lma2", "lma3", "lma3_clip1"]
        rows = [dict(id=f"n{n}_{head}_seed{seed}_lr{j}", n=n, head=head,
                     seed=seed, lr=lr, lr_index=j)
                for n in (10, 20, 40, 80) for head in heads for seed in range(100, 110)
                for j, lr in enumerate((.0003, .001))]
        permutation = 2026090804
    else:
        tasks = ["first", "cubic"] if study == "first_cubic" else ["depth2", "depth3", "depth4"]
        seeds = list(range(100, 110)) if study == "first_cubic" else list(range(200, 210))
        assert lock["tasks"] == tasks and lock["seeds"] == seeds
        heads = lock["heads"]
        expected_heads = ({"deepsets_ln", "deepsets_plain", "deepsets_wide", "janossy2", "cp_pool",
                           "lma1", "lma2", "lma3", "additive2", "additive3"} if study == "first_cubic" else
                          {"lma1", "lma2", "lma3", "transformer", "deepsets_wide", "cp_pool"})
        assert set(heads) == expected_heads and len(heads) == len(expected_heads)
        rows = [dict(id=f"{task}_{head}_seed{seed}_lr{j}", task=task, head=head,
                     seed=seed, lr=lr, lr_index=j)
                for task in tasks for head in heads for seed in seeds
                for j, lr in enumerate((.0003, .001))]
        permutation = 2026090807 if study == "first_cubic" else 2026090808
    return [rows[int(i)] for i in np.random.default_rng(permutation).permutation(len(rows))]


def check_references(study, folder, evaluation, inputs):
    import numpy as np
    rows, maximum, checked = [], 0., 0
    refs = evaluation.get("references", evaluation.get("count_lookups", []))
    assert len(refs) == (30 if study == "hierarchical_sequence" else 40)
    if study == "first_cubic":
        choices = read(folder / "closed_form_selection.json")
        assert choices["lock_sha256"] == sha(folder / "closed_form_lock.json")
        assert len(choices["candidates"]) == 120 and len(choices["selections"]) == 40
        for item in choices["candidates"]:
            assert sha(safe_relative(folder, item["path"])) == item["sha256"]
    designs = {}
    for row in refs:
        path = safe_relative(folder, row["prediction_file"])
        assert sha(path) == row["prediction_sha256"]
        if study == "first_cubic":
            group = [read(folder / "closed_form_candidates" / (cid + ".json")) for cid in row["candidate_ids"]]
            assert len(group) == 3 and len({r["lambda_value"] for r in group}) == 3
            chosen = min(group, key=lambda r: (r["validation_mse"], r["lambda_value"]))
            assert chosen["id"] == row["selected_id"]
            coefficient_path = safe_relative(folder, chosen["coefficients"])
            assert sha(coefficient_path) == chosen["coefficients_sha256"] == row["coefficients_sha256"]
            full = inputs.full(study, row)
            degree = row["degree"]
            if degree not in designs:
                signs = full["bits"].astype(np.float64) * 2 - 1
                designs[degree] = np.column_stack([np.prod(signs[:, subset], axis=1)
                    for size in range(1, degree + 1) for subset in itertools.combinations(range(12), size)])
            with np.load(coefficient_path, allow_pickle=False) as coefficient:
                prediction = designs[degree] @ coefficient["coefficient"] + coefficient["intercept"]
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], np.arange(4096))
                np.testing.assert_array_equal(saved["test_ids"], full["test_ids"])
                np.testing.assert_array_equal(saved["truth"], full[row["task"] + "_target"])
                np.testing.assert_allclose(prediction, saved["prediction"], atol=1e-10, rtol=1e-12)
                for split in ("test", "population"):
                    ids = full["test_ids"] if split == "test" else np.arange(4096)
                    result = regression(saved["truth"][ids], saved["prediction"][ids])
                    maximum = max(maximum, metric_difference(result, row[split]))
                    rows.append(metric_row(study, row, split, result, "polynomial_reference"))
        else:
            train, test = [inputs.split(study, row, split) for split in ("train", "test")]
            n = row["n"] if study == "synthetic_parity" else 3 ** int(row["task"][-1])
            totals = np.bincount(train["count"], minlength=n + 1)
            positives = np.bincount(train["count"], weights=train["truth"], minlength=n + 1)
            p = (positives + 1) / (totals + 2)
            canonical = np.column_stack((np.log1p(-p), np.log(p)))
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["training_count_frequency"], totals)
                np.testing.assert_array_equal(saved["canonical_logits" if study == "synthetic_parity" else "logits_by_count"], canonical)
                prediction = assert_prediction_rows(study, saved, test)
                np.testing.assert_array_equal(prediction, canonical[test["count"]])
                result = classification(test["truth"], prediction)
                maximum = max(maximum, metric_difference(result, row["test"]))
                rows.append(metric_row(study, row, "test", result, "count_reference"))
            if study == "synthetic_parity":
                counts = np.arange(n + 1)
                result = classification(counts % 2, canonical, [math.comb(n, j) / 2 ** n for j in counts])
                result["success_at_099"] = result["accuracy"] >= .99
                maximum = max(maximum, metric_difference(result, row["population"]))
                rows.append(metric_row(study, row, "population", result, "count_reference"))
        checked += 1
    return dict(references=checked, maximum_reference_metric_difference=maximum), rows


def records(manifest):
    import numpy as np
    inputs, summaries, metric_rows, selection_margins = Inputs(), [], [], []
    for study, expected_counts in STUDIES.items():
        folder = HERE / "frozen" / COMP_REL / study
        lock_path, selection_path = folder / "implementation_lock.json", folder / "selection_lock.json"
        lock, selection = read(lock_path), read(selection_path)
        for item in lock["files"]:
            source_path = historical_path(item["path"], manifest)
            if not declared_editorial_omission(source_path, item["sha256"]):
                assert sha(source_path) == item["sha256"]
        assert selection["implementation_lock_sha256"] == sha(lock_path)
        assert (len(lock["candidates"]), len(selection["selections"])) == expected_counts
        assert len(selection["candidate_records"]) == expected_counts[0]
        assert lock["candidates"] == expected_specifications(study, lock)
        specs = {spec["id"]: spec for spec in lock["candidates"]}
        assert len(specs) == expected_counts[0]
        candidates, maximum, validation_rows = {}, 0., 0
        for item in selection["candidate_records"]:
            path = safe_relative(folder, item["path"])
            assert sha(path) == item["sha256"]
            candidate = read(path)
            cid = candidate["spec"]["id"]
            assert cid not in candidates and candidate["spec"] == specs[cid]
            assert candidate["implementation_lock_sha256"] == sha(lock_path)
            assert candidate["status"] in ("valid", "failed")
            candidates[cid] = candidate
            if candidate["status"] != "valid":
                continue
            vp = safe_relative(folder, candidate["validation_prediction"])
            assert sha(vp) == candidate["validation_prediction_sha256"]
            cp = safe_relative(folder, candidate["checkpoint"])
            assert sha(cp) == candidate["checkpoint_sha256"]
            data = inputs.split(study, candidate["spec"], "val")
            with np.load(vp, allow_pickle=False) as saved:
                prediction = assert_prediction_rows(study, saved, data)
                result = measured(study, data["truth"], prediction)
            field = "cross_entropy" if study != "first_cubic" else "mse"
            best_field = "best_validation_cross_entropy" if study == "synthetic_parity" else "best_validation_loss"
            history_field = "validation_cross_entropy" if study == "synthetic_parity" else "validation_loss"
            delta = abs(result[field] - candidate[best_field])
            assert delta <= 1e-11, (study, cid, delta)
            maximum = max(maximum, delta)
            history = [r for r in candidate["history"] if history_field in r]
            assert [r["epoch"] for r in candidate["history"]] == list(range(1, candidate["epochs_run"] + 1))
            expected_epochs = [epoch for epoch in range(1, candidate["epochs_run"] + 1)
                               if epoch == 1 or epoch % 5 == 0 or epoch == 100]
            assert [r["epoch"] for r in history] == expected_epochs
            best = min(history, key=lambda r: (r[history_field], r["epoch"]))
            assert (best["epoch"], best[history_field]) == (candidate["best_epoch"], candidate[best_field])
            running_best, bad = float("inf"), 0
            for event in history:
                if event[history_field] < running_best:
                    running_best, bad = event[history_field], 0
                else:
                    bad += 1
            assert candidate["epochs_run"] == 100 or bad >= 8
            validation_rows += len(data["truth"])
        seen_groups, seen_candidates = set(), set()
        for choice in selection["selections"]:
            key = choice.get("task", choice.get("n")), choice["head"], choice["seed"]
            assert key not in seen_groups
            seen_groups.add(key)
            group = [candidates[cid] for cid in choice["candidate_ids"]]
            assert len(group) == 2
            assert sorted(c["spec"]["lr"] for c in group) == [.0003, .001]
            for c in group:
                spec = c["spec"]
                assert (spec.get("task", spec.get("n")), spec["head"], spec["seed"]) == key
                assert spec["id"] not in seen_candidates
                seen_candidates.add(spec["id"])
            valid = [c for c in group if c["status"] == "valid"]
            best_field = "best_validation_cross_entropy" if study == "synthetic_parity" else "best_validation_loss"
            expected = min(valid, key=lambda c: (c[best_field], c["spec"]["lr"])) if valid else None
            assert choice["selected_id"] == (expected["spec"]["id"] if expected else None)
            if expected:
                assert choice["checkpoint_sha256"] == expected["checkpoint_sha256"]
            ordered = sorted(valid, key=lambda c: (c[best_field], c["spec"]["lr"]))
            selection_margins.append(dict(study=study, task=str(key[0]), head=key[1], seed=key[2],
                selected_id=choice["selected_id"], valid_candidates=len(valid),
                recorded_loss_margin=ordered[1][best_field] - ordered[0][best_field] if len(ordered) == 2 else None,
                exact_recorded_loss_tie=len(ordered) == 2 and ordered[1][best_field] == ordered[0][best_field]))
        assert seen_candidates == set(specs)
        # Every candidate and choice is reconciled before held-out metric reconstruction.
        evaluation = read(folder / "evaluation.json")
        assert evaluation["selection_lock_sha256"] == sha(selection_path)
        assert len(evaluation["rows"]) == expected_counts[1]
        prediction_rows = 0
        for choice, row in zip(selection["selections"], evaluation["rows"]):
            for key, value in choice.items():
                assert row[key] == value
            if choice["selected_id"] is None:
                assert row["status"] == "all_candidates_failed"
                continue
            spec = candidates[choice["selected_id"]]["spec"]
            splits = ["test"] if study != "first_cubic" else ["test", "population"]
            for split in splits:
                field = "prediction_file" if study == "synthetic_parity" else split + "_prediction_file"
                path = safe_relative(folder, row[field])
                assert sha(path) == row[field[:-5] + "_sha256"]
                data = inputs.split(study, spec, split)
                with np.load(path, allow_pickle=False) as saved:
                    prediction = assert_prediction_rows(study, saved, data)
                    result = measured(study, data["truth"], prediction)
                    maximum = max(maximum, metric_difference(result, row[split]))
                    metric_rows.append(metric_row(study, spec, split, result))
                    if study == "synthetic_parity":
                        n, canonical = spec["n"], saved["canonical_logits"]
                        counts = np.arange(n + 1)
                        population = classification(counts % 2, canonical,
                            [math.comb(n, j) / 2 ** n for j in counts])
                        population["success_at_099"] = population["accuracy"] >= .99
                        maximum = max(maximum, metric_difference(population, row["population"]))
                        metric_rows.append(metric_row(study, spec, "population", population))
                prediction_rows += len(data["truth"])
        reference_summary, reference_rows = check_references(study, folder, evaluation, inputs)
        metric_rows.extend(reference_rows)
        summary = dict(study=study, candidates=len(candidates), choices=len(seen_groups),
            valid_candidates=sum(c["status"] == "valid" for c in candidates.values()),
            validation_prediction_rows=validation_rows, held_out_or_population_rows=prediction_rows,
            maximum_metric_difference=maximum, **reference_summary)
        summaries.append(summary)
        print(json.dumps(dict(stage="records_reconciled", **summary)), flush=True)
    assert inputs.files == 80
    write_csv(REPORTS / "recomputed_metrics.csv", metric_rows)
    write_csv(REPORTS / "selection_margins.csv", selection_margins)
    grouped = {}
    for row in metric_rows:
        key = tuple(row[field] for field in ("study", "task", "head", "split", "role"))
        grouped.setdefault(key, []).append(row)
    aggregated = []
    for key, group in sorted(grouped.items()):
        result = dict(zip(("study", "task", "head", "split", "role"), key), seeds=len(group))
        assert len(group) == 10
        for measure in ("mse", "mae", "accuracy", "auc", "cross_entropy", "success_at_099"):
            if measure in group[0] and all(r[measure] is not None for r in group):
                values = np.asarray([r[measure] for r in group], dtype=float)
                result[measure + "_mean"] = float(values.mean())
                result[measure + "_sample_sd"] = float(values.std(ddof=1))
        aggregated.append(result)
    write_csv(REPORTS / "recomputed_summary.csv", aggregated)
    return dict(passed=True, studies=summaries, independent_input_files=inputs.files,
                independent_generated_rows=inputs.rows, metric_rows=len(metric_rows), summary_rows=len(aggregated),
                selection_histories="Complete scheduled scalar histories reconciled; other epoch checkpoints are not regenerated.")


def write_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def import_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execution_environment(torch, np):
    import importlib.metadata
    numpy_text = io.StringIO()
    with contextlib.redirect_stdout(numpy_text):
        np.show_config()
    return dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
        machine=platform.machine(), processor=platform.processor(),
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        numpy=np.__version__, torch=torch.__version__, torch_build=torch.__config__.show(),
        numpy_build=numpy_text.getvalue(), sklearn=importlib.metadata.version("scikit-learn"),
        scipy=importlib.metadata.version("scipy"),
        environment={key: os.environ.get(key) for key in
                     ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def cpu_metrics_delta(actual, recorded):
    return {key: None if actual[key] is None or recorded[key] is None else float(actual[key]) - float(recorded[key])
            for key in recorded if not isinstance(recorded[key], bool)}


def save_cpu_prediction(study, cid, split, data, prediction):
    import numpy as np
    path = REPORTS / "predictions" / study / (cid + "_" + split + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=data["ids"], truth=data["truth"], prediction=prediction)
    temporary.replace(path)
    return dict(path=path.relative_to(REPORTS).as_posix(), sha256=sha(path))


def check_model_state(model, checkpoint, torch):
    state = model.state_dict()
    assert state.keys() == checkpoint["state_dict"].keys()
    for key, expected in checkpoint["state_dict"].items():
        actual = state[key]
        assert actual.device.type == "cpu" and actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected) and bool(torch.isfinite(actual).all())
    combination_buffers = 0
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if name.startswith("combos_"):
                order = int(name[len("combos_"):])
                expected = torch.tensor(list(itertools.combinations(range(module.M), order)), dtype=buffer.dtype)
                assert torch.equal(buffer, expected)
                combination_buffers += 1
    tensors = dict(list(model.named_parameters()) + list(model.named_buffers()))
    assert all(value.device.type == "cpu" for value in tensors.values())
    before = {name: value.detach().clone() for name, value in tensors.items()}
    return before, combination_buffers


def cpu_replay(manifest, manifest_digest):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized(), {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "available": torch.cuda.is_available(), "initialized": torch.cuda.is_initialized()}
    inputs, all_reports = Inputs(), []
    environment = execution_environment(torch, np)
    environment_digest = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    executed, reused = 0, 0
    source_digest = sha(__file__)
    atol, rtol = manifest["replay_atol"], manifest["replay_rtol"]
    for study, (_, expected) in STUDIES.items():
        folder = HERE / "frozen" / COMP_REL / study
        selection, evaluation = read(folder / "selection_lock.json"), read(folder / "evaluation.json")
        if study == "synthetic_parity":
            pm = import_at("portable_parity_models", folder / "parity_models.py")
            adapter = None
        else:
            adapter = import_at("portable_" + study, folder / "adapter.py")
        for index, (choice, row) in enumerate(zip(selection["selections"], evaluation["rows"]), 1):
            if choice["selected_id"] is None:
                all_reports.append(dict(study=study, selected_id=None, passed=False, reason="all candidates failed"))
                continue
            cid = choice["selected_id"]
            report_path = REPORTS / "models" / study / (cid + ".json")
            binding = dict(bundle_manifest_sha256=manifest_digest, replay_source_sha256=source_digest,
                           checkpoint_sha256=choice["checkpoint_sha256"], execution_environment_sha256=environment_digest)
            if report_path.exists():
                previous = read(report_path)
                assert previous["binding"] == binding, "Stale per-model replay report"
                assert previous["complete"]
                for item in previous["cpu_prediction_files"]:
                    assert sha(safe_relative(REPORTS, item["path"])) == item["sha256"]
                all_reports.append(previous)
                reused += 1
                continue
            candidate = read(folder / "candidates" / (cid + ".json"))
            checkpoint = torch.load(safe_relative(folder, candidate["checkpoint"]), map_location="cpu", weights_only=True)
            assert checkpoint["spec"] == candidate["spec"]
            assert checkpoint["implementation_lock_sha256"] == sha(folder / "implementation_lock.json")
            spec = candidate["spec"]
            model = pm.build_model(spec["seed"], spec["head"], spec["n"]) if adapter is None else adapter.build(spec)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            assert sum(p.numel() for p in model.parameters()) == candidate["total_parameters"]
            before, combination_buffers = check_model_state(model, checkpoint, torch)
            split_reports, prediction_files = [], []
            with torch.inference_mode():
                if adapter is None:
                    n = spec["n"]
                    x = (torch.arange(n)[None, :] < torch.arange(n + 1)[:, None]).float()
                    restored = model(x).double().numpy()
                    with np.load(safe_relative(folder, row["prediction_file"]), allow_pickle=False) as saved:
                        original = saved["canonical_logits"].copy()
                    canonical_comparison = compare_predictions("canonical", original, restored, atol, rtol, True)
                    counts = np.arange(n + 1)
                    weights = np.asarray([math.comb(n, j) / 2 ** n for j in counts])
                    changed = (restored[:, 1] > restored[:, 0]) != (original[:, 1] > original[:, 0])
                    population = classification(counts % 2, restored, weights)
                    population["success_at_099"] = population["accuracy"] >= .99
                    canonical_comparison.update(changed_binomial_probability_mass=math.fsum(weights[changed]),
                        cpu_population_metrics=population,
                        cpu_population_metric_deltas=cpu_metrics_delta(population, row["population"]),
                        population_success_changed=population["success_at_099"] != row["population"]["success_at_099"])
                    split_reports.append(canonical_comparison)
                    prediction_files.append(save_cpu_prediction(study, cid, "canonical",
                        dict(ids=counts, truth=counts % 2), restored))
                    for split in ("val", "test"):
                        data = inputs.split(study, spec, split)
                        reference_path = candidate["validation_prediction"] if split == "val" else row["prediction_file"]
                        with np.load(safe_relative(folder, reference_path), allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved["canonical_logits"], original)
                            reference = assert_prediction_rows(study, saved, data)
                            cpu = restored[data["count"]]
                            comparison = compare_predictions(split, reference, cpu, atol, rtol, True)
                            metrics = classification(data["truth"], cpu)
                            comparison.update(cpu_metrics=metrics,
                                cpu_metric_deltas=cpu_metrics_delta(metrics, classification(data["truth"], reference)))
                            split_reports.append(comparison)
                        prediction_files.append(save_cpu_prediction(study, cid, split, data, cpu))
                else:
                    for split in (["val", "test", "population"] if study == "first_cubic" else ["val", "test"]):
                        data = inputs.split(study, spec, split)
                        if study == "first_cubic":
                            x = torch.eye(12).expand(len(data["ids"]), -1, -1)
                            values = (x, torch.as_tensor(data["mask"], dtype=torch.bool))
                        else:
                            values = (torch.as_tensor(data["x"], dtype=torch.float32),)
                        chunks = [model(*(v[start:start + 256] for v in values)).double().numpy()
                                  for start in range(0, len(data["ids"]), 256)]
                        restored = np.concatenate(chunks)
                        reference_path = candidate["validation_prediction"] if split == "val" else row[split + "_prediction_file"]
                        with np.load(safe_relative(folder, reference_path), allow_pickle=False) as saved:
                            original = assert_prediction_rows(study, saved, data).copy()
                        comparison = compare_predictions(split, original, restored, atol, rtol,
                                                         study == "hierarchical_sequence")
                        comparison["cpu_metrics"] = measured(study, data["truth"], restored)
                        comparison["cpu_metric_deltas"] = cpu_metrics_delta(comparison["cpu_metrics"],
                            measured(study, data["truth"], original))
                        split_reports.append(comparison)
                        prediction_files.append(save_cpu_prediction(study, cid, split, data, restored))
            after = dict(list(model.named_parameters()) + list(model.named_buffers()))
            assert before.keys() == after.keys()
            assert all(torch.equal(before[name], after[name]) for name in before)
            assert not torch.cuda.is_initialized()
            result = dict(study=study, selected_id=cid, binding=binding, checked_utc=now(),
                          splits=split_reports, passed=all(r["within_tolerance"] for r in split_reports),
                          complete=True, cpu_prediction_files=prediction_files,
                          exact_parameter_and_buffer_state_unchanged=True,
                          state_tensor_count=len(before), combination_buffers_verified=combination_buffers)
            write(report_path, result)
            all_reports.append(result)
            executed += 1
            del model
            if index % 20 == 0 or index == expected:
                print(json.dumps(dict(stage="cpu_replay", study=study, completed=index, total=expected,
                    failures=sum(not r["passed"] for r in all_reports), cuda_initialized=torch.cuda.is_initialized())), flush=True)
        assert len(selection["selections"]) == expected
    assert len(all_reports) == 780 and not torch.cuda.is_initialized()
    return dict(passed=all(r["passed"] for r in all_reports), selected_models=len(all_reports),
        failed_comparisons=[dict(study=r["study"], selected_id=r["selected_id"]) for r in all_reports if not r["passed"]],
        cuda_initialized=False, cpu_threads=torch.get_num_threads(), executed_now=executed,
        reused_bound_reports=reused, execution_environment=environment,
        execution_environment_sha256=environment_digest,
        maximum_absolute_prediction_difference=max(s["maximum_absolute_difference"] for r in all_reports for s in r.get("splits", [])),
        classification_row_decision_changes=sum(s.get("decision_changes", 0) for r in all_reports for s in r.get("splits", []) if s["split"] != "canonical"),
        parity_count_decision_changes=sum(s.get("decision_changes", 0) for r in all_reports for s in r.get("splits", []) if s["split"] == "canonical"),
        maximum_changed_parity_probability_mass=max(s.get("changed_binomial_probability_mass", 0.) for r in all_reports for s in r.get("splits", [])),
        changed_parity_success_labels=sum(s.get("population_success_changed", False) for r in all_reports for s in r.get("splits", [])),
        qualification="CPU comparison of existing selected predictors, not new training or held-out efficacy evidence.")


def compare_predictions(split, original, restored, atol, rtol, is_classification):
    import numpy as np
    assert original.shape == restored.shape and np.isfinite(original).all() and np.isfinite(restored).all()
    difference = np.abs(restored - original)
    allowed = atol + rtol * np.abs(original)
    result = dict(split=split, scalar_outputs=original.size,
                  maximum_absolute_difference=float(difference.max()),
                  maximum_tolerance_ratio=float(np.max(difference / allowed)),
                  scalar_exceedances=int(np.sum(difference > allowed)),
                  within_tolerance=bool(np.all(difference <= allowed)))
    if is_classification:
        result["decision_changes"] = int(np.sum((restored[:, 1] > restored[:, 0]) != (original[:, 1] > original[:, 0])))
        result["maximum_logit_difference_change"] = float(np.max(np.abs(
            (restored[:, 1] - restored[:, 0]) - (original[:, 1] - original[:, 0]))))
    return result


def main():
    global REPORTS
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("records", "replay"))
    parser.add_argument("--output", required=True, help="New output directory outside the distributed candidate")
    args = parser.parse_args()
    REPORTS = Path(args.output).resolve()
    if within(REPORTS, CANDIDATE_ROOT.resolve()):
        raise ValueError("Output must be outside the distributed candidate")
    if REPORTS.exists():
        raise ValueError("Use a new output directory; retained records are never overwritten")
    REPORTS.mkdir(parents=True)
    manifest = read(HERE / "bundle_manifest.json")
    digest = sha(HERE / "bundle_manifest.json")
    deny_original_workspace(manifest)
    report = dict(started_utc=now(), mode=args.mode, bundle_root=str(HERE),
                  working_directory=os.getcwd(), bundle_manifest_sha256=digest,
                  passed=False, original_workspace_fallbacks=BLOCKED_READS,
                  public_entrypoint_sha256=sha(Path(__file__)),
                  prediction_paths_relative_to="external output directory",
                  public_source_scope="Only the named editorial body is excluded from original source-inventory checks; no numerical or scientific source check is waived")
    try:
        import importlib.metadata
        report["runtime"] = dict(python=sys.version, platform=platform.platform(),
            isolated=bool(sys.flags.isolated), packages={name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "scipy", "scikit-learn", "matplotlib")})
        assert sys.flags.isolated and sys.dont_write_bytecode, "Run with python -I -B"
        report["bundle_integrity"] = verify_bundle(manifest)
        report["records"] = records(manifest)
        if args.mode == "replay":
            report["cpu_replay"] = cpu_replay(manifest, digest)
        report["bundle_integrity_after_execution"] = verify_bundle(manifest)
        report["scientific_source_compilation_records"] = [dict(path=p, sha256=sha(HERE / p)) for p in sorted(SCIENTIFIC_COMPILES)]
        report["relocation_scope"] = "No original-workspace fallback observed through the Python audit hook; this is not an OS sandbox or an independent environment installation."
        report["passed"] = report["records"]["passed"] and report.get("cpu_replay", {"passed": True})["passed"]
        assert not BLOCKED_READS
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = now()
        write(REPORTS / (args.mode + "_audit.json"), report)
        print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
