"""Disposable checks of selected-predictor reconciliation and full reporting."""
import copy
import math
from pathlib import Path
import tempfile
import numpy as np
import torch
import sequence_campaign as campaign
import sequence_execution as execution
import sequence_models as models
import sequence_scoring as scoring
import sequence_selection as selection
from sequence_engine_audit import make_toy, equal_tree

HERE = Path(__file__).resolve().parent


def rejects(function):
    try:
        function()
    except (ValueError, KeyError):
        return
    raise AssertionError("An invalid scoring fixture was accepted")


def toy_groups(ids):
    full = set(ids)
    absence = dict(canonical_ligand=set(ids[:5]), complete_stored_string=set(ids[1:]),
                   any_supplied_chain=set(ids[2:6]), both_ligand_and_any_chain=set(ids[2:5]))
    groups = dict(full=full)
    for name, value in absence.items():
        groups[name+"_absent_from_train_and_val"] = value
        opposite = "either_ligand_or_any_supplied_chain" if name == "both_ligand_and_any_chain" else name
        groups[opposite+"_overlap_with_train_or_val"] = full-value
    return dict(groups={name: sorted(value) for name, value in groups.items()},
                subset_sizes={name: len(value) for name, value in groups.items()})


def run():
    execution.configure("cpu")
    checks = []
    y = np.asarray([-2., -1., 0., 1., 2.])
    p = np.asarray([-1., 0., 0., 0., 1.])
    observed = scoring.independent_metric(y, p)
    np.testing.assert_allclose([observed["rmse"], observed["mae"], observed["pearson"]],
                               [math.sqrt(.8), .8, 4/math.sqrt(20)], rtol=1e-14, atol=1e-14)
    scoring.metric_delta(observed, execution.metric(y, p))
    assert scoring.independent_metric(y, np.zeros(5))["pearson"] is None
    assert scoring.independent_metric([], []) == dict(n=0, rmse=None, mae=None, pearson=None)
    rejects(lambda: scoring.arrays(["x", "x"], [1, 2], [1, 2]))
    rejects(lambda: scoring.arrays(["x"], [float("nan")], [1]))
    rejects(lambda: scoring.arrays(["x"], [1], [float("inf")]))
    rejects(lambda: scoring.arrays(["x"], [1, 2], [1, 2]))
    rejects(lambda: scoring.metric_delta(observed, observed | dict(rmse=observed["rmse"]+.01)))
    checks.append("independent analytic RMSE/MAE/correlation, undefined correlation, empty subsets and malformed arrays")

    ids = [f"toy{i}" for i in range(7)]
    metadata = toy_groups(ids)
    groups = scoring.validate_groups(metadata, ids)
    bad_groups = copy.deepcopy(metadata)
    bad_groups["groups"]["canonical_ligand_overlap_with_train_or_val"] = ids[:2]
    rejects(lambda: scoring.validate_groups(bad_groups, ids))
    checks.append("all nine subsets, exact complements and joint-absence intersection")
    config_index = {selection.key(*pair): i for i, pair in enumerate(models.configurations())}
    manufactured = [row | dict(status="valid", best_epoch=1,
        best_validation_rmse=1+config_index[selection.key(row["setting"], row["head"])]/10+row["lr_index"]/100)
        for row in selection.candidate_plan()]
    locked = selection.select(manufactured)
    toy_truth = np.arange(-3., 4.)
    rows = []
    known_rmse = {}
    for choice in locked["selections"]:
        index = config_index[choice["configuration"]]
        # Exactly represented constant errors; the validation nominee has the worst test errors.
        offset = (11-index)**2/8+(choice["seed"]-42)/64
        prediction = toy_truth+offset
        known_rmse[(choice["configuration"], choice["seed"])] = offset
        metrics = scoring.group_metrics(ids, toy_truth, prediction, groups)
        for name, members in groups.items():
            mask = np.asarray([key in members for key in ids])
            scoring.metric_delta(metrics[name], scoring.independent_metric(toy_truth[mask], prediction[mask]))
        rows.append(choice | dict(subsets=metrics))
    summary = scoring.summarize(rows, locked, groups)
    assert len(summary["procedures"]) == 99 and len(summary["paired_contrasts"]) == 171
    assert len(summary["paired_seed_differences"]) == 855
    assert summary["nominated_procedure"] == locked["nominated_procedure"] == selection.key(*models.configurations()[0])
    assert scoring.summarize(rows[::-1], locked, groups) == summary
    for item in summary["procedures"]:
        independent = [known_rmse[(item["configuration"], seed)] for seed in selection.SEEDS]
        np.testing.assert_allclose([item["metrics"]["rmse"]["mean"], item["metrics"]["rmse"]["sample_sd"]],
                                   [np.mean(independent), np.std(independent, ddof=1)], rtol=1e-13, atol=1e-13)
    contrast_by_id = {row["id"]: row for row in locked["contrasts"]}
    for item in summary["paired_contrasts"]:
        terms = contrast_by_id[item["contrast"]]["terms"]
        expected = [sum(term["weight"]*known_rmse[(term["configuration"], seed)] for term in terms) for seed in selection.SEEDS]
        np.testing.assert_allclose([item["difference"]["mean"], item["difference"]["sample_sd"]],
                                   [np.mean(expected), np.std(expected, ddof=1)], rtol=1e-13, atol=1e-13)
        assert (item["negative"], item["zero"], item["positive"]) == tuple(sum(v < 0 for v in expected) if sign == -1 else
            sum(v == 0 for v in expected) if sign == 0 else sum(v > 0 for v in expected) for sign in (-1, 0, 1))
    checks.append("all 55 choices, 99 procedure/subset rows and 171 signed contrasts reconcile with independent known errors")
    checks.append("test results cannot change validation nomination; row ordering cannot change summaries")
    rejects(lambda: scoring.summarize(rows[:-1], locked, groups))
    rejects(lambda: scoring.summarize(rows[:-1]+[rows[0]], locked, groups))
    altered = copy.deepcopy(rows)
    altered[0]["subsets"]["full"]["n"] -= 1
    rejects(lambda: scoring.summarize(altered, locked, groups))
    failed_config = selection.key("ligand_sequence", "lma2")
    failure_records = [row | dict(status="failed") if (selection.key(row["setting"], row["head"]), row["seed"]) == (failed_config, 42) else row
                       for row in manufactured]
    failure_lock = selection.select(failure_records)
    failure_choices = {(row["configuration"], row["seed"]): row for row in failure_lock["selections"]}
    failure_rows = [failure_choices[(row["configuration"], row["seed"])] | dict(subsets=None)
                    if (row["configuration"], row["seed"]) == (failed_config, 42) else row for row in rows]
    failed_summary = scoring.summarize(failure_rows, failure_lock, groups)
    assert len(failed_summary["procedures"]) == 99 and len(failed_summary["paired_seed_differences"]) == 855
    for row in failed_summary["procedures"]:
        if row["configuration"] == failed_config:
            assert row["valid_seeds"] == 4 and row["failed_seeds"] == 1 and row["metrics"]["rmse"]["missing"] == 1
    for row in failed_summary["paired_contrasts"]:
        if any(term["configuration"] == failed_config for term in row["terms"]):
            assert row["difference"]["n"] == 4 and row["difference"]["missing"] == 1
    checks.append("missing/duplicate outcomes rejected; one failed seed remains in all affected procedure and pair denominators")

    original_campaign, original_scoring = campaign.HERE, scoring.HERE
    original_context, original_start, original_load = scoring.selection_context, scoring.gpu_start, campaign.load_split
    temporary_root = HERE/"tmp"
    temporary_root.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="scoring_audit_", dir=temporary_root) as directory:
            folder = Path(directory).resolve()
            assert folder.parent == temporary_root.resolve() and HERE.resolve() in folder.parents
            campaign.HERE = scoring.HERE = folder
            store, data = make_toy(folder)
            mean, sd = execution.scalers(data)
            plan = next(row for row in selection.candidate_plan() if
                (row["setting"], row["head"], row["seed"], row["lr_index"]) == ("ligand_sequence", "lma2", 42, 0))
            record = campaign.fit_candidate(plan, data, data, store, mean, sd, "0"*64, device="cpu", epoch_limit=2)
            choice = dict(configuration=selection.key(plan["setting"], plan["head"]),
                setting=plan["setting"], head=plan["head"], seed=plan["seed"], status="valid", selected_id=plan["id"],
                validation_rmse=record["best_validation_rmse"], best_epoch=record["best_epoch"], learning_rate=plan["lr"])
            actual = scoring.score_selected(choice, record, data, data, store, groups, device="cpu")
            reloaded = scoring.score_selected(choice, record, data, data, store, groups, device="cpu", retain=False)
            assert reloaded == actual
            checkpoint = folder/"sequence_checkpoints"/(plan["id"]+".pt")
            model = models.build_model(42, "ligand_sequence", "lma2")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            arguments, _ = execution.collate(model, data, store, data["ids"])
            with torch.no_grad():
                manual = model(**arguments).double().numpy()*sd+mean
            saved = scoring.read_predictions(folder/actual["test_prediction_file"], data)
            np.testing.assert_array_equal(manual, saved[2])
            checks.append("real two-epoch candidate, fresh selected checkpoint, manual inverse scaling and independently reloaded prediction files")
            rejects(lambda: scoring.score_selected(choice | dict(validation_rmse=choice["validation_rmse"]+.1),
                record, data, data, store, groups, device="cpu", retain=False))
            prediction_path = folder/actual["test_prediction_file"]
            preserved = prediction_path.read_bytes()
            np.savez_compressed(prediction_path, ids=saved[0], truth=saved[1], prediction=saved[2]+1)
            rejects(lambda: scoring.score_selected(choice, record, data, data, store, groups, device="cpu", retain=False))
            prediction_path.write_bytes(preserved)
            checks.append("changed validation score and changed retained prediction values rejected")
            runtime = execution.make_runtime(42, "ligand_sequence", "lma2", .001)
            runtime["model"].load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            expected_model = execution.to_cpu(runtime["model"].state_dict())
            expected_optimizer = execution.to_cpu(runtime["optimizer"].state_dict())
            checkpoint_sha = campaign.sha(checkpoint)
            for mode, batch in (("forward", 1), ("forward", 7), ("training_step", 1), ("training_step", 7)):
                measured = scoring.profile_workload(runtime, data, store, mean, sd, batch, mode, repeats=2)
                assert measured["checkpoint_restored"] and len(measured["seconds"]) == 2 and measured["median_seconds"] > 0
                equal_tree(expected_model, runtime["model"].state_dict())
                equal_tree(expected_optimizer, runtime["optimizer"].state_dict())
            assert campaign.sha(checkpoint) == checkpoint_sha
            checks.append("forward/training cost paths restore model and optimizer state and preserve checkpoint files")
            def incomplete_selection():
                raise ValueError("Disposable incomplete-selection guard")
            def forbidden(*args, **kwargs):
                raise AssertionError("GPU setup or split loading occurred before the complete selection guard")
            scoring.selection_context, scoring.gpu_start, campaign.load_split = incomplete_selection, forbidden, forbidden
            rejects(scoring.evaluate)
            checks.append("evaluation cannot load test data or initialize GPU work when the complete-selection guard fails")
    finally:
        campaign.HERE, scoring.HERE = original_campaign, original_scoring
        scoring.selection_context, scoring.gpu_start, campaign.load_split = original_context, original_start, original_load
    names = ("sequence_scoring.py", "sequence_scoring_audit.py", "sequence_campaign.py", "sequence_selection.py",
             "sequence_execution.py", "sequence_models.py", "sequence_data.py", "sequence_engine_audit.py",
             "sequence_campaign_cpu_audit.json", "sequence_selection_cpu_audit.json")
    result = dict(passed=True, completed_utc=campaign.utc(), checks=checks, configurations=11, choices=55,
        procedure_subset_rows=99, contrasts_by_subset=171, paired_seed_rows=855,
        sources=[dict(path=str(HERE/name), sha256=campaign.sha(HERE/name)) for name in names],
        qualification="Manufactured metrics/selection records plus a disposable seven-record, two-epoch CPU candidate. No retained receptor-affinity fitting or test evaluation and no GPU computation. Actual GPU feasibility and final retained checkpoint audits remain separate.")
    campaign.write_json(HERE/"sequence_scoring_cpu_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
