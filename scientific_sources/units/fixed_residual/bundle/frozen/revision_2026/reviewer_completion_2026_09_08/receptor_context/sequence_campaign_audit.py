"""Exercise the real candidate lifecycle on disposable CPU numerical records."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import sequence_campaign as campaign
import sequence_execution as execution
import sequence_selection as selection
from sequence_engine_audit import make_toy, equal_tree

HERE = Path(__file__).resolve().parent


class SimulatedInterruption(BaseException):
    pass


def run():
    execution.configure("cpu")
    original_folder, original_finish = campaign.HERE, execution.finish_epoch
    temporary_root = HERE/"tmp"
    temporary_root.mkdir(exist_ok=True)
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix="campaign_audit_", dir=temporary_root) as directory:
            folder = Path(directory).resolve()
            assert folder.parent == temporary_root.resolve() and HERE.resolve() in folder.parents
            store, data = make_toy(folder)
            mean, sd = execution.scalers(data)
            plan = next(row for row in selection.candidate_plan() if
                (row["setting"], row["head"], row["seed"], row["lr_index"]) == ("ligand_sequence", "lma2", 42, 0))
            fixture_lock = "0"*64
            def fit():
                return campaign.fit_candidate(plan, data, data, store, mean, sd, fixture_lock,
                                              device="cpu", epoch_limit=2)
            full_folder = folder/"uninterrupted"
            campaign.HERE = full_folder
            full = fit()
            assert full["status"] == "valid" and full["epochs_completed"] == 2
            checkpoint_relative = Path("sequence_checkpoints")/(plan["id"]+".pt")
            prediction_relative = Path("sequence_validation_predictions")/(plan["id"]+".npz")
            full_state = torch.load(full_folder/checkpoint_relative, map_location="cpu", weights_only=True)
            with np.load(full_folder/prediction_relative, allow_pickle=False) as content:
                full_prediction = content["prediction"].copy()
                full_truth = content["truth"].copy()
            assert abs(float(np.sqrt(np.mean((full_prediction-full_truth)**2)))-full["best_validation_rmse"]) < 1e-12
            checks.append("real candidate completion, retained state/predictions and independent validation RMSE")
            def forbidden(*args, **kwargs):
                raise AssertionError("A completed candidate was fitted again")
            execution.finish_epoch = forbidden
            assert fit() == full
            assert len(list((full_folder/"sequence_attempts"/plan["id"]).glob("*.json"))) == 1
            checks.append("completed candidate is preserved without fitting or a new attempt")
            bad = copy.deepcopy(full)
            bad["artifacts"][0]["sha256"] = "f"*64
            try:
                campaign.verify_candidate(bad, plan, fixture_lock)
            except ValueError:
                pass
            else:
                raise AssertionError("A changed artifact hash was accepted")
            checks.append("artifact-hash mismatch rejected")
            resumed_folder = folder/"interrupted"
            campaign.HERE = resumed_folder
            count = 0
            def interrupted(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise SimulatedInterruption()
                return original_finish(*args, **kwargs)
            execution.finish_epoch = interrupted
            try:
                fit()
            except SimulatedInterruption:
                pass
            else:
                raise AssertionError("The interruption fixture did not stop")
            attempt_folder = resumed_folder/"sequence_attempts"/plan["id"]
            first_attempt = campaign.read(attempt_folder/"attempt_001.json")
            assert "finished_utc" not in first_attempt
            assert not (resumed_folder/"sequence_candidates"/(plan["id"]+".json")).exists()
            saved = torch.load(resumed_folder/"sequence_continuation"/(plan["id"]+".pt"), map_location="cpu", weights_only=True)
            assert saved["bookkeeping"]["epochs_completed"] == 1
            execution.finish_epoch = original_finish
            resumed = fit()
            assert resumed["status"] == "valid" and resumed["epochs_completed"] == 2
            equal_tree(full_state, torch.load(resumed_folder/checkpoint_relative, map_location="cpu", weights_only=True))
            with np.load(resumed_folder/prediction_relative, allow_pickle=False) as content:
                np.testing.assert_array_equal(full_prediction, content["prediction"])
            resumed_attempt = campaign.read(attempt_folder/"attempt_002.json")
            assert resumed_attempt["resumed_from_epoch"] == 1 and resumed_attempt["previous_attempts"][0]["finished"] is False
            assert campaign.read(attempt_folder/"attempt_001.json") == first_attempt
            checks.append("interruption preserves unfinished attempt and resumes the real epoch snapshot exactly")
            campaign.HERE = folder/"numerical_failure"
            def numerical_failure(*args, **kwargs):
                raise FloatingPointError("Disposable numerical-failure fixture")
            execution.finish_epoch = numerical_failure
            failed = fit()
            assert failed["status"] == "failed" and failed["best_validation_rmse"] is None and failed["artifacts"] == []
            execution.finish_epoch = forbidden
            assert fit() == failed
            assert len(list((campaign.HERE/"sequence_attempts"/plan["id"]).glob("*.json"))) == 1
            checks.append("numerical failure remains failed and receives no fresh initialization")
            campaign.HERE = folder/"execution_error"
            def execution_error(*args, **kwargs):
                raise RuntimeError("Disposable implementation-error fixture")
            execution.finish_epoch = execution_error
            try:
                fit()
            except RuntimeError:
                pass
            else:
                raise AssertionError("An implementation error was silently consumed")
            attempt = campaign.read(campaign.HERE/"sequence_attempts"/plan["id"]/"attempt_001.json")
            assert attempt["status"] == "execution_error"
            assert not (campaign.HERE/"sequence_candidates"/(plan["id"]+".json")).exists()
            checks.append("unexpected implementation error aborts and is separately recorded")
            path = folder/"immutable.json"
            campaign.write_json(path, {"value":1}, immutable=True)
            try:
                campaign.write_json(path, {"value":2}, immutable=True)
            except FileExistsError:
                pass
            else:
                raise AssertionError("An immutable result was overwritten")
            assert campaign.read(path) == {"value":1}
            checks.append("immutable completed record cannot be overwritten")
            with campaign.exclusive_process(folder/"mutex"):
                try:
                    with campaign.exclusive_process(folder/"mutex"):
                        raise AssertionError("Concurrent fitting lock was acquired")
                except OSError:
                    pass
            with campaign.exclusive_process(folder/"mutex"):
                pass
            checks.append("exclusive fitting lock rejects a second holder and releases on exit")
    finally:
        campaign.HERE, execution.finish_epoch = original_folder, original_finish
    source_names = ("sequence_campaign.py", "sequence_campaign_audit.py", "sequence_execution.py",
                    "sequence_selection.py", "sequence_engine_cpu_audit.json", "sequence_selection_cpu_audit.json")
    output = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
        sources=[dict(path=str(HERE/name), sha256=campaign.sha(HERE/name)) for name in source_names],
        qualification="Disposable seven-record CPU fixture, two epochs, one joint order-two model; no retained affinity candidate, prediction or test evaluation. It checks orchestration around the independently audited eleven-model update engine. GPU feasibility, retained source lock, final scoring and profiling remain separate.")
    campaign.write_json(HERE/"sequence_campaign_cpu_audit.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    run()
