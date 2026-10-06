"""Reconcile all saved candidates, choices, predictions and the frozen package."""
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import models
import study

HERE = study.HERE


def independent_metrics(y, p):
    n = len(y)
    rmse = math.sqrt(math.fsum((float(a)-float(b))**2 for a, b in zip(y, p))/n)
    mae = math.fsum(abs(float(a)-float(b)) for a, b in zip(y, p))/n
    ym = math.fsum(map(float, y))/n
    pm = math.fsum(map(float, p))/n
    cross = math.fsum((float(a)-ym)*(float(b)-pm) for a, b in zip(y, p))
    sy = math.fsum((float(a)-ym)**2 for a in y)
    sp = math.fsum((float(b)-pm)**2 for b in p)
    return dict(rmse=rmse, mae=mae, pearson=cross/math.sqrt(sy*sp) if sp>0 and sy>0 else None)


def main():
    study.configure()
    lock = study.verify_lock()
    choices = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    evaluation = json.loads((HERE/"evaluation.json").read_text(encoding="utf-8"))
    profiles = json.loads((HERE/"profiles.json").read_text(encoding="utf-8"))
    assert profiles["complete"] and len(profiles["rows"]) == 17
    assert evaluation["selection_lock_sha256"] == study.sha(HERE/"selection_lock.json")
    assert len(choices["candidate_records"]) == len(lock["candidates"]) == 170
    assert len(choices["selections"]) == len(evaluation["rows"]) == 85
    train = torch.load(study.DATA/"pdbbind_train.pt", weights_only=True, map_location="cpu")
    val = torch.load(study.DATA/"pdbbind_val.pt", weights_only=True, map_location="cpu")
    test = study.load_split("test")
    test_ids, test_y = np.asarray(test["ids"]), test["y"].double().cpu().numpy()
    assert not set(train["ids"]) & set(test["ids"])
    assert not set(val["ids"]) & set(test["ids"])
    mean = float(train["y"].double().mean())
    sd = float(train["y"].double().std(unbiased=False))
    records = {}
    max_validation_delta = 0.
    for item in choices["candidate_records"]:
        path = HERE/item["path"]
        assert study.sha(path) == item["sha256"]
        row = json.loads(path.read_text(encoding="utf-8"))
        records[row["spec"]["id"]] = row
        assert abs(row["target_mean"]-mean) < 1e-13 and abs(row["target_sd"]-sd) < 1e-13
        assert row["implementation_lock_sha256"] == study.sha(HERE/"implementation_lock.json")
        assert datetime.fromisoformat(row["started_utc"]) >= datetime.fromisoformat(lock["created_utc"])
        assert datetime.fromisoformat(row["finished_utc"]) <= datetime.fromisoformat(choices["locked_utc"])
        if row["status"] != "valid":
            continue
        assert study.sha(HERE/row["checkpoint"]) == row["checkpoint_sha256"]
        assert study.sha(HERE/row["validation_predictions"]) == row["validation_prediction_sha256"]
        archive = np.load(HERE/row["validation_predictions"])
        assert np.array_equal(archive["ids"], np.asarray(val["ids"]))
        assert np.array_equal(archive["truth"], val["y"].double().numpy())
        actual = independent_metrics(archive["truth"], archive["prediction"])["rmse"]
        max_validation_delta = max(max_validation_delta, abs(actual-row["best_validation_rmse"]))
        best_check = min((h for h in row["history"] if "validation_rmse" in h),
                         key=lambda h: (h["validation_rmse"], h["epoch"]))
        assert best_check["epoch"] == row["best_epoch"]
        assert best_check["validation_rmse"] == row["best_validation_rmse"]
        assert row["epochs_run"] == len(row["history"])
        assert [h["epoch"] for h in row["history"]] == list(range(1, row["epochs_run"]+1))
    reload_rows = []
    max_test_delta = 0.
    for choice, result in zip(choices["selections"], evaluation["rows"]):
        candidates = [records[i] for i in choice["candidate_ids"] if records[i]["status"]=="valid"]
        best = min(candidates, key=lambda r: (r["best_validation_rmse"], r["spec"]["lr"])) if candidates else None
        assert choice["selected_id"] == (best["spec"]["id"] if best else None)
        if best is None:
            assert result["status"] == "all_candidates_failed"
            continue
        assert result["selected_id"] == best["spec"]["id"]
        archive_path = HERE/result["prediction_file"]
        assert study.sha(archive_path) == result["prediction_sha256"]
        archive = np.load(archive_path)
        assert np.array_equal(archive["ids"], test_ids)
        assert np.array_equal(archive["truth"], test_y)
        measured = independent_metrics(test_y, archive["prediction"])
        for key in ("rmse", "mae", "pearson"):
            if measured[key] is None:
                assert result["test"][key] is None
            else:
                max_test_delta = max(max_test_delta, abs(measured[key]-result["test"][key]))
        checkpoint = torch.load(HERE/best["checkpoint"], weights_only=True, map_location="cpu")
        model = models.build_model(choice["seed"], choice["head"], choice["depth"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        p = study.predict(model, test, checkpoint["target_mean"], checkpoint["target_sd"])
        delta = float(np.max(np.abs(p-archive["prediction"])))
        assert delta < 1e-10
        reload_rows.append(dict(id=choice["selected_id"], maximum_prediction_delta=delta))
        del model
    assert max_validation_delta < 1e-12 and max_test_delta < 1e-12
    manifest_path = HERE.parent/"manuscript_application"/"output"/"application_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = HERE.parent.parent
    modified = []
    for entry in old_manifest["files"]:
        if study.sha(workspace/entry["path"]) != entry["sha256"]:
            modified.append(entry["path"])
    assert not modified, modified
    # The manifest itself is not one of its entries; its prior recorded digest
    # provides the additional frozen-package check.
    prior_manifest_sha = "b399fb830c9b836d13bd4909417b14114d97e34fa3da5723d90b986704785239"
    assert study.sha(manifest_path) == prior_manifest_sha
    output = dict(audited_utc=study.now(), passed=True, candidates=len(records),
        valid_candidates=sum(r["status"]=="valid" for r in records.values()),
        failed_candidates=sum(r["status"]!="valid" for r in records.values()),
        selected=len(choices["selections"]), valid_selected=len(reload_rows),
        max_independent_validation_metric_delta=max_validation_delta,
        max_independent_test_metric_delta=max_test_delta,
        selected_checkpoint_reload=reload_rows,
        previous_package_files_verified=len(old_manifest["files"])+1, previous_package_modified=modified,
        all_test_choices_reconciled=True, all_validation_checkpoint_choices_reconciled=True,
        scope="Numerical/bookkeeping integrity, not global optimization, independent data replication or theorem transfer")
    study.write_json(HERE/"final_audit.json", output)
    print(json.dumps({k:v for k,v in output.items() if k!="selected_checkpoint_reload"}), flush=True)


if __name__ == "__main__":
    main()
