"""Auditable fixed-protocol fitting for the two ordinary-inference synthetic studies."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"synthetic_parity"))
import parity_common as u


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_adapter(name):
    path = HERE/name/"adapter.py"
    spec = importlib.util.spec_from_file_location("finite_adapter_"+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def specifications(a):
    rows = [dict(id=f"{task}_{head}_seed{seed}_lr{j}", task=task, head=head,
                 seed=seed, lr=lr, lr_index=j)
            for task in a.TASKS for head in a.HEADS for seed in a.SEEDS
            for j, lr in enumerate(a.LRS)]
    return [rows[int(i)] for i in np.random.default_rng(a.ORDER_SEED).permutation(len(rows))]


def sources(a):
    return list(dict.fromkeys([Path(__file__).resolve(), Path(u.__file__).resolve()]+a.sources()))


def metrics(a, y, pred):
    y, pred = np.asarray(y), np.asarray(pred, dtype=np.float64)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite prediction")
    if a.KIND == "classification":
        return u.metrics(y, pred)
    assert pred.shape == y.shape
    delta = pred-y.astype(np.float64)
    return dict(mse=float(np.mean(delta**2)), mae=float(np.mean(np.abs(delta))))


def criterion(a, result):
    return result["cross_entropy" if a.KIND == "classification" else "mse"]


def truth(data):
    return data["truth"] if "truth" in data else data["y"].detach().cpu().numpy()


def loss_value(a, pred, y):
    return nn.functional.cross_entropy(pred, y) if a.KIND == "classification" else nn.functional.mse_loss(pred, y)


@torch.no_grad()
def predict(a, model, data):
    model.eval()
    out = []
    for start in range(0, len(data["y"]), a.BATCH):
        values = tuple(t[start:start+a.BATCH] for t in data["inputs"])
        pred = model(*values)
        expected = (len(values[0]), 2) if a.KIND == "classification" else (len(values[0]),)
        if pred.shape != expected or not bool(torch.isfinite(pred).all()):
            raise FloatingPointError("Nonfinite or malformed ordinary-batch prediction")
        out.append(pred.double().cpu().numpy())
    return np.concatenate(out)


def lock(a):
    assert not (a.HERE/"implementation_lock.json").exists()
    cpu, gpu = read(a.HERE/"cpu_audit.json"), read(a.HERE/"prefit_audit.json")
    assert cpu["passed"] and gpu["passed"]
    assert gpu["cpu_audit_sha256"] == u.sha(a.HERE/"cpu_audit.json")
    for item in cpu["sources"]:
        assert u.sha(item["path"]) == item["sha256"], item["path"]
    specs = specifications(a)
    files = sources(a)+a.lock_files()+[a.HERE/"cpu_audit.json", a.HERE/"prefit_audit.json"]
    u.write_json(a.HERE/"implementation_lock.json", dict(locked_utc=u.now(),
        candidates=specs, candidate_count=len(specs), selection_count=len(specs)//len(a.LRS),
        kind=a.KIND, tasks=a.TASKS, heads=a.HEADS, seeds=a.SEEDS, learning_rates=a.LRS,
        epochs=a.EPOCHS, batch_size=a.BATCH, patience_checks=a.PATIENCE, clip_norm=a.CLIP,
        optimizer="AdamW weight_decay=0.0001; cosine horizon=100", sizes=a.SIZES,
        candidate_permutation_seed=a.ORDER_SEED,
        minibatch_seed="CPU torch.Generator 20000+seed", stochastic_seed="torch.manual_seed(30000+seed) after construction",
        inference="Ordinary inputs in fixed row-order chunks of 256; eval mode; no canonical count substitution",
        selection="Earliest strict validation loss minimum; lower numerical learning rate on exact tie",
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                     platform=platform.platform(), gpu=torch.cuda.get_device_name(), cpu_threads=4, tf32=False),
        files=[dict(path=str(p), sha256=u.sha(p), bytes=p.stat().st_size) for p in dict.fromkeys(files)]))
    print(json.dumps(dict(stage="implementation_locked", study=a.HERE.name,
                          candidates=len(specs), selections=len(specs)//len(a.LRS))), flush=True)


def verify(a):
    r = read(a.HERE/"implementation_lock.json")
    for item in r["files"]:
        assert u.sha(item["path"]) == item["sha256"], item["path"]
    assert r["candidates"] == specifications(a)
    return r


def fit(a, spec, lock_sha):
    destination = a.HERE/"candidates"/(spec["id"]+".json")
    if destination.exists():
        old = read(destination)
        assert old["spec"] == spec and old["implementation_lock_sha256"] == lock_sha
        assert old["status"] in ("valid", "failed"), "Diagnose implementation error before resuming"
        return old
    attempts = a.HERE/"attempts"
    attempts.mkdir(exist_ok=True)
    attempt = 1+len(list(attempts.glob(spec["id"]+"_attempt*.json")))
    attempt_path = attempts/f'{spec["id"]}_attempt{attempt}.json'
    u.write_json(attempt_path, dict(started_utc=u.now(), spec=spec, attempt=attempt,
        note="Only an unfinished interruption may restart from the original seed; numerical failures are retained."))
    train, val = a.load(spec, "train", "cuda"), a.load(spec, "val", "cuda")
    yval = truth(val)
    model = a.build(spec).cuda()
    torch.manual_seed(30000+spec["seed"])
    gen = torch.Generator().manual_seed(20000+spec["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.EPOCHS)
    best, best_epoch, state, bad_checks = float("inf"), 0, None, 0
    history, fatal = [], None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = dict(spec=spec, attempt=attempt, started_utc=u.now(), implementation_lock_sha256=lock_sha,
                  total_parameters=sum(p.numel() for p in model.parameters()))
    try:
        for epoch in range(1, a.EPOCHS+1):
            ep_started = time.perf_counter()
            model.train()
            order = torch.randperm(len(train["y"]), generator=gen).cuda()
            total, max_norm = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
            for start in range(0, len(order), a.BATCH):
                idx = order[start:start+a.BATCH]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*(t[idx] for t in train["inputs"]))
                loss = loss_value(a, prediction, train["y"][idx])
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), a.CLIP)
                optimizer.step()
                total += loss.detach()*len(idx)
                max_norm = torch.maximum(max_norm, norm.detach())
            scheduler.step()
            if not bool(torch.isfinite(total)) or not bool(torch.isfinite(max_norm)):
                raise FloatingPointError("Nonfinite training loss/preclip gradient at epoch "+str(epoch))
            row = dict(epoch=epoch, training_loss=float(total/len(order)),
                       maximum_preclip_gradient_norm=float(max_norm))
            if epoch == 1 or epoch % 5 == 0 or epoch == a.EPOCHS:
                pred = predict(a, model, val)
                measures = metrics(a, yval, pred)
                score = criterion(a, measures)
                row.update(validation_loss=score, validation=measures)
                diag = a.diagnostics(model, val)
                if diag is not None:
                    row["validation_diagnostics"] = diag
                if score < best:
                    best, best_epoch, bad_checks = score, epoch, 0
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_checks += 1
            row["epoch_wall_seconds"] = time.perf_counter()-ep_started
            history.append(row)
            if epoch % 25 == 0:
                print(json.dumps(dict(stage="training", study=a.HERE.name, id=spec["id"], epoch=epoch,
                                      best_validation_loss=best)), flush=True)
            if bad_checks >= a.PATIENCE:
                break
        if state is None:
            raise FloatingPointError("No valid validation checkpoint")
        model.load_state_dict(state)
        pred = predict(a, model, val)
        assert abs(criterion(a, metrics(a, yval, pred))-best) <= 1e-12
        checkpoint = a.HERE/"checkpoints"/(spec["id"]+".pt")
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict=state, spec=spec, implementation_lock_sha256=lock_sha), checkpoint)
        predictions = a.HERE/"validation_predictions"/(spec["id"]+".npz")
        predictions.parent.mkdir(exist_ok=True)
        np.savez_compressed(predictions, ids=val["ids"], truth=yval, prediction=pred)
        result.update(status="valid", best_validation_loss=best, best_epoch=best_epoch,
            checkpoint=str(checkpoint.relative_to(a.HERE)), checkpoint_sha256=u.sha(checkpoint),
            validation_prediction=str(predictions.relative_to(a.HERE)), validation_prediction_sha256=u.sha(predictions))
    except Exception as exc:
        scientific = isinstance(exc, (FloatingPointError, torch.OutOfMemoryError))
        result.update(status="failed" if scientific else "implementation_error", error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(), best_validation_loss=None, best_epoch=None)
        if not scientific:
            fatal = exc
    torch.cuda.synchronize()
    result.update(finished_utc=u.now(), history=history, epochs_run=len(history), reached_epoch_cap=len(history)==a.EPOCHS,
        wall_seconds=time.perf_counter()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    u.write_json(destination, result)
    u.write_json(attempt_path, dict(finished_utc=u.now(), spec=spec, attempt=attempt, status=result["status"]))
    print(json.dumps(dict(stage="candidate_complete", study=a.HERE.name, id=spec["id"], status=result["status"],
                          best_validation_loss=result["best_validation_loss"], seconds=result["wall_seconds"])), flush=True)
    del model, optimizer, scheduler, train, val
    torch.cuda.empty_cache()
    if fatal is not None:
        raise fatal
    return result


def train(a):
    lock_record = verify(a)
    assert not (a.HERE/"selection_lock.json").exists() and not (a.HERE/"evaluation.json").exists()
    done = []
    for spec in lock_record["candidates"]:
        done.append(fit(a, spec, u.sha(a.HERE/"implementation_lock.json")))
        u.write_json(a.HERE/"progress.json", dict(updated_utc=u.now(), completed=len(done), total=len(lock_record["candidates"]),
            valid=sum(r["status"]=="valid" for r in done), failed=sum(r["status"]=="failed" for r in done),
            aggregate_fit_seconds=sum(r["wall_seconds"] for r in done), last_id=spec["id"]))
    selections = []
    for task in a.TASKS:
        for head in a.HEADS:
            for seed in a.SEEDS:
                group = [r for r in done if (r["spec"]["task"], r["spec"]["head"], r["spec"]["seed"]) == (task, head, seed)]
                assert len(group) == len(a.LRS)
                valid = [r for r in group if r["status"] == "valid"]
                chosen = min(valid, key=lambda r: (r["best_validation_loss"], r["spec"]["lr"])) if valid else None
                selections.append(dict(task=task, head=head, seed=seed,
                    selected_id=chosen["spec"]["id"] if chosen else None,
                    validation_loss=chosen["best_validation_loss"] if chosen else None,
                    checkpoint_sha256=chosen["checkpoint_sha256"] if chosen else None,
                    candidate_ids=[r["spec"]["id"] for r in group]))
    u.write_json(a.HERE/"selection_lock.json", dict(locked_utc=u.now(), selections=selections,
        implementation_lock_sha256=u.sha(a.HERE/"implementation_lock.json"),
        candidate_records=[dict(path=f'candidates/{r["spec"]["id"]}.json',
                               sha256=u.sha(a.HERE/"candidates"/(r["spec"]["id"]+".json"))) for r in done]))
    print(json.dumps(dict(stage="all_selections_locked", study=a.HERE.name, count=len(selections))), flush=True)


def verify_selection(a):
    record = verify(a)
    choices = read(a.HERE/"selection_lock.json")
    assert choices["implementation_lock_sha256"] == u.sha(a.HERE/"implementation_lock.json")
    assert len(choices["selections"]) == record["selection_count"]
    assert len(choices["candidate_records"]) == record["candidate_count"]
    for item in choices["candidate_records"]:
        assert u.sha(a.HERE/item["path"]) == item["sha256"]
    return choices


def selected_model(a, choice):
    candidate = read(a.HERE/"candidates"/(choice["selected_id"]+".json"))
    path = a.HERE/candidate["checkpoint"]
    assert u.sha(path) == choice["checkpoint_sha256"] == candidate["checkpoint_sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["spec"] == candidate["spec"]
    assert checkpoint["implementation_lock_sha256"] == u.sha(a.HERE/"implementation_lock.json")
    model = a.build(candidate["spec"]).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    return model, candidate


def evaluate(a):
    choices = verify_selection(a)
    assert not (a.HERE/"evaluation.json").exists()
    rows = []
    directory = a.HERE/"test_predictions"
    directory.mkdir(exist_ok=True)
    for choice in choices["selections"]:
        if choice["selected_id"] is None:
            rows.append(dict(**choice, status="all_candidates_failed", test=None))
            continue
        model, candidate = selected_model(a, choice)
        record = dict(**choice, status="valid", total_parameters=candidate["total_parameters"],
                      best_epoch=candidate["best_epoch"], epochs_run=candidate["epochs_run"], selected_lr=candidate["spec"]["lr"])
        for split in a.EVALUATION_SPLITS:
            data = a.load(candidate["spec"], split, "cuda")
            pred, y = predict(a, model, data), truth(data)
            path = directory/f'{choice["selected_id"]}_{split}.npz'
            np.savez_compressed(path, ids=data["ids"], truth=y, prediction=pred)
            record[split] = metrics(a, y, pred)
            record[split+"_prediction_file"] = str(path.relative_to(a.HERE))
            record[split+"_prediction_sha256"] = u.sha(path)
        record["test_diagnostics"] = a.diagnostics(model, a.load(candidate["spec"], "test", "cuda"))
        rows.append(record)
        del model
    extras = a.evaluate_references()
    u.write_json(a.HERE/"evaluation.json", dict(evaluated_utc=u.now(), rows=rows, references=extras,
        selection_lock_sha256=u.sha(a.HERE/"selection_lock.json")))
    print(json.dumps(dict(stage="evaluated", study=a.HERE.name, selected=len(rows))), flush=True)


def independent_metrics(a, y, pred):
    if a.KIND == "regression":
        errors = [float(p)-float(t) for p, t in zip(pred, y)]
        return dict(mse=math.fsum(e*e for e in errors)/len(errors), mae=math.fsum(abs(e) for e in errors)/len(errors))
    from sklearn.metrics import roc_auc_score
    score = pred[:, 1]-pred[:, 0]
    return dict(accuracy=sum(int((s > 0) == t) for s, t in zip(score, y))/len(y),
        auc=float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
        cross_entropy=math.fsum(float(np.logaddexp(0., (1-2*int(t))*s)) for s, t in zip(score, y))/len(y))


def audit(a):
    choices, evaluation = verify_selection(a), read(a.HERE/"evaluation.json")
    assert evaluation["selection_lock_sha256"] == u.sha(a.HERE/"selection_lock.json")
    rows = {r["selected_id"]: r for r in evaluation["rows"] if r["selected_id"] is not None}
    maximum_metric, maximum_reload = 0., 0.
    candidate_records = {}
    for item in choices["candidate_records"]:
        candidate = read(a.HERE/item["path"])
        candidate_records[candidate["spec"]["id"]] = candidate
        if candidate["status"] != "valid":
            continue
        path = a.HERE/candidate["validation_prediction"]
        assert u.sha(path) == candidate["validation_prediction_sha256"]
        with np.load(path) as z:
            computed = independent_metrics(a, z["truth"], z["prediction"])
            data = a.load(candidate["spec"], "val", "cpu")
            assert np.array_equal(z["ids"], data["ids"]) and np.array_equal(z["truth"], truth(data))
            delta = abs(criterion(a, computed)-candidate["best_validation_loss"])
            maximum_metric = max(maximum_metric, delta)
            assert delta <= 1e-11
        checked = [r for r in candidate["history"] if "validation_loss" in r]
        best = min(checked, key=lambda r: (r["validation_loss"], r["epoch"]))
        assert best["epoch"] == candidate["best_epoch"] and best["validation_loss"] == candidate["best_validation_loss"]
    for choice in choices["selections"]:
        available = [candidate_records[cid] for cid in choice["candidate_ids"] if candidate_records[cid]["status"] == "valid"]
        expected = min(available, key=lambda r: (r["best_validation_loss"], r["spec"]["lr"])) if available else None
        assert choice["selected_id"] == (expected["spec"]["id"] if expected else None)
        if expected is None:
            continue
        model, candidate = selected_model(a, choice)
        row = rows[choice["selected_id"]]
        for split in ["val"]+a.EVALUATION_SPLITS:
            path = a.HERE/(candidate["validation_prediction"] if split == "val" else row[split+"_prediction_file"])
            digest = candidate["validation_prediction_sha256"] if split == "val" else row[split+"_prediction_sha256"]
            assert u.sha(path) == digest
            data = a.load(candidate["spec"], split, "cuda")
            restored = predict(a, model, data)
            with np.load(path) as z:
                assert np.array_equal(z["ids"], data["ids"]) and np.array_equal(z["truth"], truth(data))
                delta = float(np.max(np.abs(restored-z["prediction"])))
                assert delta <= 1e-10
                maximum_reload = max(maximum_reload, delta)
                check = independent_metrics(a, z["truth"], z["prediction"])
                recorded = metrics(a, z["truth"], z["prediction"]) if split == "val" else row[split]
                for key, value in check.items():
                    if value is None:
                        assert recorded[key] is None
                    else:
                        delta = abs(value-recorded[key])
                        assert delta <= 1e-11
                        maximum_metric = max(maximum_metric, delta)
        del model
    a.audit_references(evaluation["references"])
    result = dict(passed=True, audited_utc=u.now(), candidates=len(candidate_records), selections=len(choices["selections"]),
        valid_candidates=sum(r["status"]=="valid" for r in candidate_records.values()),
        valid_selected=len(rows), maximum_independent_metric_delta=maximum_metric,
        maximum_checkpoint_prediction_reload_delta=maximum_reload,
        qualification="Implementation, metric and selection audit; no global optimization or population ranking certificate")
    u.write_json(a.HERE/"final_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["first_cubic", "hierarchical_sequence"])
    parser.add_argument("stage", choices=["lock", "train", "evaluate", "audit"])
    args = parser.parse_args()
    u.configure()
    adapter = load_adapter(args.study)
    {"lock": lock, "train": train, "evaluate": evaluate, "audit": audit}[args.stage](adapter)
