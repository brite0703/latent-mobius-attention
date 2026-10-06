"""Locked train/validation campaign; test evaluation is a separate gated stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import models

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/"data"/"lp_pdbbind"/"tensors_reconstructed"
SEEDS = [42, 43, 44, 45, 46]
LRS = [0.0003, 0.001]
EPOCHS = 150
BATCH = 256
PREDICT_BATCH = 64
PATIENCE = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, allow_nan=False)
    temp = path.with_suffix(path.suffix+".pending")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA environment is unavailable.")


def specifications():
    configs = [(1, name) for name in models.HEADS] + [(3, name) for name in models.DEPTH3_HEADS]
    rows = []
    for depth, name in configs:
        for seed in SEEDS:
            for lr_index, lr in enumerate(LRS):
                rows.append(dict(id=f"gcn{depth}_{name}_seed{seed}_lr{lr_index}",
                                 depth=depth, head=name, seed=seed, lr=lr, lr_index=lr_index))
    order = np.random.default_rng(2026090721).permutation(len(rows))
    return [rows[i] for i in order]


def lock_files():
    return [HERE/"models.py", HERE/"study.py", HERE/"prefit_audit.py", HERE/"protocol.md",
            HERE/"prefit_audit.json", HERE/"web_review_21_disposition.md",
            HERE.parent/"lma_revision.py", HERE.parent/"pdbbind_rerun.py",
            HERE.parent.parent/"pdbbind_tensors_experiment.py",
            DATA/"data_audit.json"] + [DATA/f"pdbbind_{s}.pt" for s in ("train", "val", "test")]


def make_lock():
    destination = HERE/"implementation_lock.json"
    if destination.exists():
        raise RuntimeError("An implementation lock already exists; it must not be overwritten.")
    audit = json.loads((HERE/"prefit_audit.json").read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("Pre-fit audit did not pass.")
    for item in audit["audited_sources"]:
        if sha(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("Source changed after the pre-fit audit: "+item["path"])
    configurations = []
    for depth, heads in [(1, models.HEADS), (3, models.DEPTH3_HEADS)]:
        for name in heads:
            model = models.build_model(42, name, depth)
            configurations.append(dict(depth=depth, head=name,
                total_parameters=models.parameter_count(model),
                encoder_parameters=models.parameter_count(model.encoder),
                head_parameters=models.parameter_count(model.head)))
    lock = dict(created_utc=now(), purpose="Exploratory reviewer-directed ligand-only graph pooling comparison",
        heads_and_sizes=models.SIZES, configurations=configurations, candidates=specifications(),
        candidate_count=170, selection_count=85, seeds=SEEDS, learning_rates=LRS,
        epochs=EPOCHS, batch_size=BATCH, prediction_batch_size=PREDICT_BATCH,
        optimizer="AdamW", weight_decay=0.0001, cosine_T_max=EPOCHS, gradient_clip_norm=10.,
        validation_interval=5, patience_checks=PATIENCE, precision="float32; CP product and tanh float64",
        target_scaling="train mean and population SD computed in float64",
        balance_loss_weight=0., search="two learning rates independently per architecture and paired seed",
        status_of_test="Previously inspected, development-used split; no confirmatory claim",
        selection="Minimum finite validation RMSE; earlier epoch within candidate; lower numerical LR for exact cross-candidate ties",
        failure="Nonfinite training invalidates entire candidate; no replacement if both candidates fail",
        determinism="Recorded seeds and same local environment; bitwise cross-hardware identity not asserted",
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                     platform=platform.platform(), gpu=torch.cuda.get_device_name(),
                     gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                     cpu_threads=4, tf32=False),
        files=[dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size) for p in lock_files()])
    write_json(destination, lock)
    print(json.dumps(dict(stage="locked", candidates=170, selections=85, sha256=sha(destination))), flush=True)


def verify_lock():
    lock = json.loads((HERE/"implementation_lock.json").read_text(encoding="utf-8"))
    for entry in lock["files"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Locked source or data changed: "+entry["path"])
    assert lock["candidates"] == specifications()
    return lock


def load_split(split):
    b = torch.load(DATA/f"pdbbind_{split}.pt", weights_only=True, map_location="cpu")
    assert b["mask"].dtype == torch.bool
    counts = b["mask"].sum(1)
    assert bool((counts > 0).all())
    # Only this proven prefix-mask property permits trimming batches by counts.
    assert torch.equal(b["mask"], torch.arange(b["mask"].shape[1])[None, :] < counts[:, None])
    assert not bool(b["X"][~b["mask"]].any())
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}


def get_batch(blob, idx):
    mask = blob["mask"][idx]
    # Prefix-mask condition checked once on loading; no outcome-dependent trim.
    n = int(mask.sum(1).max())
    return blob["X"][idx, :n], mask[:, :n], blob["adj"][idx, :n, :n]


def predict(model, blob, target_mean, target_sd):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(blob["y"]), PREDICT_BATCH):
            idx = torch.arange(start, min(start+PREDICT_BATCH, len(blob["y"])), device="cuda")
            p = model(*get_batch(blob, idx))
            if p.shape != idx.shape:
                raise ValueError("Prediction shape mismatch")
            predictions.append(p.double()*target_sd+target_mean)
    return torch.cat(predictions).cpu().numpy()


def metrics(truth, prediction):
    y, p = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Invalid regression vectors")
    return dict(rmse=float(np.sqrt(np.mean((p-y)**2))), mae=float(np.mean(np.abs(p-y))),
                pearson=float(np.corrcoef(p, y)[0, 1]) if np.std(p)>0 and np.std(y)>0 else None)


def train_candidate(spec, train, val, mean, sd, lock_sha):
    output = HERE/"candidates"/(spec["id"]+".json")
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old["implementation_lock_sha256"] != lock_sha or old["spec"] != spec:
            raise RuntimeError("Incompatible completed candidate")
        return old
    attempt_dir = HERE/"attempts"
    attempt_dir.mkdir(exist_ok=True)
    attempt = sum(1 for _ in attempt_dir.glob(spec["id"]+"_attempt*.json"))+1
    attempt_path = attempt_dir/f'{spec["id"]}_attempt{attempt}.json'
    write_json(attempt_path, dict(started_utc=now(), spec=spec, attempt=attempt,
                                  note="An interrupted unfinished fit may restart from its recorded seed. Scientific failures are not retried."))
    model = models.build_model(spec["seed"], spec["head"], spec["depth"]).cuda()
    torch.manual_seed(30000+spec["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    generator = torch.Generator().manual_seed(20000+spec["seed"])
    target = ((train["y"].double()-mean)/sd).float()
    val_truth = val["y"].double().cpu().numpy()
    best, best_epoch, best_state, bad_checks = float("inf"), 0, None, 0
    history = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = dict(spec=spec, started_utc=now(), implementation_lock_sha256=lock_sha,
                  attempt=attempt, total_parameters=models.parameter_count(model),
                  encoder_parameters=models.parameter_count(model.encoder),
                  head_parameters=models.parameter_count(model.head), target_mean=mean, target_sd=sd)
    try:
        for epoch in range(1, EPOCHS+1):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(target), generator=generator).cuda()
            squared = torch.zeros((), device="cuda")
            largest_norm = torch.zeros((), device="cuda")
            cp_maximum = torch.zeros((), device="cuda", dtype=torch.float64)
            cp_saturated = torch.zeros((), device="cuda", dtype=torch.int64)
            cp_zero = torch.zeros((), device="cuda", dtype=torch.int64)
            cp_factor_gradient = torch.zeros((), device="cuda")
            cp_product_count = 0
            for start in range(0, len(target), BATCH):
                idx = order[start:start+BATCH]
                optimizer.zero_grad(set_to_none=True)
                p = model(*get_batch(train, idx))
                assert p.shape == target[idx].shape
                loss = (p-target[idx]).square().mean()
                loss.backward()
                if spec["head"] == "cp_pool":
                    product = model.head.last_product.abs()
                    cp_maximum = torch.maximum(cp_maximum, product.max())
                    cp_saturated += (product >= 10.).sum()
                    cp_zero += (product == 0.).sum()
                    cp_product_count += product.numel()
                    cp_factor_gradient = torch.maximum(cp_factor_gradient, model.head.factor.weight.grad.detach().norm())
                norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
                largest_norm = torch.maximum(largest_norm, norm.detach())
                optimizer.step()
                squared += loss.detach()*len(idx)
            scheduler.step()
            if not bool(torch.isfinite(squared)) or not bool(torch.isfinite(largest_norm)):
                raise FloatingPointError("Nonfinite loss or gradient at epoch "+str(epoch))
            row = dict(epoch=epoch, training_standardized_mse=float(squared/len(target)),
                       maximum_preclip_gradient_norm=float(largest_norm))
            if spec["head"] == "cp_pool":
                row["cp_numerics"] = dict(max_abs_pretanh_product=float(cp_maximum) if bool(torch.isfinite(cp_maximum)) else None,
                                         maximum_product_finite=bool(torch.isfinite(cp_maximum)),
                                         fraction_abs_product_at_least_10=float(cp_saturated)/cp_product_count,
                                         fraction_zero_products=float(cp_zero)/cp_product_count,
                                         maximum_factor_weight_gradient_norm=float(cp_factor_gradient))
            if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                vp = predict(model, val, mean, sd)
                value = metrics(val_truth, vp)["rmse"]
                row["validation_rmse"] = value
                if value < best:
                    best, best_epoch, bad_checks = value, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_checks += 1
            row["epoch_wall_seconds"] = time.perf_counter()-epoch_start
            history.append(row)
            if epoch % 25 == 0:
                print(json.dumps(dict(stage="training", id=spec["id"], epoch=epoch,
                                      best_validation_rmse=best)), flush=True)
            if bad_checks >= PATIENCE:
                break
        if best_state is None:
            raise FloatingPointError("No finite validation checkpoint")
        model.load_state_dict(best_state)
        validation_prediction = predict(model, val, mean, sd)
        if abs(metrics(val_truth, validation_prediction)["rmse"]-best) > 1e-10:
            raise RuntimeError("Selected checkpoint does not reproduce validation score")
        checkpoint = HERE/"checkpoints"/(spec["id"]+".pt")
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict=best_state, spec=spec, target_mean=mean, target_sd=sd,
                        implementation_lock_sha256=lock_sha), checkpoint)
        val_file = HERE/"validation_predictions"/(spec["id"]+".npz")
        val_file.parent.mkdir(exist_ok=True)
        np.savez_compressed(val_file, ids=np.asarray(val["ids"]), truth=val_truth,
                            prediction=validation_prediction)
        result.update(status="valid", best_validation_rmse=best, best_epoch=best_epoch,
                      checkpoint=str(checkpoint.relative_to(HERE)), checkpoint_sha256=sha(checkpoint),
                      validation_predictions=str(val_file.relative_to(HERE)), validation_prediction_sha256=sha(val_file))
    except Exception as exc:
        # Scientific nonfinite/OOM events remain in the record, including history.
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), best_validation_rmse=None, best_epoch=None)
    torch.cuda.synchronize()
    result.update(finished_utc=now(), epochs_run=len(history), history=history,
                  reached_epoch_cap=len(history)==EPOCHS,
                  wall_seconds=time.perf_counter()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write_json(output, result)
    write_json(attempt_path, dict(spec=spec, attempt=attempt, finished_utc=now(),
                                  status=result["status"], completed_record=str(output.relative_to(HERE))))
    print(json.dumps(dict(stage="candidate_complete", id=spec["id"], status=result["status"],
                          best_validation_rmse=result["best_validation_rmse"],
                          seconds=result["wall_seconds"])), flush=True)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def train_all():
    lock = verify_lock()
    lock_sha = sha(HERE/"implementation_lock.json")
    # This stage never loads the test tensors.
    train, val = load_split("train"), load_split("val")
    assert not set(train["ids"]) & set(val["ids"])
    mean = float(train["y"].double().mean())
    sd = float(train["y"].double().std(unbiased=False))
    for index, spec in enumerate(lock["candidates"]):
        train_candidate(spec, train, val, mean, sd, lock_sha)
        records = [json.loads(p.read_text(encoding="utf-8")) for p in (HERE/"candidates").glob("*.json")]
        write_json(HERE/"progress.json", dict(updated_utc=now(), completed=len(records), total=170,
                   valid=sum(r["status"]=="valid" for r in records),
                   failed=sum(r["status"]=="failed" for r in records),
                   aggregate_fit_seconds=sum(r["wall_seconds"] for r in records), last_id=spec["id"]))
    select_all(lock)


def select_all(lock=None):
    lock = verify_lock() if lock is None else lock
    destination = HERE/"selection_lock.json"
    if destination.exists():
        raise RuntimeError("Global selection has already been locked")
    selections, candidate_records = [], []
    for config in lock["configurations"]:
        for seed in SEEDS:
            candidates = []
            for spec in lock["candidates"]:
                if (spec["depth"], spec["head"], spec["seed"]) == (config["depth"], config["head"], seed):
                    p = HERE/"candidates"/(spec["id"]+".json")
                    record = json.loads(p.read_text(encoding="utf-8"))
                    candidates.append(record)
                    candidate_records.append(dict(path=str(p.relative_to(HERE)), sha256=sha(p)))
            assert len(candidates) == 2
            valid = [r for r in candidates if r["status"] == "valid"]
            chosen = min(valid, key=lambda r: (r["best_validation_rmse"], r["spec"]["lr"])) if valid else None
            selections.append(dict(depth=config["depth"], head=config["head"], seed=seed,
                selected_id=chosen["spec"]["id"] if chosen else None,
                validation_rmse=chosen["best_validation_rmse"] if chosen else None,
                candidate_ids=[r["spec"]["id"] for r in candidates],
                checkpoint_sha256=chosen.get("checkpoint_sha256") if chosen else None))
    assert len(selections) == 85 and len(candidate_records) == 170
    write_json(destination, dict(locked_utc=now(), implementation_lock_sha256=sha(HERE/"implementation_lock.json"),
               candidate_records=candidate_records, selections=selections,
               test_status="All new choices fixed globally before this campaign's test scoring; prior development reuse remains."))
    print(json.dumps(dict(stage="all_selections_locked", count=85, sha256=sha(destination))), flush=True)


def evaluate():
    verify_lock()
    selection = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    for entry in selection["candidate_records"]:
        if sha(HERE/entry["path"]) != entry["sha256"]:
            raise RuntimeError("Candidate changed after selection")
    test = load_split("test")
    y = test["y"].double().cpu().numpy()
    rows = []
    output_dir = HERE/"test_predictions"
    output_dir.mkdir(exist_ok=True)
    for chosen in selection["selections"]:
        if chosen["selected_id"] is None:
            rows.append(dict(**chosen, status="all_candidates_failed", test=None))
            continue
        candidate = json.loads((HERE/"candidates"/(chosen["selected_id"]+".json")).read_text(encoding="utf-8"))
        checkpoint_path = HERE/candidate["checkpoint"]
        if sha(checkpoint_path) != chosen["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        model = models.build_model(chosen["seed"], chosen["head"], chosen["depth"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        p = predict(model, test, checkpoint["target_mean"], checkpoint["target_sd"])
        path = output_dir/(chosen["selected_id"]+".npz")
        np.savez_compressed(path, ids=np.asarray(test["ids"]), truth=y, prediction=p)
        rows.append(dict(**chosen, status="valid", test=metrics(y, p),
                         total_parameters=candidate["total_parameters"],
                         selected_epochs=candidate["epochs_run"], best_epoch=candidate["best_epoch"],
                         selected_lr=candidate["spec"]["lr"],
                         prediction_file=str(path.relative_to(HERE)), prediction_sha256=sha(path)))
        del model
    first = next(r for r in rows if r["status"]=="valid")
    first_candidate = json.loads((HERE/"candidates"/(first["selected_id"]+".json")).read_text(encoding="utf-8"))
    null = metrics(y, np.full(len(y), first_candidate["target_mean"]))
    write_json(HERE/"evaluation.json", dict(evaluated_utc=now(), selection_lock_sha256=sha(HERE/"selection_lock.json"),
               test_size=len(y), training_mean_baseline=null, rows=rows))
    print(json.dumps(dict(stage="evaluation_complete", selected=len(rows), valid=sum(r["status"]=="valid" for r in rows))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "train", "select", "evaluate"])
    args = parser.parse_args()
    configure()
    {"lock": make_lock, "train": train_all, "select": select_all, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
