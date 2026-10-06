"""Copied candidate loop with explicit component loss; original source preserved."""
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import component_models as models
import study as base

HERE=Path(__file__).resolve().parent
EPOCHS,BATCH,PATIENCE=150,256,8
now,sha,write_json=base.now,base.sha,base.write_json
get_batch,predict,metrics=base.get_batch,base.predict,base.metrics

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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec["lr"], weight_decay=1e-4)
    penalty_weight = models.CONFIGS[spec["head"]]["penalty"]
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
                  head_parameters=models.parameter_count(model.head), target_mean=mean, target_sd=sd,
                  trainable_parameters=models.trainable_count(model),
                  frozen_parameters=models.parameter_count(model)-models.trainable_count(model),
                  balance_loss_weight=penalty_weight)
    try:
        for epoch in range(1, EPOCHS+1):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(target), generator=generator).cuda()
            squared = torch.zeros((), device="cuda")
            penalty_sum = torch.zeros((), device="cuda")
            largest_norm = torch.zeros((), device="cuda")
            for start in range(0, len(target), BATCH):
                idx = order[start:start+BATCH]
                optimizer.zero_grad(set_to_none=True)
                p = model(*get_batch(train, idx))
                assert p.shape == target[idx].shape
                prediction_loss = (p-target[idx]).square().mean()
                raw_penalty = model.balance_loss()
                loss = prediction_loss + penalty_weight*raw_penalty if penalty_weight else prediction_loss
                loss.backward()
                norm = nn.utils.clip_grad_norm_(trainable, 10.)
                largest_norm = torch.maximum(largest_norm, norm.detach())
                optimizer.step()
                squared += prediction_loss.detach()*len(idx)
                penalty_sum += raw_penalty.detach()*len(idx)
            scheduler.step()
            if not bool(torch.isfinite(squared)) or not bool(torch.isfinite(largest_norm)) or not bool(torch.isfinite(penalty_sum)):
                raise FloatingPointError("Nonfinite loss or gradient at epoch "+str(epoch))
            row = dict(epoch=epoch, training_standardized_mse=float(squared/len(target)),
                       raw_balance_penalty=float(penalty_sum/len(target)),
                       training_total_objective=float((squared+penalty_weight*penalty_sum)/len(target)),
                       maximum_preclip_gradient_norm=float(largest_norm))
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



