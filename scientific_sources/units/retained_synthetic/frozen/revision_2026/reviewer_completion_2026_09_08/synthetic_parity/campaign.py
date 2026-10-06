"""Train all declared candidates, lock all choices, then expose test predictions."""
import argparse
import json
import platform
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import parity_models as pm
import parity_common as c

HERE = c.HERE


def specifications():
    rows = [dict(id=f"n{n}_{head}_seed{seed}_lr{j}", n=n, head=head, seed=seed, lr=lr, lr_index=j)
            for n in c.LENGTHS for head in pm.HEADS for seed in c.SEEDS for j, lr in enumerate(c.LRS)]
    return [rows[i] for i in np.random.default_rng(2026090804).permutation(len(rows))]


def locked_sources():
    return pm.source_paths()+[HERE/"parity_common.py", HERE/"campaign.py", HERE/"audit.py"]


def lock():
    assert not (HERE/"implementation_lock.json").exists()
    audit = json.loads((HERE/"prefit_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    for item in audit["sources"]:
        assert c.sha(item["path"]) == item["sha256"]
    files = locked_sources()+[HERE/"protocol.md", HERE/"cpu_audit.json", HERE/"prefit_audit.json",
                              HERE/"data_manifest.json", HERE/"web_review_27_disposition.md"]
    files += [c.data_path(n, seed) for n in c.LENGTHS for seed in c.SEEDS]
    counts = {head: pm.parameter_count(pm.build_model(100, head, 10)) for head in pm.HEADS}
    record = dict(locked_utc=c.now(), candidates=specifications(), candidate_count=800,
        selection_count=400, analytic_lookup_count=40, sizes=pm.SIZES, total_parameters=counts,
        lengths=c.LENGTHS, seeds=c.SEEDS, learning_rates=c.LRS, epochs=c.EPOCHS, batch_size=c.BATCH,
        optimizer="AdamW; weight_decay=0.0001; cosine T_max=100", clip="10 except lma3_clip1=1",
        minibatch_seed="CPU torch.Generator seed 20000+seed", stochastic_layer_seed="torch.manual_seed(30000+seed) after construction",
        selection="Earliest strict validation-CE minimum; lower numerical LR on exact cross-candidate tie",
        evaluation="N+1 canonical vectors, one fixed batch; row predictions indexed by count",
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                     gpu=torch.cuda.get_device_name(), cpu_threads=4, tf32=False),
        files=[dict(path=str(p), sha256=c.sha(p), bytes=p.stat().st_size) for p in files])
    c.write_json(HERE/"implementation_lock.json", record)
    print(json.dumps(dict(stage="parity_implementation_locked", sha256=c.sha(HERE/"implementation_lock.json"),
                          candidates=800, selections=400)), flush=True)


def verify():
    record = json.loads((HERE/"implementation_lock.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert c.sha(item["path"]) == item["sha256"], item["path"]
    assert record["candidates"] == specifications()
    return record


def train_candidate(spec, lock_sha):
    destination = HERE/"candidates"/(spec["id"]+".json")
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        assert old["spec"] == spec and old["implementation_lock_sha256"] == lock_sha
        assert old["status"] in ("valid", "failed"), "Implementation error requires diagnosis"
        return old
    attempts = HERE/"attempts"
    attempts.mkdir(exist_ok=True)
    attempt = 1+len(list(attempts.glob(spec["id"]+"_attempt*.json")))
    attempt_path = attempts/f'{spec["id"]}_attempt{attempt}.json'
    c.write_json(attempt_path, dict(started_utc=c.now(), spec=spec, attempt=attempt,
                                  note="Only an unfinished interruption may restart from its original seed; numerical failures are retained."))
    train = c.load_data(spec["n"], spec["seed"], "train", "cuda")
    val = c.load_data(spec["n"], spec["seed"], "val")
    y_val = val["y"].numpy()
    model = pm.build_model(spec["seed"], spec["head"], spec["n"]).cuda()
    torch.manual_seed(30000+spec["seed"])
    generator = torch.Generator().manual_seed(20000+spec["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.EPOCHS)
    best, best_epoch, best_state, bad_checks = float("inf"), 0, None, 0
    history, fatal = [], None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = dict(spec=spec, started_utc=c.now(), attempt=attempt, implementation_lock_sha256=lock_sha,
                  total_parameters=pm.parameter_count(model), clip_norm=pm.clip_threshold(spec["head"]))
    try:
        for epoch in range(1, c.EPOCHS+1):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(train["y"]), generator=generator).cuda()
            sum_loss = torch.zeros((), device="cuda")
            correct = torch.zeros((), device="cuda")
            max_norm = torch.zeros((), device="cuda")
            for start in range(0, len(order), c.BATCH):
                idx = order[start:start+c.BATCH]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train["x"][idx])
                assert logits.shape == (len(idx), 2)
                loss = nn.functional.cross_entropy(logits, train["y"][idx])
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), pm.clip_threshold(spec["head"]))
                optimizer.step()
                sum_loss += loss.detach()*len(idx)
                correct += (logits.detach().argmax(1) == train["y"][idx]).sum()
                max_norm = torch.maximum(max_norm, norm.detach())
            scheduler.step()
            if not bool(torch.isfinite(sum_loss)) or not bool(torch.isfinite(max_norm)):
                raise FloatingPointError("Nonfinite training loss or preclip gradient at epoch "+str(epoch))
            row = dict(epoch=epoch, training_cross_entropy=float(sum_loss/len(order)),
                       training_accuracy=float(correct/len(order)), maximum_preclip_gradient_norm=float(max_norm))
            if epoch == 1 or epoch % 5 == 0 or epoch == c.EPOCHS:
                canonical = c.canonical_logits(model, spec["n"])
                score = c.metrics(y_val, canonical[val["count"]])
                row.update(validation_cross_entropy=score["cross_entropy"], validation_accuracy=score["accuracy"])
                diag = c.cp_diagnostics(model, c.canonical(spec["n"], "cuda"))
                if diag is not None:
                    row["canonical_cp_diagnostics"] = diag
                if score["cross_entropy"] < best:
                    best, best_epoch, bad_checks = score["cross_entropy"], epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_checks += 1
            row["epoch_wall_seconds"] = time.perf_counter()-epoch_start
            history.append(row)
            if epoch % 25 == 0:
                print(json.dumps(dict(stage="parity_training", id=spec["id"], epoch=epoch,
                                      best_validation_cross_entropy=best)), flush=True)
            if bad_checks >= c.PATIENCE:
                break
        if best_state is None:
            raise FloatingPointError("No finite validation checkpoint")
        model.load_state_dict(best_state)
        canonical = c.canonical_logits(model, spec["n"])
        validation_logits = canonical[val["count"]]
        assert abs(c.metrics(y_val, validation_logits)["cross_entropy"]-best) <= 1e-12
        checkpoint = HERE/"checkpoints"/(spec["id"]+".pt")
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict=best_state, spec=spec, implementation_lock_sha256=lock_sha), checkpoint)
        prediction_path = HERE/"validation_predictions"/(spec["id"]+".npz")
        prediction_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(prediction_path, count=val["count"], truth=y_val, logits=validation_logits,
                            canonical_logits=canonical)
        result.update(status="valid", best_validation_cross_entropy=best, best_epoch=best_epoch,
                      checkpoint=str(checkpoint.relative_to(HERE)), checkpoint_sha256=c.sha(checkpoint),
                      validation_prediction=str(prediction_path.relative_to(HERE)),
                      validation_prediction_sha256=c.sha(prediction_path))
    except Exception as exc:
        scientific = isinstance(exc, (FloatingPointError, torch.OutOfMemoryError))
        result.update(status="failed" if scientific else "implementation_error", error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc(), best_validation_cross_entropy=None,
                      best_epoch=None)
        if not scientific:
            fatal = exc
    torch.cuda.synchronize()
    result.update(finished_utc=c.now(), epochs_run=len(history), reached_epoch_cap=len(history)==c.EPOCHS,
                  history=history, wall_seconds=time.perf_counter()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    c.write_json(destination, result)
    c.write_json(attempt_path, dict(finished_utc=c.now(), spec=spec, attempt=attempt, status=result["status"]))
    print(json.dumps(dict(stage="parity_candidate_complete", id=spec["id"], status=result["status"],
                          validation_cross_entropy=result["best_validation_cross_entropy"], seconds=result["wall_seconds"])), flush=True)
    del model, optimizer, scheduler, train, val
    torch.cuda.empty_cache()
    if fatal is not None:
        raise fatal
    return result


def train():
    record = verify()
    assert not (HERE/"selection_lock.json").exists() and not (HERE/"evaluation.json").exists()
    lock_sha = c.sha(HERE/"implementation_lock.json")
    done = []
    for spec in record["candidates"]:
        done.append(train_candidate(spec, lock_sha))
        c.write_json(HERE/"progress.json", dict(updated_utc=c.now(), completed=len(done), total=800,
            valid=sum(r["status"]=="valid" for r in done), failed=sum(r["status"]=="failed" for r in done),
            aggregate_fit_seconds=sum(r["wall_seconds"] for r in done), last_id=spec["id"]))
    selections = []
    for n in c.LENGTHS:
        for head in pm.HEADS:
            for seed in c.SEEDS:
                group = [r for r in done if (r["spec"]["n"], r["spec"]["head"], r["spec"]["seed"]) == (n, head, seed)]
                assert len(group) == 2
                valid = [r for r in group if r["status"] == "valid"]
                chosen = min(valid, key=lambda r: (r["best_validation_cross_entropy"], r["spec"]["lr"])) if valid else None
                selections.append(dict(n=n, head=head, seed=seed, selected_id=chosen["spec"]["id"] if chosen else None,
                    validation_cross_entropy=chosen["best_validation_cross_entropy"] if chosen else None,
                    checkpoint_sha256=chosen["checkpoint_sha256"] if chosen else None,
                    candidate_ids=[r["spec"]["id"] for r in group]))
    c.write_json(HERE/"selection_lock.json", dict(locked_utc=c.now(), selections=selections,
        implementation_lock_sha256=lock_sha,
        candidate_records=[dict(path=f'candidates/{r["spec"]["id"]}.json',
                                sha256=c.sha(HERE/"candidates"/(r["spec"]["id"]+".json"))) for r in done]))
    print(json.dumps(dict(stage="parity_all_selections_locked", count=len(selections))), flush=True)


def verify_selection():
    verify()
    record = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    assert record["implementation_lock_sha256"] == c.sha(HERE/"implementation_lock.json")
    assert len(record["selections"]) == 400 and len(record["candidate_records"]) == 800
    for item in record["candidate_records"]:
        assert c.sha(HERE/item["path"]) == item["sha256"]
    return record


def evaluate():
    record = verify_selection()
    assert not (HERE/"evaluation.json").exists()
    output = HERE/"test_predictions"
    output.mkdir(exist_ok=True)
    rows = []
    for choice in record["selections"]:
        if choice["selected_id"] is None:
            rows.append(dict(**choice, status="all_candidates_failed", test=None, population=None))
            continue
        candidate = json.loads((HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
        checkpoint_path = HERE/candidate["checkpoint"]
        assert c.sha(checkpoint_path) == choice["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = pm.build_model(choice["seed"], choice["head"], choice["n"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        canonical = c.canonical_logits(model, choice["n"])
        test = c.load_data(choice["n"], choice["seed"], "test")
        truth, logits = test["y"].numpy(), canonical[test["count"]]
        path = output/(choice["selected_id"]+".npz")
        np.savez_compressed(path, count=test["count"], truth=truth, logits=logits, canonical_logits=canonical)
        rows.append(dict(**choice, status="valid", test=c.metrics(truth, logits),
            population=c.population_metrics(choice["n"], canonical), total_parameters=candidate["total_parameters"],
            best_epoch=candidate["best_epoch"], epochs_run=candidate["epochs_run"], selected_lr=candidate["spec"]["lr"],
            cp_diagnostics=c.cp_diagnostics(model, c.canonical(choice["n"], "cuda")),
            prediction_file=str(path.relative_to(HERE)), prediction_sha256=c.sha(path)))
        del model
    lookups = []
    for n in c.LENGTHS:
        for seed in c.SEEDS:
            training, test = c.load_data(n, seed, "train"), c.load_data(n, seed, "test")
            canonical, totals = c.count_lookup(n, training)
            path = output/f"n{n}_count_lookup_seed{seed}.npz"
            np.savez_compressed(path, count=test["count"], truth=test["y"].numpy(),
                                logits=canonical[test["count"]], canonical_logits=canonical, training_count_frequency=totals)
            lookups.append(dict(n=n, seed=seed, head="count_lookup", test=c.metrics(test["y"].numpy(), canonical[test["count"]]),
                population=c.population_metrics(n, canonical), training_observed_counts=int((totals>0).sum()),
                prediction_file=str(path.relative_to(HERE)), prediction_sha256=c.sha(path)))
    c.write_json(HERE/"evaluation.json", dict(evaluated_utc=c.now(), rows=rows, count_lookups=lookups,
        selection_lock_sha256=c.sha(HERE/"selection_lock.json")))
    print(json.dumps(dict(stage="parity_evaluated", neural=len(rows), lookups=len(lookups))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "train", "evaluate"])
    args = parser.parse_args()
    c.configure()
    assert torch.cuda.is_available()
    {"lock": lock, "train": train, "evaluate": evaluate}[args.stage]()
