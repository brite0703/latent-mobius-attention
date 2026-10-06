"""Six matched native N20 fits: perturbed known witness versus original initialization."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, importlib.util, json, hashlib, time, traceback, math
import numpy as np
import torch
from torch import nn
from training_chain_checks import native_chain_check, optimizer_coverage, capture_first_update, check_first_update

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / "lboia_routing_parity_pilot_20261006"
sys.path.insert(0, str(FIRST))


def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = load_module("original_first_pilot_diagnostics", FIRST / "pilot.py")
pm, pc = diag.pm, diag.pc


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".pending")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def append(p, obj):
    with Path(p).open("a") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")


def read(p):
    return json.loads(Path(p).read_text())


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}



def scored_metrics(y, logits):
    result = pc.metrics(y, logits)
    signed = (2*np.asarray(y, dtype=np.int64)-1)*(np.asarray(logits)[:, 1]-np.asarray(logits)[:, 0])
    result.update(minimum_signed_margin=float(signed.min()), mean_signed_margin=float(signed.mean()),
                  median_signed_margin=float(np.median(signed)),
                  percentile05_signed_margin=float(np.quantile(signed, .05)),
                  negative_margin_fraction=float(np.mean(signed < 0)))
    return result

def specs():
    return [{"id": f"n20_{condition}_seed{seed}", "kind": "parity", "head": "lma3",
             "n": 20, "seed": seed, "initializer": condition, "lr": .001, "clip": 10.}
            for seed in [100, 101, 102] for condition in ["original", "perturbed_witness"]]


def verify_inputs():
    lock = read(HERE / "evidence/protocol_lock.json")
    for e in lock["files"]:
        assert sha(e["path"]) == e["sha256"], "Identity mismatch: " + e["path"]
    assert Path(pm.source.__file__).resolve() == (FIRST / "original_workspace/LMA/lma.py").resolve()
    assert sha(pm.source.__file__) == lock["native_lma_sha256"]
    assert Path(pm.source.LMANetwork.forward.__code__.co_filename).resolve() == Path(pm.source.__file__).resolve()
    assert sha(FIRST / "pilot.py") == lock["snapshot_source_sha256"]
    assert torch.cuda.is_available(), "Authorized GPU unavailable; no implicit CPU substitution"
    pc.configure()
    torch.use_deterministic_algorithms(True)


def deadline():
    if time.time() >= read(HERE / "evidence/budget.json")["deadline_unix"]:
        raise TimeoutError("Prespecified 1800-second wall budget exhausted")


def instantiate(spec):
    model = pm.build_model(spec["seed"], "lma3", 20).cuda()
    checkpoint = torch.load(HERE / "initial_states" / (spec["id"] + ".pt"),
                            map_location="cpu", weights_only=True)
    assert checkpoint["spec"] == spec
    model.load_state_dict(checkpoint["state_dict"])
    assert sum(p.numel() for p in model.parameters()) == 12525
    assert all(p.requires_grad for p in model.parameters())
    torch.manual_seed(30000 + spec["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    gen = torch.Generator(device="cpu").manual_seed(20000 + spec["seed"])
    train = pc.load_data(20, spec["seed"], "train", "cuda")
    val = pc.load_data(20, spec["seed"], "val")
    optimizer_coverage(model, opt)
    return model, opt, sched, gen, train, val


def canonical_diagnostic(model, spec, train, val, epoch, folder, initial=False):
    # One canonical batch supplies diagnostics and count-lookup metrics.
    # None of these canonical representatives is used for gradient training.
    was = model.training
    before = state_cpu(model)
    rng, crng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    captured = []
    handle = model.layers[0].register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    logits = pc.canonical_logits(model, 20)
    handle.remove()
    features = captured[0].mean(1).double().cpu().numpy()
    token_spread = float((captured[0][:, :, 2].max(1).values -
                          captured[0][:, :, 2].min(1).values).max())
    distance = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    train_y = train["y"].cpu().numpy()
    train_score = scored_metrics(train_y, logits[train["count"]])
    val_score = pc.metrics(val["y"].numpy(), logits[val["count"]])
    feature_record = {
        "epoch": epoch, "canonical_mean_layer_features": features.tolist(),
        "minimum_distinct_count_feature_l2_distance": float(distance.min()),
        "count_coordinate2": features[:, 2].tolist(),
        "count_coordinate2_token_spread": token_spread,
        "canonical_logits": logits.tolist(),
        "training_count_lookup_metrics": train_score,
        "validation_metrics": val_score,
        "not_used_for_gradient_training": True}
    if initial:
        actual = []
        with torch.inference_mode():
            for start in range(0, len(train_y), 256):
                actual.append(model(train["x"][start:start+256]).double().cpu().numpy())
        actual = np.concatenate(actual)
        feature_record["actual_row_training_metrics"] = pc.metrics(train_y, actual)
        feature_record["max_actual_canonical_logit_difference"] = float(np.max(np.abs(
            actual - logits[train["count"]])))
        if spec["initializer"] == "perturbed_witness":
            eps16, eps32 = model.layers[0].interaction_mlps[0][0].eps, model.layers[0].layer_norm.eps
            def count_state(c):
                u = (c-10)/10
                t = u/math.sqrt((1+u*u)/8+eps16)
                return t/math.sqrt((1+t*t)/16+eps32)
            ideal = np.asarray([count_state(c) for c in range(21)])
            feature_record["maximum_ideal_count_state_error"] = float(np.max(np.abs(
                features[:, 2] - ideal)))
            assert feature_record["maximum_ideal_count_state_error"] < 1e-5
            assert np.all(np.diff(features[:, 2]) > 0), "Declared injective count initialization failed"
        # Initial performance is reported, never used to change seeds or configuration.
        feature_record["already_meets_training_success_before_updates"] = (
            feature_record["actual_row_training_metrics"]["accuracy"] >= .99 and
            feature_record["actual_row_training_metrics"]["cross_entropy"] <= .05)
    assert all(torch.equal(t, model.state_dict()[n].detach().cpu()) for n, t in before.items())
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(crng, torch.cuda.get_rng_state())
    model.train(was)
    append(folder / "count_features.jsonl", feature_record)
    return train_score, val_score


def train_one(spec, epochs=100):
    folder = HERE / "runs" / spec["id"]
    assert not folder.exists(), "No duplicate or resumed fit"
    folder.mkdir(parents=True)
    started_count = len(list((HERE / "runs").iterdir()))
    assert started_count <= 6
    append(HERE / "evidence/started_runs.jsonl",
           {"spec": spec, "started_utc": pc.now(), "started_training_runs": started_count})
    model, opt, sched, gen, train, val = instantiate(spec)
    history, best, best_epoch, best_state = [], math.inf, None, None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    peak_rss = diag.rss_mib()
    dump(folder / "started.json", {"spec": spec, "started_utc": pc.now(),
                                  "initial_state_sha256": sha(HERE / "initial_states" / (spec["id"]+".pt")),
                                  "all_parameters_trainable": True})
    try:
        deadline()
        print(json.dumps({"stage": "ACTUAL_TRAINING_RUN_START", "utc": pc.now(),
                          "run_id": spec["id"], "started_training_runs": started_count}), flush=True)
        initial_train, _ = canonical_diagnostic(model, spec, train, val, 0, folder, initial=True)
        diag.snapshot(model, spec, train, None, None, 0, folder)
        for epoch in range(1, epochs+1):
            deadline()
            tick = time.perf_counter()
            model.train()
            order = torch.randperm(len(train["y"]), generator=gen).cuda()
            batch_hash = hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            sumloss, correct, maxnorm, clipped = 0., 0, 0., 0
            for start in range(0, len(order), 256):
                deadline()
                idx = order[start:start+256]
                opt.zero_grad(set_to_none=True)
                out = model(train["x"][idx])
                loss = nn.functional.cross_entropy(out, train["y"][idx])
                loss.backward()
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), 10.))
                if not math.isfinite(float(loss.detach())) or not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite training loss/gradient")
                first_update = capture_first_update(model, opt) if epoch == 1 and start == 0 else None
                opt.step()
                if first_update is not None:
                    dump(folder / "optimizer_update_step1.json", check_first_update(model, opt, first_update))
                sumloss += float(loss.detach()) * len(idx)
                correct += int((out.detach().argmax(1) == train["y"][idx]).sum())
                maxnorm, clipped = max(maxnorm, norm), clipped + (norm > 10.)
                append(folder / "minibatches.jsonl",
                       {"epoch": epoch, "batch_index": start//256, "rows": len(idx),
                        "prediction_loss": float(loss.detach()), "preclip_global_gradient_norm": norm,
                        "clip_factor": min(1., 10./(norm+1e-6))})
            sched.step()
            row = {"epoch": epoch, "batch_order_sha256": batch_hash,
                   "online_training_cross_entropy": sumloss / len(order),
                   "online_training_accuracy": correct / len(order),
                   "maximum_preclip_gradient_norm": maxnorm, "clipped_minibatches": int(clipped),
                   "total_minibatches": math.ceil(len(order)/256),
                   "learning_rate_after_epoch": sched.get_last_lr()[0]}
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                fit, validation = canonical_diagnostic(model, spec, train, val, epoch, folder)
                row["fitting_count_lookup"] = fit
                row["validation"] = validation
                if validation["cross_entropy"] < best:
                    best, best_epoch, best_state = validation["cross_entropy"], epoch, state_cpu(model)
            diag.snapshot(model, spec, train, None, None, epoch, folder)
            torch.cuda.synchronize()
            row.update(epoch_wall_seconds=time.perf_counter()-tick,
                       cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,
                       cuda_reserved_mib=torch.cuda.memory_reserved()/2**20, rss_mib=diag.rss_mib())
            peak_rss = max(peak_rss, row["rss_mib"])
            history.append(row)
            append(folder / "epochs.jsonl", row)
            dump(HERE / "evidence/progress.json",
                 {"updated_utc": pc.now(), "run_id": spec["id"], "epoch": epoch,
                  "maximum_epochs": epochs, "started_training_runs": started_count,
                  "elapsed_budget_seconds": time.time()-read(HERE / "evidence/budget.json")["start_unix"]})
            if started_count == 1 and epoch == 2:
                dump(HERE / "evidence/first_original_run_feasibility.json",
                     {"counted_as_run": 1, "run_id": spec["id"],
                      "first_two_epoch_seconds": [x["epoch_wall_seconds"] for x in history],
                      "horizon_remains_fixed100": True, "configuration_not_adapted_to_outcomes": True})
            if epoch % 10 == 0 or epoch == epochs:
                print(json.dumps({"stage": "training", "run_id": spec["id"],
                                  "epoch": epoch, "wall_seconds": time.perf_counter()-started}), flush=True)
        assert best_state is not None
        torch.save({"state_dict": state_cpu(model), "spec": spec}, folder / "final.pt")
        torch.save({"state_dict": best_state, "spec": spec}, folder / "best_validation.pt")
        result = {"spec": spec, "status": "completed", "epochs": len(history),
                  "optimizer_steps": len(history)*math.ceil(len(train["y"])/256),
                  "initial_training_metrics": initial_train,
                  "best_validation_cross_entropy": best, "best_validation_epoch": best_epoch,
                  "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                  "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                  "peak_sampled_rss_mib": peak_rss, "finished_utc": pc.now(),
                  "final_checkpoint_sha256": sha(folder / "final.pt"),
                  "best_validation_checkpoint_sha256": sha(folder / "best_validation.pt")}
        dump(folder / "result.json", result)
        print(json.dumps({"stage": "run_complete", "run_id": spec["id"], "result": result}), flush=True)
        return "completed"
    except Exception as exc:
        torch.save({"state_dict": state_cpu(model), "spec": spec}, folder / "failed_or_interrupted.pt")
        status = ("budget_terminated" if isinstance(exc, TimeoutError) else
                  "scientific_failure" if isinstance(exc, (FloatingPointError, torch.OutOfMemoryError)) else
                  "implementation_error")
        dump(folder / "result.json",
             {"spec": spec, "status": status, "epochs": len(history),
              "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
              "wall_seconds": time.perf_counter()-started, "finished_utc": pc.now()})
        if status == "implementation_error":
            raise
        return status
    finally:
        del model, opt, sched, gen, train, val
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate():
    lock = read(HERE / "evidence/all_fits_locked.json")
    evaluations = []
    for entry in lock["results"]:
        deadline()
        folder = HERE / "runs" / entry["id"]
        assert sha(folder / "result.json") == entry["record_sha256"]
        result = read(folder / "result.json")
        spec = result["spec"]
        if result["status"] != "completed":
            evaluations.append({"spec": spec, "status": result["status"]})
            continue
        model = pm.build_model(spec["seed"], "lma3", 20).cuda()
        row = {"spec": spec, "status": "completed"}
        for name in ["final", "best_validation"]:
            checkpoint = folder / (name+".pt")
            assert sha(checkpoint) == entry[name+"_checkpoint_sha256"]
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
            model.eval()
            canonical = pc.canonical_logits(model, 20)
            scores = {}
            for split in ["train", "val", "test"]:
                deadline()
                blob = pc.load_data(20, spec["seed"], split, "cuda")
                actual = np.concatenate([model(blob["x"][start:start+256]).double().cpu().numpy()
                                         for start in range(0, len(blob["y"]), 256)])
                lookup = canonical[blob["count"]]
                assert np.allclose(actual, lookup, atol=5e-4, rtol=5e-4), "Actual/canonical disagreement"
                y = blob["y"].cpu().numpy()
                scores[split] = pc.metrics(y, actual)
                scores[split]["count_lookup_metrics"] = pc.metrics(y, lookup)
                scores[split]["max_actual_canonical_logit_difference"] = float(np.max(np.abs(actual-lookup)))
                if split == "train":
                    scores[split].update(
                        fit_success_accuracy_ge_099=scores[split]["accuracy"] >= .99,
                        fit_success_ce_le_005=scores[split]["cross_entropy"] <= .05,
                        fit_success_both=scores[split]["accuracy"] >= .99 and scores[split]["cross_entropy"] <= .05)
                np.savez_compressed(folder / (name+"_"+split+"_predictions.npz"),
                                    truth=y, count=blob["count"], actual_row_logits=actual,
                                    canonical_logits=canonical)
            scores["population_from_count_orbits"] = pc.population_metrics(20, canonical)
            row[name] = scores
        evaluations.append(row)
        del model
        torch.cuda.empty_cache()
    dump(HERE / "evidence/evaluation.json",
         {"evaluated_utc": pc.now(), "all_fits_lock_sha256": sha(HERE / "evidence/all_fits_locked.json"),
          "rows": evaluations, "elapsed_budget_seconds": time.time()-read(HERE / "evidence/budget.json")["start_unix"]})
    return evaluations


def main():
    verify_inputs()
    assert not (HERE / "evidence/budget.json").exists(), "This campaign executes once"
    (HERE / "runs").mkdir(exist_ok=False)
    start = time.time()
    dump(HERE / "evidence/budget.json",
         {"start_unix": start, "deadline_unix": start+1800., "start_utc": pc.now(),
          "maximum_started_training_runs": 6, "maximum_wall_seconds": 1800})
    print(json.dumps({"stage": "AUTHORIZED_GPU_DIAGNOSTICS_START", "utc": pc.now(),
                      "gpu": torch.cuda.get_device_name(0), "protocol_sha256": sha(HERE / "evidence/protocol_frozen.json"),
                      "max_new_fits": 6, "maximum_seconds": 1800}), flush=True)
    halted = False
    try:
        source_model = pm.build_model(100, "lma3", 20)
        source_model.load_state_dict(torch.load(HERE / "initial_states/n20_original_seed100.pt",
                                                map_location="cpu", weights_only=True)["state_dict"])
        diagnostic_data = pc.load_data(20, 100, "train", "cuda")
        check_start = time.perf_counter()
        checks = native_chain_check(source_model, diagnostic_data["x"][:16],
                                    diagnostic_data["y"][:16], HERE / "evidence")
        checks["wall_seconds"] = time.perf_counter()-check_start
        dump(HERE / "evidence/training_chain_checks.json", checks)
        print(json.dumps({"stage": "native_chain_checks_passed", "finite_difference_checks": 30,
                          "optimizer_steps": 0, "wall_seconds": checks["wall_seconds"]}), flush=True)
        del source_model, diagnostic_data
        torch.cuda.empty_cache()
    except Exception as exc:
        dump(HERE / "evidence/training_chain_check_failure.json",
             {"failed_utc": pc.now(), "error": str(exc), "traceback": traceback.format_exc(),
              "training_runs_started": 0, "optimizer_steps": 0})
        raise
    for spec in specs():
        try:
            deadline()
            status = train_one(spec)
            if status == "budget_terminated":
                halted = True
                break
        except Exception as exc:
            dump(HERE / "evidence/campaign_failure.json",
                 {"failed_utc": pc.now(), "error": str(exc), "traceback": traceback.format_exc()})
            halted = True
            break
    results = []
    for spec in specs():
        folder = HERE / "runs" / spec["id"]
        if (folder / "result.json").exists():
            r = read(folder / "result.json")
            results.append({"id": spec["id"], "status": r["status"],
                            "record_sha256": sha(folder / "result.json"),
                            "final_checkpoint_sha256": r.get("final_checkpoint_sha256"),
                            "best_validation_checkpoint_sha256": r.get("best_validation_checkpoint_sha256")})
        else:
            results.append({"id": spec["id"], "status": "not_started_or_incomplete",
                            "reason": "budget or campaign stop", "record_sha256": None})
    dump(HERE / "evidence/all_fits_locked.json",
         {"locked_utc": pc.now(), "protocol_sha256": sha(HERE / "evidence/protocol_frozen.json"),
          "results": results, "started_training_runs": len(list((HERE / "runs").iterdir())),
          "elapsed_budget_seconds": time.time()-start, "halted": halted})
    if not halted:
        evaluate()
    verify_inputs()
    dump(HERE / "evidence/completion.json",
         {"finished_utc": pc.now(), "elapsed_wall_seconds": time.time()-start,
          "started_training_runs": len(list((HERE / "runs").iterdir())),
          "completed_training_runs": sum(r["status"] == "completed" for r in results),
          "halted": halted, "protected_inputs_unchanged": True,
          "evaluation_completed": (HERE / "evidence/evaluation.json").exists()})
    print(json.dumps(read(HERE / "evidence/completion.json")), flush=True)
    return 1 if halted else 0


if __name__ == "__main__":
    sys.exit(main())

