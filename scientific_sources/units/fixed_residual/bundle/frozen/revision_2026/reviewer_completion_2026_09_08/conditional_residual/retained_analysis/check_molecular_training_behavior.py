"""Reconcile all 30 molecular terminal states on fitting/validation caches.

This diagnostic follows the observed common epoch-zero selection. It performs
no training, no test-data access and no new checkpoint selection. The saved
terminal residual equations are evaluated directly without the model module.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDY = ROOT / "retained_study"
SOURCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound(record, tensor=False):
    path = Path(record["path"])
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    require(digest == record["sha256"], "Changed source: " + str(path))
    SOURCES[str(path.resolve())] = digest
    return torch.load(io.BytesIO(blob), weights_only=True, map_location="cpu") if tensor else json.loads(blob)


def reference(state, arm, z):
    pairs = state["pairs"]
    require(torch.equal(pairs, torch.combinations(torch.arange(8), r=2)), "Changed pair order")
    if arm == "pair_mlp":
        joined = torch.cat((z[:, pairs[:, 0]], z[:, pairs[:, 1]]), -1)
        hidden = F.gelu(F.linear(joined, state["pair_map.0.weight"], state["pair_map.0.bias"]))
        interaction = F.linear(hidden, state["pair_map.2.weight"], state["pair_map.2.bias"])
    else:
        left, right = F.linear(z, state["legs.weight"], state["legs.bias"]).chunk(2, -1)
        a, b = left[:, pairs[:, 0]], right[:, pairs[:, 1]]
        interaction = a * b if arm == "product" else (a + b) / 2
    normalized = F.layer_norm(interaction, (16,), state["norm.weight"], state["norm.bias"], 1e-5)
    hidden = F.gelu(F.linear(normalized, state["post.weight"], state["post.bias"]))
    return F.linear(hidden.mean(1), state["output.weight"], state["output.bias"]).squeeze(-1)


def mse(truth, prediction):
    differences = [float(p)-float(y) for y, p in zip(truth, prediction)]
    require(len(differences) > 0 and all(math.isfinite(x) for x in differences), "Invalid residual prediction")
    return math.fsum(x*x for x in differences)/len(differences)


def main():
    status = read(ROOT / "retained_pipeline_status.json")
    require(status["stage"] != "profile" or status["status"] == "complete", "Defer analysis during retained timing")
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CPU-only diagnostic")
    selection_path = STUDY / "selection_lock.json"
    locked = bound(dict(path=str(selection_path), sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest()))
    rows = []
    for artifact in locked["candidates"]:
        result = bound(artifact)
        spec = result["spec"]
        if spec["domain"] != "ligand_contact":
            continue
        require(result["status"] == "valid" and result["best_epoch"] == 0 and result["completed_epochs"] == 35,
                "The observed molecular selection pattern changed")
        runtime = bound(result["continuation_state"], tensor=True)
        binding = bound(result["binding"])
        state = runtime["model_state"]
        require(result["optimizer_updates"] == runtime["updates"] == 175, "Incorrect executed update count")
        require(len(runtime["epoch_rows"]) == 35 and all(row["batch_sizes"] == [256, 256, 256, 256, 72]
                                                       for row in runtime["epoch_rows"]), "Final batch was omitted or resized")
        changed = [key for key in state if not torch.equal(state[key], runtime["best_state"][key])]
        require("output.weight" in changed and torch.count_nonzero(state["output.weight"]) > 0,
                "Terminal branch remained at its zero-output initialization")
        flat = dict(spec, completed_epochs=35, optimizer_updates=175, changed_state_tensors=len(changed),
                    changed_state_names=changed, terminal_output_weight_norm=float(torch.linalg.vector_norm(state["output.weight"])))
        for split in ("train", "val"):
            cache = bound(binding["caches"][split], tensor=True)
            require(cache["split"] == split and cache["parent_checkpoint_sha256"] == result["parent_checkpoint_sha256"],
                    "Cache or parent differs")
            require(len(cache["ids"]) == (1096 if split == "train" else 150), "Changed fitting/validation cohort")
            with torch.inference_mode():
                corrections = torch.cat([reference(state, spec["arm"], cache["z"][i:i+256])
                                         for i in range(0, len(cache["ids"]), 256)])
                terminal = (cache["f0"] + corrections).double()*cache["target_sd"] + cache["target_mean"]
                baseline = cache["f0"].double()*cache["target_sd"] + cache["target_mean"]
            flat[split+"_baseline_mse"] = mse(cache["truth"].tolist(), baseline.tolist())
            flat[split+"_terminal_mse"] = mse(cache["truth"].tolist(), terminal.tolist())
            flat[split+"_mse_difference"] = flat[split+"_terminal_mse"] - flat[split+"_baseline_mse"]
            flat[split+"_correction_rms_parent_units"] = math.sqrt(math.fsum(float(x)**2 for x in corrections)/len(corrections))
            if split == "val":
                delta = abs(flat["val_terminal_mse"] - runtime["history"][-1]["validation_mse"])
                require(delta <= 1e-12 + 1e-11*abs(runtime["history"][-1]["validation_mse"]), "Terminal validation calculation does not reproduce")
                flat["logged_terminal_validation_max_delta"] = delta
        rows.append(flat)
    require(len(rows) == 30 and len({row["id"] for row in rows}) == 30, "Incomplete molecular diagnostic")
    for path, digest in SOURCES.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "Changed source during diagnostic")
    groups = []
    for arm in ("product", "additive", "pair_mlp"):
        group = [row for row in rows if row["arm"] == arm]
        groups.append(dict(arm=arm, candidates=len(group), fitting_mse_lower=sum(row["train_mse_difference"] < 0 for row in group),
                           validation_mse_higher=sum(row["val_mse_difference"] > 0 for row in group),
                           mean_fitting_mse_difference=statistics.mean(row["train_mse_difference"] for row in group),
                           mean_validation_mse_difference=statistics.mean(row["val_mse_difference"] for row in group)))
    output = HERE / "molecular_training_behavior.csv"
    receipt = HERE / "molecular_training_behavior.json"
    require(not output.exists() and not receipt.exists(), "Preserve the existing diagnostic")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidates=30, groups=groups,
                  terminal_validation_reproduced_max_delta=max(row["logged_terminal_validation_max_delta"] for row in rows),
                  all_candidates_executed_175_updates=True, all_terminal_outputs_nonzero=True, all_final_batch_sizes_verified=True,
                  source_files_unchanged=True, sources=[dict(path=path, sha256=digest) for path, digest in sorted(SOURCES.items())],
                  output=dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
                  source_program=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                  new_fits=0, test_predictions_generated=0, selection_changes=0,
                  interpretation_scope="Direct equation replay of all terminal molecular candidates on fitting/validation inputs. Describes fitting versus validation behavior, not a causal diagnosis of overfitting, optimization, representation sufficiency or multiplication benefit.")
    receipt.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}))


if __name__ == "__main__":
    main()
