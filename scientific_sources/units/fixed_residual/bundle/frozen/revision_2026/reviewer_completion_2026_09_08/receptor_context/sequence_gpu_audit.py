"""Discarded largest-input CUDA checks and measured sequence-study feasibility."""
from datetime import datetime, timezone
import hashlib
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import sequence_execution as execution
from sequence_data import SequenceStore
from sequence_data import MAX_PADDED_TOKENS, MAX_RECORDS, MANIFEST
import sequence_models as models
from sequence_engine_audit import make_toy, equal_tree

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]/"data/lp_pdbbind/tensors_reconstructed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_inputs():
    for filename in ("sequence_model_cpu_audit.json", "sequence_engine_cpu_audit.json", "sequence_metadata.json"):
        audit = json.loads((HERE/filename).read_text(encoding="utf-8"))
        assert audit["passed"]
        for item in audit["sources"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True)
    other_python = []
    for line in query.stdout.splitlines():
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip().isdigit() and int(parts[0].strip()) != os.getpid() and "python" in parts[1].lower():
            other_python.append(line.strip())
    if other_python:
        raise RuntimeError("Other Python CUDA work is active; this audit must wait: "+repr(other_python))


def full_cohort_envelopes(store, folder):
    """Use length/atom metadata and artificial inputs, never held-out targets."""
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(store.by_id) or {row["pdbid"] for row in rows} != set(store.by_id):
        raise ValueError("Shape metadata identifiers changed")
    bounds = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        bounds[split] = dict(records=len(selected),
            maximum_ligand_atoms=max(int(row["atoms"]) for row in selected),
            maximum_chain_length=max(store.sizes[row["pdbid"]][1] for row in selected),
            maximum_supplied_chains=max(store.sizes[row["pdbid"]][0] for row in selected),
            maximum_record_padded_tokens=max(math.prod(store.sizes[row["pdbid"]]) for row in selected))
        if bounds[split]["maximum_record_padded_tokens"] > MAX_PADDED_TOKENS:
            raise ValueError("A full-cohort singleton exceeds the frozen token cap")
    maximum_atoms = max(row["maximum_ligand_atoms"] for row in bounds.values())
    maximum_length = max(row["maximum_chain_length"] for row in bounds.values())
    maximum_chains = max(row["maximum_supplied_chains"] for row in bounds.values())
    specs = [("record_cap", MAX_RECORDS, 1, MAX_PADDED_TOKENS//MAX_RECORDS),
             ("longest_chain", min(MAX_RECORDS, MAX_PADDED_TOKENS//maximum_length), 1, maximum_length),
             ("chain_count", 1, maximum_chains, min(maximum_length, MAX_PADDED_TOKENS//maximum_chains))]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"artificial_shape_envelopes.csv"
    fixture_rows, case_ids = [], {}
    for name, records, chains, length in specs:
        if records < 1 or length < 1 or records*chains*length > MAX_PADDED_TOKENS:
            raise ValueError("Invalid full-cohort resource envelope")
        ids = [f"artificial_{name}_{index}" for index in range(records)]
        case_ids[name] = ids
        fixture_rows.extend(dict(pdbid=key, split="shape_fixture", sequence=":".join(["A"*length]*chains)) for key in ids)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pdbid", "split", "sequence"])
        writer.writeheader()
        writer.writerows(fixture_rows)
    fixture_store = SequenceStore(path)
    blobs = {}
    for name, ids in case_ids.items():
        count = len(ids)
        mask = torch.ones(count, maximum_atoms, dtype=torch.bool)
        blobs[name] = execution.indexed_blob(dict(ids=ids, y=torch.zeros(count), mask=mask,
            X=torch.ones(count, maximum_atoms, 53),
            adj=torch.eye(maximum_atoms).unsqueeze(0).repeat(count, 1, 1)))
        assert list(fixture_store.chunks(ids)) == [ids]
    guard = dict(bounds_by_split=bounds, heldout_dimension_maxima_within_fitting_maxima=all(
        bounds[split][key] <= bounds["train"][key] for split in ("val", "test")
        for key in ("maximum_ligand_atoms", "maximum_chain_length", "maximum_supplied_chains", "maximum_record_padded_tokens")),
        artificial_envelopes=[dict(name=name, records=records, chains_per_record=chains,
            residues_per_chain=length, ligand_atoms_per_record=maximum_atoms,
            padded_tokens=records*chains*length) for name, records, chains, length in specs],
        heldout_tensor_files_loaded=False, affinity_targets_parsed=False,
        scope="Metadata dimension checks over all splits, followed by artificial all-A chains and constant ligand tensors at conservative allocation envelopes. These are discarded resource tests, not held-out affinity predictions or an eligibility amendment.")
    return fixture_store, blobs, guard


def run():
    assert_inputs()
    execution.configure("cuda")
    store = SequenceStore()
    train = execution.indexed_blob(torch.load(DATA/"pdbbind_train.pt", map_location="cpu", weights_only=True))
    assert len(train["ids"]) == 7384 and all(store.by_id[key]["split"] == "train" for key in train["ids"])
    mean, sd = execution.scalers(train)
    ranked = sorted(train["ids"], key=lambda key: (-store.sizes[key][0]*store.sizes[key][1], key))[:256]
    anchors = sorted({max(train["ids"], key=lambda key: store.sizes[key][1]),
                      max(train["ids"], key=lambda key: store.sizes[key][0]), ranked[0],
                      train["ids"][int(train["mask"].sum(1).argmax())]})
    ordinary = train["ids"][:256]
    toy_folder = HERE/"tmp/sequence_gpu_audit"
    toy_folder.mkdir(parents=True, exist_ok=True)
    envelope_store, envelope_blobs, cohort_guard = full_cohort_envelopes(store, toy_folder)
    toy_store, toy = make_toy(toy_folder)
    toy_mean, toy_sd = execution.scalers(toy)
    results = []
    for setting, head in models.configurations():
        runtime = execution.make_runtime(42, setting, head, .001, device="cuda")
        model, optimizer = runtime["model"], runtime["optimizer"]
        for key in anchors:
            execution.effective_step(model, optimizer, train, store, [key], mean, sd)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        began = time.perf_counter()
        worst = execution.effective_step(model, optimizer, train, store, ranked, mean, sd)
        torch.cuda.synchronize()
        worst_seconds = time.perf_counter()-began
        worst_peak = torch.cuda.max_memory_allocated()
        # Five separately timed effective batches; disposal prevents retained selection.
        times = []
        for _ in range(5):
            began = time.perf_counter()
            step = execution.effective_step(model, optimizer, train, store, ordinary, mean, sd)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-began)
        small = execution.indexed_blob({key: train[key][:64] for key in ("X", "mask", "adj", "y", "ids")})
        began = time.perf_counter()
        evaluated = execution.predict(model, small, store, mean, sd)
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter()-began
        assert len(evaluated) == 64 and np.isfinite(evaluated).all()
        envelope_results = []
        for name, blob in envelope_blobs.items():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            artificial_prediction = execution.predict(model, blob, envelope_store, mean, sd)
            torch.cuda.synchronize()
            assert len(artificial_prediction) == len(blob["ids"]) and np.isfinite(artificial_prediction).all()
            envelope_results.append(dict(name=name, records=len(blob["ids"]),
                seconds=time.perf_counter()-began, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        # Check actual CUDA continuation with the same serialized execution path.
        def fresh():
            return execution.make_runtime(42, setting, head, .001, device="cuda", epochs=2)
        full = fresh()
        for _ in range(2):
            execution.finish_epoch(full, toy, toy, toy_store, toy_mean, toy_sd, effective_batch=5, max_records=2)
        partial = fresh()
        execution.finish_epoch(partial, toy, toy, toy_store, toy_mean, toy_sd, effective_batch=5, max_records=2)
        path = toy_folder/f"{setting}_{head}.pt"
        execution.save_snapshot(path, partial, toy_mean, toy_sd)
        restored = fresh()
        execution.restore_snapshot(path, restored, toy_mean, toy_sd)
        execution.finish_epoch(restored, toy, toy, toy_store, toy_mean, toy_sd, effective_batch=5, max_records=2)
        equal_tree(full["model"].state_dict(), restored["model"].state_dict())
        equal_tree(full["optimizer"].state_dict(), restored["optimizer"].state_dict())
        equal_tree(full["scheduler"].state_dict(), restored["scheduler"].state_dict())
        equal_tree(full["generator"].get_state(), restored["generator"].get_state())
        np.testing.assert_array_equal(execution.predict(full["model"], toy, toy_store, toy_mean, toy_sd),
                                      execution.predict(restored["model"], toy, toy_store, toy_mean, toy_sd))
        results.append(dict(setting=setting, head=head, parameters=model.parameter_counts(), anchor_records=anchors,
            worst_256_step=worst, worst_256_seconds=worst_seconds, baseline_allocated_bytes=before,
            worst_256_peak_allocated_bytes=worst_peak, ordinary_256_seconds=times,
            ordinary_256_median_seconds=float(np.median(times)), ordinary_step=step,
            evaluation_64_seconds=evaluation_seconds, artificial_full_cohort_envelopes=envelope_results,
            cuda_continuation_exact=True))
        print(json.dumps({key: value for key, value in results[-1].items() if key not in ("parameters", "worst_256_step", "ordinary_step")}), flush=True)
        del runtime, model, optimizer, full, partial, restored
        torch.cuda.empty_cache()
    paths = [HERE/name for name in ("sequence_gpu_audit.py", "sequence_execution.py", "sequence_model_cpu_audit.json",
        "sequence_engine_cpu_audit.json", "sequence_metadata.json", "sequence_protocol.md")]+[DATA/"pdbbind_train.pt", MANIFEST]
    output = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True, configurations=11, rows=results,
        full_cohort_shape_guard=cohort_guard,
        device=torch.cuda.get_device_name(), torch_version=torch.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        target_mean=mean, target_population_sd=sd, sources=[dict(path=str(path), sha256=sha(path)) for path in paths],
        qualification="All updates are discarded feasibility work. No retained candidate or validation selection exists. Timings include input collation/transfers, receptor encoding and actual microbatching, exclude initial dataset loading, and are local to this GPU. Five update timings are not a complete training-time estimate or equal-compute comparison.")
    (HERE/"sequence_gpu_audit.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(passed=True, configurations=11, all_cuda_continuations_exact=True)), flush=True)


if __name__ == "__main__":
    run()
