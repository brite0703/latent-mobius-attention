"""Bounded verification of one audited, hand-assigned native N20 witness.

No optimizer, backward pass, training, parameter tuning, or architecture patch.
Only the saved, frozen forward protocol is executed. Failures are retained.
"""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import hashlib
import csv
import math
import time
import traceback
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    assert not path.exists(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preserved_inputs(manifest):
    bad = []
    for item in manifest:
        p = Path(item["path"])
        if not p.is_file() or sha(p) != item["sha256"]:
            bad.append(str(p))
    return bad


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F

    started = time.monotonic()
    result = dict(started_utc=now(), purpose="native_representability_verification",
                  training_runs=0, optimizer_steps=0, dtypes={}, failures=[])
    try:
        proto = json.loads((HERE / "evidence/protocol_frozen.json").read_text())
        lock = json.loads((HERE / "evidence/protocol_lock.json").read_text())
        for item in lock["protocol_files"]:
            assert sha(HERE / item["relative_path"]) == item["sha256"], item
        assert not preserved_inputs(lock["preserved_inputs"]), "Pre-execution identity mismatch"
        assert torch.cuda.is_available(), "Authorized GPU required; no CPU substitution"
        torch.set_num_threads(proto["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        root = HERE / "native_original_sources"
        par = root / "LMA/revision_2026/reviewer_completion_2026_09_08/synthetic_parity"
        for p in [root / "LMA/revision_2026",
                  root / "LMA/revision_2026/neural_reviewer_study_2026_09_07", par]:
            sys.path.insert(0, str(p))
        pm = import_file("witness_original_parity_models", par / "parity_models.py")
        pc = import_file("witness_original_parity_common", par / "parity_common.py")
        recipe = import_file("audited_n20_assignment", HERE / "input/witness/assign_native_n20_witness.py")
        actual_lma = Path(pm.source.__file__).resolve()
        assert actual_lma == (root / "LMA/lma.py").resolve()
        assert sha(actual_lma) == proto["required_hashes"]["lma.py"]
        assert sha(par / "parity_models.py") == proto["required_hashes"]["parity_models.py"]
        assert sha(par / "parity_common.py") == proto["required_hashes"]["parity_common.py"]
        for klass in [pm.source.LMANetwork, pm.source.LatentMobiusAttention]:
            assert Path(klass.forward.__code__.co_filename).resolve() == actual_lma
        result["actually_imported_sources"] = {
            "lma.py": {"path": str(actual_lma), "sha256": sha(actual_lma)},
            "parity_models.py": {"path": str(par / "parity_models.py"),
                                  "sha256": sha(par / "parity_models.py")},
            "parity_common.py": {"path": str(par / "parity_common.py"),
                                  "sha256": sha(par / "parity_common.py")},
            "assignment.py": {"path": str(HERE / "input/witness/assign_native_n20_witness.py"),
                               "sha256": sha(HERE / "input/witness/assign_native_n20_witness.py")}}
        result["resources"] = {
            "python": sys.version, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cpu_threads": torch.get_num_threads()}
        write_json(HERE / "evidence/actual_start.json",
                   {"started_utc": now(), "protocol_sha256": sha(HERE / "evidence/protocol_frozen.json"),
                    "verification_script_sha256": sha(Path(__file__)),
                    "training_runs": 0, "optimizer_steps": 0, "device": result["resources"]["device"]})
        print(json.dumps({"actual_start": now(), "device": result["resources"]["device"],
                          "training_runs": 0}), flush=True)

        perm_generator = torch.Generator(device="cpu").manual_seed(proto["permutation_seed"])
        permutations = [[torch.randperm(20, generator=perm_generator).tolist()
                         for _ in range(proto["permutations_per_count"])] for _ in range(21)]
        write_json(HERE / "evidence/permutation_indices.json",
                   {"seed": proto["permutation_seed"], "permutations_by_count": permutations,
                    "duplicates_retained": True})
        rows = []
        canonical_outputs = {}
        for dtype_name in proto["dtypes_in_order"]:
            assert time.monotonic() - started < proto["maximum_wall_seconds"]
            dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
            torch.manual_seed(proto["fresh_model_seed"])
            # Crucial: construct and convert the fresh source-native model before assignment.
            model = pm.build_model(proto["fresh_model_seed"], "lma3", 20).to(
                device="cuda", dtype=dtype)
            assert type(model) is pm.source.LMANetwork
            assert model.use_positional is False
            modules_before = [(name, id(m), type(m)) for name, m in model.named_modules()]
            params_before = [(name, id(p), tuple(p.shape), p.requires_grad)
                             for name, p in model.named_parameters()]
            buffers_before = {n: t.detach().clone() for n, t in model.named_buffers()}
            combos_before = {r: t.detach().clone() for r, t in model.layers[0].combos.items()}
            eps_before = {n: m.eps for n, m in model.named_modules()
                          if isinstance(m, torch.nn.LayerNorm)}
            assert sum(p.numel() for p in model.parameters()) == 12525
            assert [len(combos_before[r]) for r in (1, 2, 3)] == [8, 28, 56]
            assert all(p.requires_grad for p in model.parameters())
            metadata = recipe.assign_native_n20_witness(model)
            assert modules_before == [(n, id(m), type(m)) for n, m in model.named_modules()]
            assert params_before == [(n, id(p), tuple(p.shape), p.requires_grad)
                                     for n, p in model.named_parameters()]
            assert eps_before == {n: m.eps for n, m in model.named_modules()
                                  if isinstance(m, torch.nn.LayerNorm)}
            assert all(torch.equal(t, dict(model.named_buffers())[n])
                       for n, t in buffers_before.items())
            assert all(torch.equal(t, model.layers[0].combos[r]) for r, t in combos_before.items())
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            assert all(p.grad is None for p in model.parameters())
            state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
            checkpoint = HERE / ("output/native_n20_hand_assigned_" + dtype_name + ".pt")
            assert not checkpoint.exists()
            torch.save({"kind": "hand_assigned_representability_witness_NOT_TRAINED",
                        "model": "source.LMANetwork(20,32,8,3,16,1,False)",
                        "dtype": dtype_name, "parameter_count": 12525,
                        "metadata": metadata, "state_dict": state}, checkpoint)
            canonical = pc.canonical(20, device="cuda", dtype=dtype)
            permutation_input = torch.stack([
                canonical[c, torch.tensor(perm, device="cuda", dtype=torch.long)]
                for c in range(21) for perm in permutations[c]])
            counts = np.repeat(np.arange(21), proto["permutations_per_count"])
            for x in [canonical, permutation_input]:
                # Native seq_len is not enforced; verifier explicitly enforces the proven domain.
                assert x.ndim == 2 and x.shape[1] == 20
                assert bool(torch.isfinite(x).all())
                assert bool(((x == 0) | (x == 1)).all())
            assert torch.equal(canonical.sum(1).long(), torch.arange(21, device="cuda"))
            assert np.array_equal(permutation_input.sum(1).long().cpu().numpy(), counts)
            labels = torch.arange(21, device="cuda", dtype=torch.long) % 2
            p_labels = torch.as_tensor(counts % 2, device="cuda", dtype=torch.long)
            captures = []
            handle = model.layers[0].register_forward_hook(
                lambda mod, inp, out: captures.append(out.detach().cpu().double()))
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gpu_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(canonical)
                permutation_outputs = model(permutation_input)
            torch.cuda.synchronize()
            forward_wall = time.perf_counter() - gpu_start
            handle.remove()
            assert outputs.shape == (21, 2)
            assert permutation_outputs.shape == (21 * proto["permutations_per_count"], 2)
            assert bool(torch.isfinite(outputs).all()) and bool(torch.isfinite(permutation_outputs).all())
            assert len(captures) == 2 and all(bool(torch.isfinite(t).all()) for t in captures)
            out = outputs.double().cpu().numpy()
            pout = permutation_outputs.double().cpu().numpy()
            canonical_outputs[dtype_name] = out
            margins = (2 * (np.arange(21) % 2) - 1) * (out[:, 1] - out[:, 0])
            pmargins = (2 * (counts % 2) - 1) * (pout[:, 1] - pout[:, 0])
            ce = np.logaddexp(0.0, -margins)
            pce = np.logaddexp(0.0, -pmargins)
            max_perm_error = float(np.max(np.abs(pout - out[counts])))
            tol = proto["floating_tolerances"][dtype_name]
            ideal_bound = metadata["signed_binary_margin_lower_bound_ideal_real"]
            checks = {
                "finite_parameters_and_outputs": True,
                "canonical_correct_21_of_21": bool(np.array_equal(np.argmax(out, 1), np.arange(21) % 2)),
                "permutation_sample_correct": bool(np.array_equal(np.argmax(pout, 1), counts % 2)),
                "positive_canonical_margins": bool((margins > 0).all()),
                "positive_permutation_margins": bool((pmargins > 0).all()),
                "minimum_margin_near_ideal_bound": bool(min(margins.min(), pmargins.min()) >= ideal_bound - tol),
                "permutation_error_within_frozen_tolerance": max_perm_error <= tol,
                "parameters_unchanged_after_forwards": all(
                    torch.equal(t, model.state_dict()[n].detach().cpu()) for n, t in state.items()),
                "no_gradients_or_optimizer_steps": all(p.grad is None for p in model.parameters()),
                "module_parameter_buffer_epsilon_identity": True}
            details = {
                "checks": checks, "checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint),
                "assignment_metadata": metadata, "parameter_count": 12525,
                "parameter_maximum_absolute": max(float(p.detach().abs().max()) for p in model.parameters()),
                "layernorm_epsilons": eps_before, "memory_slots_by_order": [8, 28, 56],
                "memory_slots_total": 92, "order_gates": model.layers[0].order_gates.detach().cpu().tolist(),
                "canonical_input_sha256": hashlib.sha256(canonical.cpu().numpy().tobytes()).hexdigest(),
                "permutation_input_sha256": hashlib.sha256(permutation_input.cpu().numpy().tobytes()).hexdigest(),
                "canonical_minimum_signed_margin": float(margins.min()),
                "permutation_minimum_signed_margin": float(pmargins.min()),
                "canonical_maximum_cross_entropy": float(ce.max()),
                "permutation_maximum_cross_entropy": float(pce.max()),
                "canonical_mean_cross_entropy": float(ce.mean()),
                "permutation_maximum_absolute_logit_difference": max_perm_error,
                "count_state_maximum_absolute_ideal_real_error": float(np.max(np.abs(
                    captures[0][:, 0, 2].numpy() - np.asarray(metadata["count_states_ideal_real"])))),
                "count_state_token_maximum_spread": float(
                    (captures[0][:, :, 2].max(1).values - captures[0][:, :, 2].min(1).values).max()),
                "native_original_metrics": pc.metrics(np.arange(21) % 2, out),
                "ideal_binomial_population_metrics_from_21_orbit_representatives":
                    pc.population_metrics(20, out),
                "forward_wall_seconds": forward_wall,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "module_inventory": [{"name": n, "class": type(m).__module__ + "." + type(m).__name__}
                                     for n, m in model.named_modules()],
                "parameter_inventory": [{"name": n, "shape": list(p.shape), "requires_grad": p.requires_grad}
                                        for n, p in model.named_parameters()]}
            result["dtypes"][dtype_name] = details
            for c in range(21):
                rows.append(dict(dtype=dtype_name, domain="canonical", count=c, sample_index=0,
                                 label=c % 2, logit0=float(out[c, 0]), logit1=float(out[c, 1]),
                                 signed_margin=float(margins[c]), cross_entropy=float(ce[c])))
            for i, c in enumerate(counts):
                rows.append(dict(dtype=dtype_name, domain="deterministic_permutation", count=int(c),
                                 sample_index=i % proto["permutations_per_count"], label=int(c % 2),
                                 logit0=float(pout[i, 0]), logit1=float(pout[i, 1]),
                                 signed_margin=float(pmargins[i]), cross_entropy=float(pce[i])))
            np.savez(HERE / ("output/layer_count_states_" + dtype_name + ".npz"),
                     canonical_layer_outputs=captures[0].numpy(),
                     permutation_layer_outputs=captures[1].numpy())
            with (HERE / "output/all_native_outputs.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({"dtype": dtype_name, "checks": checks,
                              "minimum_margin": details["canonical_minimum_signed_margin"],
                              "maximum_permutation_error": max_perm_error,
                              "forward_wall_seconds": forward_wall}), flush=True)
            failed = [n for n, v in checks.items() if not v]
            assert not failed, "Native results disagree; stop without tuning: " + ", ".join(failed)
            del model
        precision_error = float(np.max(np.abs(canonical_outputs["float32"] -
                                              canonical_outputs["float64"])))
        result["cross_precision_maximum_absolute_canonical_logit_difference"] = precision_error
        assert precision_error <= proto["floating_tolerances"]["float32"]
        bad = preserved_inputs(lock["preserved_inputs"])
        result["preservation_checks"] = {"checked_files": len(lock["preserved_inputs"]),
                                         "changed_or_missing": bad}
        assert not bad, "Original or first-pilot file changed"
        for item in lock["protocol_files"]:
            assert sha(HERE / item["relative_path"]) == item["sha256"], item
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL_STOPPED_NO_TUNING"
        result["failures"].append({"type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc()})
    result["ended_utc"] = now()
    result["total_wall_seconds"] = time.monotonic() - started
    write_json(HERE / "evidence/native_verification.json", result)
    print(json.dumps({"status": result["status"], "wall_seconds": result["total_wall_seconds"],
                      "training_runs": 0}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
