"""Frozen-weight, fitting-only readout diagnosis; no CUDA or optimizer use."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, sys
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "gate_readout"
sys.path.insert(0, str(GATE))
import gate_models as gm

DATA = gm.REVISION / "data/lp_pdbbind/tensors_reconstructed/pdbbind_train.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finish(model, h, mask, readout):
    layer = model.head.layers[0]
    valid = mask.unsqueeze(-1)
    value = layer.layer_norm(h + layer.W_out(readout)) * valid
    value = (value + model.head.ffn(value)) * valid
    pooled = value.sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
    return model.head.head(model.head.norm(pooled)).squeeze(-1)


def replay(model, x, mask, adj, intervention="full"):
    if intervention not in ("full", "first_only", "zero_higher"):
        raise ValueError(intervention)
    h = model.encoder(x, mask, adj)
    layer = model.head.layers[0]
    valid = mask.unsqueeze(-1)
    pi = F.softmax(layer.W_H(layer.W_k(h)), dim=-1)
    z = (pi * valid).transpose(1, 2) @ layer.W_v(h)
    q = layer.W_q(h)
    memories = []
    for order in range(1, layer.k + 1):
        legs = layer.interaction_projs[order - 1](z).chunk(order, dim=-1)
        tuples = getattr(layer, f"combos_{order}")
        values = torch.stack([legs[j][:, tuples[:, j], :] for j in range(order)])
        gamma = values.prod(0) if layer.feature_mode == "product" else values.mean(0)
        memories.append(layer.interaction_mlps[order - 1](gamma))
    separate = isinstance(layer, gm.SeparatedOrders)
    gates = [layer.order_gates[i] for i in range(layer.k)]
    if intervention == "zero_higher":
        gates = [gates[0]] + [g * 0 for g in gates[1:]]
    if intervention == "first_only":
        memories, gates = memories[:1], gates[:1]
    diagnostics = {}
    if separate:
        branches = []
        for i, memory in enumerate(memories):
            used = gates[0] * memory if i == 0 else memory
            attention = F.softmax(q @ used.transpose(1, 2) / math.sqrt(layer.d_model), -1)
            branch = attention @ used
            branches.append(branch if i == 0 else gates[i] * branch)
            entropy = -(torch.special.xlogy(attention, attention)).sum(-1)
            diagnostics[f"entropy_order{i+1}"] = (entropy * mask).sum(1) / mask.sum(1) / math.log(memory.shape[1])
        readout = torch.stack(branches).sum(0)
    else:
        used = torch.cat([g * m for g, m in zip(gates, memories)], 1)
        attention = F.softmax(q @ used.transpose(1, 2) / math.sqrt(layer.d_model), -1)
        readout = attention @ used
        entropy = -torch.special.xlogy(attention, attention).sum(-1)
        diagnostics["joint_entropy"] = (entropy * mask).sum(1) / mask.sum(1) / math.log(used.shape[1])
        if len(memories) == 2:
            mass = attention[:, :, memories[0].shape[1]:].sum(-1)
            diagnostics["order2_attention_mass"] = (mass * mask).sum(1) / mask.sum(1)
    average = (pi * valid).sum(1) / mask.sum(1, keepdim=True)
    variation = ((pi - average[:, None]).square() * valid).sum((1, 2)) / mask.sum(1)
    diagnostics["within_graph_routing_variation"] = variation
    for i, memory in enumerate(memories):
        denominator = memory.square().mean((1, 2))
        numerator = (memory - memory.mean(1, keepdim=True)).square().mean((1, 2))
        diagnostics[f"memory{i+1}_relative_variation"] = numerator / denominator.clamp_min(torch.finfo(memory.dtype).tiny)
    return finish(model, h, mask, readout), diagnostics


def audit():
    generator = torch.Generator().manual_seed(92137)
    x = torch.randn(3, 7, 53, generator=generator, dtype=torch.float64)
    mask = torch.arange(7)[None] < torch.tensor([7, 5, 3])[:, None]
    adj = torch.eye(7, dtype=torch.float64).repeat(3, 1, 1)
    permutation = torch.tensor([3, 1, 6, 0, 4, 2, 5])
    rows = []
    for name in gm.HEADS:
        model = gm.build(42, name, 1).double().eval()
        baseline = model(x, mask, adj)
        reconstructed = replay(model, x, mask, adj)[0]
        torch.testing.assert_close(baseline, reconstructed, rtol=1e-10, atol=1e-11)
        other = replay(model, x[:, permutation], mask[:, permutation],
                       adj[:, permutation][:, :, permutation])[0]
        torch.testing.assert_close(reconstructed, other, rtol=1e-10, atol=1e-11)
        xp = F.pad(x, (0, 0, 0, 3), value=7.25)
        mp = F.pad(mask, (0, 3))
        ap = F.pad(adj, (0, 3, 0, 3))
        torch.testing.assert_close(reconstructed, replay(model, xp, mp, ap)[0], rtol=1e-10, atol=1e-11)
        derivative_error = None
        layer = model.head.layers[0]
        if layer.k == 2:
            analytic = float(torch.autograd.grad(baseline.square().mean(), layer.order_gates)[0][1])
            saved = float(layer.order_gates[1])
            outputs = []
            for offset in (1e-5, -1e-5):
                with torch.no_grad():
                    layer.order_gates[1] = saved + offset
                    outputs.append(float(replay(model, x, mask, adj)[0].square().mean()))
            with torch.no_grad():
                layer.order_gates[1] = saved
            finite = (outputs[0] - outputs[1]) / 2e-5
            derivative_error = abs(finite - analytic)
            assert math.isclose(finite, analytic, rel_tol=2e-5, abs_tol=2e-8), (name, finite, analytic)
            dropped = replay(model, x, mask, adj, "first_only")[0]
            zeroed = replay(model, x, mask, adj, "zero_higher")[0]
            if name.startswith("separated"):
                torch.testing.assert_close(dropped, zeroed, rtol=0, atol=0)
            else:
                assert float((dropped - zeroed).abs().max()) > 1e-5
        rows.append(dict(head=name, native_replay_max_delta=float((baseline-reconstructed).abs().max()),
                         derivative_absolute_error=derivative_error))
    return dict(passed=True, configurations=7, rows=rows,
                checks=["Native versus independent replay", "Permutation and padding invariance",
                        "Native autograd versus replay central finite difference",
                        "Separate off-state equality and joint-normalization non-equivalence"])


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(n=len(a), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,
                minimum=float(a.min()), maximum=float(a.max()))


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination = HERE / "diagnosis.json"
    if destination.exists():
        receipt = read(destination)
        assert receipt["complete"]
        for item in receipt["sources"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
        for item in receipt["artifacts"]:
            assert sha(HERE / item["path"]) == item["sha256"]
        print(json.dumps({"preserved_existing_diagnosis": True}))
        return
    checks = audit()
    source_paths = {Path(__file__), HERE/"protocol.md", DATA, GATE/"implementation_lock.json",
                    GATE/"selection_lock.json", GATE/"final_audit.json"}
    lock = read(GATE/"implementation_lock.json")
    for source in lock["files"]:
        assert sha(source["path"]) == source["sha256"], source["path"]
        source_paths.add(Path(source["path"]))
    assert read(GATE/"final_audit.json")["passed"]
    selection = read(GATE/"selection_lock.json")
    for candidate in selection["candidate_records"]:
        path = GATE/candidate["path"]
        assert sha(path) == candidate["sha256"]
        source_paths.add(path)
    blob = torch.load(DATA, map_location="cpu", weights_only=True)
    ranked = sorted(range(len(blob["ids"])), key=lambda i: hashlib.sha256(
        ("LMA-readout-diagnosis-v1|" + blob["ids"][i]).encode()).hexdigest())[:256]
    index = torch.tensor(ranked)
    ids = [blob["ids"][i] for i in ranked]
    mask = blob["mask"][index]
    end = int(mask.sum(1).max())
    x, mask, adj = blob["X"][index, :end], mask[:, :end], blob["adj"][index, :end, :end]
    truth = blob["y"][index].double().numpy()
    rows, graph_rows = [], []
    selected = selection["selections"]
    assert len(selected) == 35 and {(r["head"], r["seed"]) for r in selected} == {
        (h, s) for h in gm.HEADS for s in range(42,47)}
    for choice in selected:
        assert choice["selected_id"] is not None
        candidate = read(GATE/"candidates"/(choice["selected_id"]+".json"))
        path = GATE/candidate["checkpoint"]
        assert sha(path) == choice["checkpoint_sha256"] == candidate["checkpoint_sha256"]
        source_paths.add(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["spec"] == candidate["spec"]
        model = gm.build(choice["seed"], choice["head"], 1).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.no_grad():
            native = model(x, mask, adj)
            full, diagnosis = replay(model, x, mask, adj)
            torch.testing.assert_close(native, full, rtol=3e-5, atol=3e-5)
            first = replay(model, x, mask, adj, "first_only")[0]
            zero = replay(model, x, mask, adj, "zero_higher")[0]
        for name, value in model.state_dict().items():
            assert torch.equal(value, checkpoint["state_dict"][name])
        def original_scale(p):
            return p.double().numpy()*checkpoint["target_sd"]+checkpoint["target_mean"]
        native_values, full_values, first_values, zero_values = map(original_scale, (native, full, first, zero))
        assert all(np.isfinite(v).all() for v in (native_values, full_values, first_values, zero_values))
        per_graph = {k:v.double().numpy() for k,v in diagnosis.items()}
        assert all(np.isfinite(v).all() for v in per_graph.values())
        if choice["head"].startswith("separated") or choice["head"] == "legacy1":
            np.testing.assert_array_equal(first_values, zero_values)
        row = dict(head=choice["head"], seed=choice["seed"], selected_id=choice["selected_id"],
                   checkpoint_sha256=sha(path), gates=model.head.layers[0].order_gates.detach().tolist(),
                   records=256, archived_validation_rmse=choice["validation_rmse"],
                   cpu_native_replay_max_normalized_delta=float((native-full).abs().max()),
                   fitting_rmse=float(np.sqrt(np.mean((full_values-truth)**2))),
                   first_only_prediction_change_rms=float(np.sqrt(np.mean((first_values-full_values)**2))),
                   zero_higher_prediction_change_rms=float(np.sqrt(np.mean((zero_values-full_values)**2))),
                   first_only_minus_full_fitting_mse=float(np.mean((first_values-truth)**2-(full_values-truth)**2)),
                   zero_higher_minus_full_fitting_mse=float(np.mean((zero_values-truth)**2-(full_values-truth)**2)),
                   diagnostics={k:describe(v) for k,v in per_graph.items()})
        rows.append(row)
        for i, key in enumerate(ids):
            graph_rows.append(dict(head=choice["head"], seed=choice["seed"], id=key, truth=float(truth[i]),
                full_prediction=float(full_values[i]), first_only_prediction=float(first_values[i]),
                zero_higher_prediction=float(zero_values[i]), **{k:float(v[i]) for k,v in per_graph.items()}))
        print(json.dumps({"checked":len(rows),"total":35,"head":choice["head"],"seed":choice["seed"],
                          "first_only_change_rms":row["first_only_prediction_change_rms"]}), flush=True)
    assert not torch.cuda.is_initialized()
    fields = list(dict.fromkeys(k for row in graph_rows for k in row))
    graph_path = HERE/"fitting_record_diagnostics.csv"
    with graph_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(graph_rows)
    audit_path = HERE/"cpu_replay_audit.json"
    write(audit_path, checks)
    metrics = ["fitting_rmse", "first_only_prediction_change_rms", "zero_higher_prediction_change_rms",
               "first_only_minus_full_fitting_mse", "zero_higher_minus_full_fitting_mse"]
    groups = []
    for head in gm.HEADS:
        subset = [row for row in rows if row["head"]==head]
        group = dict(head=head, **{key:describe([r[key] for r in subset]) for key in metrics})
        group["diagnostics"] = {key:describe([r["diagnostics"][key]["mean"] for r in subset])
                                for key in subset[0]["diagnostics"]}
        groups.append(group)
    result = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(), input_ids=ids,
        sample_rule="Lowest 256 SHA256('LMA-readout-diagnosis-v1|' + fitting ID)",
        fitting_records=256, selected_predictors=35, per_record_rows=len(graph_rows),
        cuda_initialized=False, cpu_threads=1, optimizer_created=False, validation_or_test_tensors_loaded=False,
        independent_replay_audit=checks, rows=rows, groups=groups,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(source_paths)],
        artifacts=[dict(path=p.name,sha256=sha(p)) for p in (graph_path,audit_path)],
        qualification="Exploratory dependence diagnostics on fitting records, with fixed validation-selected weights. Prediction sensitivity is not generalization benefit, causal training attribution, molecular Bayes risk or a new model-selection result.")
    write(destination,result)
    print(json.dumps({"complete":True,"predictors":35,"fitting_records":256,"record_rows":len(graph_rows),
                      "cuda_initialized":False}),flush=True)


if __name__ == "__main__":
    main()
