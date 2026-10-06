"""Independent formula/gradient/data checks and discarded largest-batch update."""
import argparse
import itertools
import json
import math
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss
import parity_common as c
import parity_models as pm
import campaign

HERE = c.HERE


def close(a, b, atol=1e-10, rtol=1e-9):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a-b).abs().max()) if a.numel() else 0.


def gradient_comparison(model, a, b):
    weights = torch.linspace(-.9, 1.3, a.numel(), dtype=a.dtype, device=a.device).reshape_as(a)
    params = list(model.parameters())
    ga = torch.autograd.grad((a*weights).sum(), params)
    gb = torch.autograd.grad((b*weights).sum(), params)
    return max(close(x, y, 2e-9, 2e-8) for x, y in zip(ga, gb))


def independent_layer(layer, x):
    linear = nn.functional.linear
    q = linear(x, layer.W_q.weight, layer.W_q.bias)
    key = linear(x, layer.W_k.weight, layer.W_k.bias)
    values = linear(x, layer.W_v.weight, layer.W_v.bias)
    route = linear(key, layer.W_H.weight, layer.W_H.bias).softmax(-1)
    # Explicit bucket sums and combination loop, distinct from source bmm/gather.
    buckets = torch.stack([(route[:, :, j:j+1]*values).sum(1) for j in range(layer.M)], 1)
    memories = []
    for r in range(1, layer.k+1):
        p = layer.interaction_projs[r-1]
        projected = linear(buckets, p.weight, p.bias).chunk(r, -1)
        for indices in itertools.combinations(range(layer.M), r):
            feature = torch.ones_like(projected[0][:, 0])
            for leg, bucket in enumerate(indices):
                feature = feature*projected[leg][:, bucket]
            memories.append(layer.order_gates[r-1]*layer.interaction_mlps[r-1](feature))
    memory = torch.stack(memories, 1)
    scores = torch.einsum("bnd,bld->bnl", q, memory)/math.sqrt(layer.d_model)
    output = torch.einsum("bnl,bld->bnd", scores.softmax(-1), memory)
    return layer.layer_norm(x+linear(output, layer.W_out.weight, layer.W_out.bias))


def prepare_data():
    rows = []
    for n in c.LENGTHS:
        for seed in c.SEEDS:
            generated = c.generate(n, seed)
            assert generated["x"].dtype == np.uint8
            assert set(np.unique(generated["x"])) == {0, 1}
            # Independent XOR reduction versus sum modulo 2.
            assert np.array_equal(generated["y"], np.bitwise_xor.reduce(generated["x"], 1))
            path = c.data_path(n, seed)
            if path.exists():
                with np.load(path) as old:
                    for key in generated:
                        assert np.array_equal(old[key], generated[key])
            else:
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **generated)
            splits, vectors = {}, {}
            for split, sl in c.SLICES.items():
                x, y, count = (generated[key][sl] for key in ("x", "y", "count"))
                packed = np.packbits(x, axis=1)
                keys = [row.tobytes() for row in packed]
                vectors[split] = keys
                histogram = np.bincount(count, minlength=n+1)
                covered_mass = math.fsum(math.comb(n, j)/2**n for j in range(n+1) if histogram[j] > 0)
                splits[split] = dict(rows=len(x), class_counts=np.bincount(y, minlength=2).tolist(),
                    count_histogram=histogram.tolist(), observed_counts=int((histogram>0).sum()),
                    observed_count_population_probability_mass=covered_mass,
                    unique_exact_vectors=len(set(keys)), repeated_exact_vectors=len(x)-len(set(keys)))
            overlaps = {}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                first = set(vectors[a])
                overlaps[a+"_"+b] = dict(shared_unique_exact_vectors=len(first & set(vectors[b])),
                    second_split_rows_also_in_first=sum(v in first for v in vectors[b]))
            training = c.load_data(n, seed, "train")
            lookup, totals = c.count_lookup(n, training)
            assert np.array_equal(totals, np.asarray(splits["train"]["count_histogram"]))
            positives = [0]*(n+1)
            for label, count in zip(training["y"].numpy(), training["count"]):
                positives[int(count)] += int(label)
            expected = (np.asarray(positives)+1)/(totals+2)
            np.testing.assert_allclose(lookup[:, 1], np.log(expected), atol=1e-14, rtol=0)
            assert np.array_equal(lookup[totals == 0, 0], lookup[totals == 0, 1])
            rows.append(dict(n=n, seed=seed, path=str(path.relative_to(HERE)), sha256=c.sha(path),
                generator_seed=2026090803+1000*n+seed, splits=splits, exact_vector_overlap=overlaps,
                lookup_training_only_formula_checked=True))
    c.write_json(HERE/"data_manifest.json", dict(created_utc=c.now(), datasets=rows,
        generator="numpy.default_rng(seed).integers(0,2,size=(6000,N),dtype=uint8)",
        qualification="iid discrete draws may repeat; a held-out row is not necessarily an unseen unpositioned multiset"))
    return len(rows)


def metrics_audit():
    records = []
    for n in (2, 4, 6, 10):
        counts = np.arange(n+1)
        for mode in ("ties", "nonmonotone", "oracle"):
            score = np.zeros(n+1) if mode == "ties" else np.sin(counts*1.31)*2
            if mode == "oracle":
                score = (2*(counts % 2)-1)*2.
            logits = np.column_stack([np.zeros(n+1), score])
            x = np.asarray(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
            rows = x.sum(1)
            y = np.bitwise_xor.reduce(x, 1)
            p = 1/(1+np.exp(-score[rows]))
            reference_auc = roc_auc_score(y, score[rows])
            reference_ce = log_loss(y, np.column_stack([1-p, p]))
            pop = c.population_metrics(n, logits)
            assert abs(pop["auc"]-reference_auc) < 1e-13
            assert abs(pop["cross_entropy"]-reference_ce) < 1e-13
            assert pop["accuracy"] == float(((score[rows] > 0) == y).mean())
            if mode == "ties":
                assert pop["accuracy"] == .5 and pop["auc"] == .5
            if mode == "oracle":
                assert pop["accuracy"] == 1 and pop["auc"] == 1
            records.append(dict(n=n, mode=mode, population=pop))
    return records


def cpu():
    data_count = prepare_data()
    metric_checks = metrics_audit()
    checks = []
    # All retained parameter pairing, all declared seeds, independent of N.
    for seed in c.SEEDS:
        reference = pm.template(seed, 10).state_dict()
        for name in ["lma1", "lma2", "lma3", "lma3_clip1"]:
            model = pm.build_model(seed, name, 10)
            for key, value in model.state_dict().items():
                expected = reference[key][:len(value)] if key.endswith("order_gates") else reference[key]
                assert torch.equal(value, expected), (seed, name, key)
        a, b = pm.build_model(seed, "deepsets_ln", 10), pm.build_model(seed, "deepsets_plain", 10)
        for key, value in b.state_dict().items():
            assert torch.equal(value, a.state_dict()[key]), (seed, key)
        for name in ("transformer", "janossy2", "cp_pool"):
            model = pm.build_model(seed, name, 10)
            assert torch.equal(model.embedding.weight, reference["embedding.weight"])
            assert torch.equal(model.embedding.bias, reference["embedding.bias"])
    counts = {name: pm.parameter_count(pm.build_model(100, name, 10)) for name in pm.HEADS}
    assert counts["lma2"] == pm.SIZES["reference_total_parameters"]
    w, j, r = (pm.SIZES[k] for k in ("deepsets_width", "janossy_width", "cp_rank"))
    assert counts["deepsets_ln"] == 2*w*w+8*w+2
    assert counts["deepsets_plain"] == 2*w*w+6*w+2
    assert counts["janossy2"] == 2*j*j+72*j+66
    assert counts["cp_pool"] == 2274+65*r
    assert counts["lma3"] == counts["lma3_clip1"]
    for n in c.LENGTHS:
        x = c.canonical(n, dtype=torch.float64)
        for name in pm.HEADS:
            model = pm.build_model(100, name, n).double().eval()
            with torch.no_grad():
                expected = model(x)
                generator = torch.Generator().manual_seed(401+n)
                permutation = torch.randperm(n, generator=generator)
                permuted = model(x[:, permutation])
                split = torch.cat([model(part) for part in x.split(7)], 0)
                pd = close(expected, permuted, 2e-9, 2e-8)
                bd = close(expected, split, 2e-9, 2e-8)
                ordinary = c.load_data(n, 100, "train")["x"][:31].double()
                actual = model(ordinary)
                canonical = expected[ordinary.sum(1).long()]
                cd = close(actual, canonical, 2e-9, 2e-8)
            checks.append(dict(n=n, head=name, permutation_max_delta=pd, batch_max_delta=bd,
                               canonical_vs_sampled_rows_max_delta=cd))
    # Independent order-by-order LMA layer equation and all parameter gradients.
    for k in (1, 2, 3):
        torch.manual_seed(81+k)
        layer = pm.source.LatentMobiusAttention(8, 4, k, 3).double()
        x = torch.randn(3, 7, 8, dtype=torch.float64)
        a, b = layer(x), independent_layer(layer, x)
        checks.append(dict(formula="source_lma", k=k, output_max_delta=close(a, b),
                           gradient_max_delta=gradient_comparison(layer, a, b)))
    for n in (2, 5, 10):
        model = pm.build_model(101, "janossy2", n).double()
        x = c.canonical(n, dtype=torch.float64)
        a, b = model(x), model.literal(x)
        checks.append(dict(formula="binary_exact_Janossy_vs_literal_ordered_pairs", n=n,
                           output_max_delta=close(a, b), gradient_max_delta=gradient_comparison(model, a, b)))
    model = pm.build_model(102, "cp_pool", 5).double()
    x = c.canonical(5, dtype=torch.float64)
    a = model(x)
    h = model.embedding(x.unsqueeze(-1))
    factors = model.head.factor(h)
    product = torch.ones_like(factors[:, 0])
    for i in range(x.shape[1]):
        product = product*factors[:, i]
    cp = torch.relu(nn.functional.linear(product.tanh(), model.head.mix.weight))
    low = torch.relu(nn.functional.linear(sum(h[:, i] for i in range(x.shape[1])), model.head.low_order.weight))
    b = model.head.readout(model.head.norm(cp+low))
    checks.append(dict(formula="CP_product_then_tanh_plus_sum", output_max_delta=close(a, b),
                       gradient_max_delta=gradient_comparison(model, a, b)))
    # Product derivatives at single and multiple zero factors.
    f = torch.tensor([[0., 2., 3.], [0., 0., 3.]], dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(f.prod(1).sum(), f)[0]
    close(grad, torch.tensor([[6., 0., 0.], [0., 0., 0.]], dtype=torch.float64), 0, 0)
    # No construction consumes the explicitly independent minibatch stream.
    orders = []
    for name in pm.HEADS:
        pm.build_model(100, name, 10)
        gen = torch.Generator().manual_seed(20100)
        orders.append(torch.randperm(4800, generator=gen))
    assert all(torch.equal(orders[0], order) for order in orders[1:])
    specs = campaign.specifications()
    assert len(specs) == 800 and len({s["id"] for s in specs}) == 800
    c.write_json(HERE/"cpu_audit.json", dict(completed_utc=c.now(), passed=True, data_sets=data_count,
        parameter_counts=counts, sizes=pm.SIZES, formula_and_invariance_checks=checks, metric_checks=metric_checks,
        parameter_pairing_seeds=c.SEEDS, source_class_use="Original LMA, Transformer, DeepSets classes imported unchanged",
        sources=[dict(path=str(p), sha256=c.sha(p)) for p in campaign.locked_sources()]))
    print(json.dumps(dict(stage="parity_cpu_audit_passed", datasets=data_count, checks=len(checks), parameters=counts)), flush=True)


def cuda():
    audit = json.loads((HERE/"cpu_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    for item in audit["sources"]:
        assert c.sha(item["path"]) == item["sha256"]
    assert torch.cuda.is_available()
    train = c.load_data(80, 100, "train", "cuda")
    checks = []
    for name in pm.HEADS:
        model = pm.build_model(100, name, 80).cuda()
        torch.manual_seed(30100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        logits = model(train["x"][:c.BATCH])
        loss = nn.functional.cross_entropy(logits, train["y"][:c.BATCH])
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm = nn.utils.clip_grad_norm_(model.parameters(), pm.clip_threshold(name))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            canon = model(c.canonical(80, "cuda"))
            actual = model(train["x"][:31])
            expected = canon[train["x"][:31].sum(1).long()]
            delta = close(actual, expected, 5e-4, 5e-4)
            assert bool(torch.isfinite(canon).all())
        torch.cuda.synchronize()
        checks.append(dict(head=name, n=80, batch_size=c.BATCH, discarded_update=True,
            loss=float(loss.detach()), preclip_gradient_norm=float(norm),
            canonical_vs_sampled_rows_max_delta=delta, wall_seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model, optimizer
        torch.cuda.empty_cache()
    c.write_json(HERE/"prefit_audit.json", dict(completed_utc=c.now(), passed=True,
        cpu_audit_sha256=c.sha(HERE/"cpu_audit.json"), largest_batch_discarded_updates=checks,
        sources=audit["sources"]))
    print(json.dumps(dict(stage="parity_cuda_audit_passed", discarded_updates=len(checks))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cpu", "cuda"])
    args = parser.parse_args()
    c.configure()
    {"cpu": cpu, "cuda": cuda}[args.stage]()
