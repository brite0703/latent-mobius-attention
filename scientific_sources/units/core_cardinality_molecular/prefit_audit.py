"""Mathematical, masking, data and resource checks before benchmark fitting."""
from itertools import permutations
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import models
import study

HERE = Path(__file__).resolve().parent


def graph_batch(batch=3, n=6, d=53, dtype=torch.float64):
    generator = torch.Generator().manual_seed(913)
    x = torch.randn(batch, n, d, generator=generator, dtype=dtype)
    mask = torch.arange(n)[None, :] < torch.tensor([n, max(1, n//2), 1])[:batch, None]
    edges = torch.rand(batch, n, n, generator=generator, dtype=dtype)
    edges = (edges+edges.transpose(1, 2) > 1.0).to(dtype)
    edges += torch.eye(n, dtype=dtype)[None]
    edges *= mask[:, :, None]*mask[:, None, :]
    degrees = edges.sum(2).clamp_min(1).rsqrt()
    adj = degrees[:, :, None]*edges*degrees[:, None, :]
    return x, mask, adj


def invariance_checks():
    checks = []
    x, mask, adj = graph_batch()
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    for depth, heads in [(1, models.HEADS), (3, models.DEPTH3_HEADS)]:
        for name in heads:
            model = models.build_model(42, name, depth).double().eval()
            padded = F.pad(x, (0, 0, 0, 3))
            padded[:, -3:] = 17.
            padded.requires_grad_()
            pm = F.pad(mask, (0, 3))
            pa = F.pad(adj, (0, 3, 0, 3))
            p = model(x, mask, adj)
            q = model(x[:, perm], mask[:, perm], adj[:, perm][:, :, perm])
            z = model(padded, pm, pa)
            independent = torch.cat([model(x[i:i+1], mask[i:i+1], adj[i:i+1]) for i in range(3)])
            errors = dict(permutation=float((p-q).abs().max().detach()),
                          padding=float((p-z).abs().max().detach()),
                          batch=float((p-independent).abs().max().detach()))
            assert all(v < 3e-10 for v in errors.values()), (depth, name, errors)
            z.square().sum().backward()
            assert bool(torch.isfinite(padded.grad).all())
            assert float(padded.grad[~pm].abs().max()) == 0.
            checks.append(dict(depth=depth, head=name, **errors, masked_input_gradient=0.))
    return checks


def baseline_equations():
    torch.manual_seed(721)
    h = torch.randn(3, 4, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, True], [False, True, False, True], [False, False, True, False]])
    jp = models.Janossy2Head(3, 5).double().eval()
    references = []
    for row, valid in zip(h, mask):
        values = row[valid]
        if len(values) == 1:
            joined = torch.cat([values[0], torch.zeros_like(values[0])])
            references.append(F.gelu(jp.pair_second(F.gelu(jp.pair_first(joined)))))
            continue
        terms = []
        for i, j in permutations(range(len(values)), 2):
            joined = torch.cat([values[i], values[j]])
            terms.append(F.gelu(jp.pair_second(F.gelu(jp.pair_first(joined)))))
        references.append(torch.stack(terms).mean(0))
    explicit = torch.stack(references)
    exact = jp.pooled_pairs(h, mask, exact=True)
    jp_error = float((explicit-exact).abs().max().detach())
    assert jp_error < 1e-12
    # MC empirical mean checked against the exact finite-pair average, using its
    # actual variance rather than an arbitrary absolute agreement threshold.
    jp.samples = 65536
    torch.manual_seed(447)
    mc = jp.pooled_pairs(h, mask, exact=False)
    values = h[0]
    terms = torch.stack([jp.pair_values(values[i], values[j]) for i, j in permutations(range(4), 2)])
    se = terms.std(0, unbiased=False)/np.sqrt(jp.samples)
    max_z = float(((mc[0]-exact[0]).abs()/se.clamp_min(1e-12)).max().detach())
    assert max_z < 7.
    dc = models.DCNV2Head(3, 5).double().eval()
    x0 = dc.representation(h, mask)
    a = x0.clone()
    for layer in dc.cross:
        # Explicit coordinate definition, independent of nn.Linear's invocation.
        cross = torch.stack([sum(layer.weight[i, j]*a[:, j] for j in range(a.shape[1]))+layer.bias[i]
                             for i in range(a.shape[1])], 1)
        a = x0*cross+a
    dc_ref = dc.out(torch.cat([a, dc.deep(x0)], -1)).squeeze(-1)
    dc_error = float((dc(h, mask)-dc_ref).abs().max().detach())
    assert dc_error < 1e-12
    cp = models.CPPoolHead(3, 2).double().eval()
    one = h[:1, :2]
    one_mask = torch.ones(1, 2, dtype=torch.bool)
    w = torch.cat([cp.factor.weight, cp.factor.bias[:, None]], 1)
    homogeneous = torch.cat([one[0], torch.ones(2, 1, dtype=torch.float64)], 1)
    # Construct the explicit rank-R tensor for two inputs and contract. Tanh
    # lies before M, so compare the contraction without it separately.
    tensor = torch.einsum("or,ri,rj->oij", cp.mix.weight, w, w)
    contraction = torch.einsum("oij,i,j->o", tensor, homogeneous[0], homogeneous[1])
    factor_product = torch.stack([torch.dot(w[r], homogeneous[0])*torch.dot(w[r], homogeneous[1])
                                  for r in range(cp.rank)])
    contraction_error = float((contraction-cp.mix(factor_product)).abs().max().detach())
    activated = F.relu(cp.mix(torch.tanh(factor_product)))
    cp_error = float((activated-cp.cp_features(one, one_mask)[0]).abs().max().detach())
    assert contraction_error < 1e-12 and cp_error < 1e-12
    # The zero-factor derivative must not use product/factor or log(abs(factor)).
    for factors in (torch.tensor([[0., 2.], [3., 0.], [4., 5.]], dtype=torch.float64),
                    torch.tensor([[0., 2.], [0., 3.], [4., 5.]], dtype=torch.float64)):
        factors.requires_grad_()
        product = factors.prod(0)
        gradient, = torch.autograd.grad(torch.tanh(product).sum(), factors)
        expected = torch.stack([factors.detach()[[j for j in range(3) if j != i]].prod(0)
                                for i in range(3)])*(1-torch.tanh(product.detach()).square())
        assert torch.equal(gradient, expected)
    grad_input = (one.clone()+0.17).requires_grad_()
    gradcheck = torch.autograd.gradcheck(lambda z: cp.cp_features(z, one_mask),
                                       (grad_input,), eps=1e-6, atol=2e-5, rtol=2e-4)
    assert gradcheck
    return dict(janossy_explicit_pair_error=jp_error, janossy_MC_max_standard_errors=max_z,
                dcn_coordinate_equation_error=dc_error, cp_explicit_tensor_error=contraction_error,
                cp_activated_equation_error=cp_error, cp_zero_factor_derivatives=True, cp_gradcheck=gradcheck)


def matching_and_pairing():
    reference = models.build_model(42, "lma2", 1)
    additive = models.build_model(42, "additive2", 1)
    assert reference.state_dict().keys() == additive.state_dict().keys()
    assert all(torch.equal(v, additive.state_dict()[k]) for k, v in reference.state_dict().items())
    for depth, heads in [(1, models.HEADS), (3, models.DEPTH3_HEADS)]:
        for seed in study.SEEDS:
            ref = models.build_model(seed, "lma2", depth)
            for name in heads:
                model = models.build_model(seed, name, depth)
                assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in ref.encoder.state_dict().items())
    # Regression gradient check: fitting labels must remain vector-valued, not
    # broadcast to a batch-by-batch matrix.
    x, mask, adj = graph_batch(dtype=torch.float32)
    model = models.build_model(42, "lma2", 1)
    target = torch.tensor([-.7, .2, 1.1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    initial = float((model(x, mask, adj)-target).square().mean().detach())
    for _ in range(40):
        p = model(x, mask, adj)
        assert p.shape == target.shape == (3,)
        optimizer.zero_grad()
        loss = (p-target).square().mean()
        loss.backward()
        optimizer.step()
    final = float((model(x, mask, adj)-target).square().mean().detach())
    assert final < initial*.2, (initial, final)
    return dict(product_additive_identical_initial_state=True, shared_backbone_initial_state=True,
                artificial_3_example_initial_mse=initial, artificial_3_example_final_mse=final,
                size_selection=models.SIZES)


def data_and_gpu_checks():
    from pdbbind_rerun import validate_blob
    train_cpu = torch.load(study.DATA/"pdbbind_train.pt", weights_only=True, map_location="cpu")
    val_cpu = torch.load(study.DATA/"pdbbind_val.pt", weights_only=True, map_location="cpu")
    for blob in (train_cpu, val_cpu):
        validate_blob(blob)
        counts = blob["mask"].sum(1)
        assert torch.equal(blob["mask"], torch.arange(blob["X"].shape[1])[None, :] < counts[:, None])
    assert not set(train_cpu["ids"]) & set(val_cpu["ids"])
    largest = train_cpu["mask"].sum(1).argsort(descending=True)[:256]
    x, mask, adj = (train_cpu[k][largest].cuda() for k in ("X", "mask", "adj"))
    target = train_cpu["y"][largest].cuda()
    target = (target-target.mean())/target.std(unbiased=False)
    checks = []
    for depth, name in [(1, h) for h in models.HEADS]+[(3, h) for h in models.DEPTH3_HEADS]:
        model = models.build_model(42, name, depth).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.train()
        p = model(x, mask, adj)
        assert p.shape == target.shape
        loss = (p-target).square().mean()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
        optimizer.step()
        # Eval branch of sampled Janossy is different and needs a separate check.
        model.eval()
        with torch.no_grad():
            output = model(x[:64], mask[:64], adj[:64])
            assert bool(torch.isfinite(output).all())
        torch.cuda.synchronize()
        checks.append(dict(depth=depth, head=name, train_batch_size=256, n_padded=x.shape[1],
                           eval_batch_size=64, one_step_plus_eval_seconds=time.perf_counter()-start,
                           preclip_gradient_norm=float(norm),
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model, optimizer
    return dict(train_size=len(train_cpu["y"]), validation_size=len(val_cpu["y"]),
                train_val_id_disjoint=True, test_tensors_loaded=False, checks=checks)


def main():
    study.configure()
    output = dict(created_utc=study.now(), passed=False)
    output["invariance"] = invariance_checks()
    output["defining_operations"] = baseline_equations()
    output["pairing"] = matching_and_pairing()
    output["data_and_resource"] = data_and_gpu_checks()
    output["audited_sources"] = [dict(path=str(p), sha256=study.sha(p)) for p in
                                [HERE/"models.py", HERE/"study.py", HERE/"prefit_audit.py",
                                 HERE.parent/"lma_revision.py", HERE.parent/"pdbbind_rerun.py",
                                 HERE.parent.parent/"pdbbind_tensors_experiment.py"]]
    output["passed"] = True
    study.write_json(HERE/"prefit_audit.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
