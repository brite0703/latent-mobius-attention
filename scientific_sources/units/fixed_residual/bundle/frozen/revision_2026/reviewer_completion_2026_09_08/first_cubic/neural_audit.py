"""Pre-fit formula, gradient, mask, initialization, data and GPU checks."""
from pathlib import Path
import argparse
import itertools
import json
import math
import sys
import time
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import finite_neural_engine as engine
a = engine.load_adapter("first_cubic")
m, u = a.m, a.u


def close(x, y, atol=2e-10, rtol=2e-9):
    torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    return float((x-y).detach().abs().max())


def gradients(model, x, y):
    params = tuple(model.parameters())
    left = torch.autograd.grad(x.sum(), params, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(y.sum(), params, allow_unused=True)
    maximum = 0.
    for p, q in zip(left, right):
        assert (p is None) == (q is None)
        if p is not None:
            maximum = max(maximum, close(p, q, 1e-9, 1e-8))
    return maximum


def independent_layer(layer, h, mask):
    valid = mask.unsqueeze(-1)
    route = torch.softmax(layer.W_H(layer.W_k(h)), -1)
    value = layer.W_v(h)*valid
    buckets = torch.stack([sum(route[:, i, j, None]*value[:, i] for i in range(h.shape[1]))
                           for j in range(layer.M)], 1)
    tokens = []
    for r in range(1, layer.k+1):
        projection = layer.interaction_projs[r-1]
        for indices in itertools.combinations(range(layer.M), r):
            parts = [nn.functional.linear(buckets[:, j],
                projection.weight[leg*layer.d_latent:(leg+1)*layer.d_latent],
                projection.bias[leg*layer.d_latent:(leg+1)*layer.d_latent]) for leg, j in enumerate(indices)]
            feature = parts[0]
            for part in parts[1:]:
                feature = feature*part if layer.feature_mode == "product" else feature+part
            if layer.feature_mode == "additive":
                feature = feature/r
            tokens.append(layer.order_gates[r-1]*layer.interaction_mlps[r-1](feature))
    memory = torch.stack(tokens, 1)
    scores = (layer.W_q(h)[:, :, None, :]*memory[:, None, :, :]).sum(-1)/math.sqrt(layer.d_model)
    output = (scores.softmax(-1)[..., None]*memory[:, None, :, :]).sum(2)
    return layer.layer_norm(h+layer.W_out(output))*valid


def independent_model(model, name, x, mask):
    h, head = model.encoder(x), model.head
    if name.startswith("lma") or name.startswith("additive"):
        for layer in head.layers:
            h = independent_layer(layer, h, mask)
            h = (h+head.ffn(h))*mask.unsqueeze(-1)
        pooled = h.sum(1)/(mask.sum(1, keepdim=True)+1e-6)
        return head.head(head.norm(pooled)).squeeze(-1)
    if name.startswith("deepsets"):
        pooled = torch.stack([sum((head.phi(h[b, i]) for i in range(h.shape[1]) if mask[b, i]),
                                  h.new_zeros(head.phi[0].out_features)) for b in range(len(h))])
        return head.rho(pooled).squeeze(-1)
    if name == "janossy2":
        pooled = []
        for b in range(len(h)):
            active = torch.where(mask[b])[0].tolist()
            if len(active) >= 2:
                pooled.append(torch.stack([head.pair_values(h[b, i], h[b, j])
                                           for i in active for j in active if i != j]).mean(0))
            else:
                zero = h.new_zeros(h.shape[-1])
                pooled.append(head.pair_values(h[b, active[0]] if active else zero, zero))
        n = mask.sum(1, keepdim=True).to(h.dtype).log1p()
        return head.readout(torch.cat([head.norm(torch.stack(pooled)), n], 1)).squeeze(-1)
    factors = head.factor(h)
    product = torch.ones_like(factors[:, 0]).double()
    low = torch.zeros_like(h[:, 0])
    for i in range(h.shape[1]):
        product = product*torch.where(mask[:, i, None], factors[:, i].double(), torch.ones_like(product))
        low = low+torch.where(mask[:, i, None], h[:, i], torch.zeros_like(low))
    feature = torch.relu(nn.functional.linear(product.tanh().to(h.dtype), head.mix.weight))
    feature = feature+torch.relu(nn.functional.linear(low, head.low_order.weight))
    return head.readout(head.norm(feature)).squeeze(-1)


def audit_cpu():
    a.prepare()
    checks, counts = [], {}
    bits = torch.tensor([[0]*12, [1]+[0]*11, [1,0,0,1]+[0]*8,
                         [1,0,1,1,0,0,0,1,0,0,1,0], [1]*12], dtype=torch.bool)
    x = torch.eye(12, dtype=torch.float64).expand(5,-1,-1).clone()
    torch.manual_seed(711)
    permutation = torch.randperm(12)
    extra = torch.randn(5, 3, 12, dtype=torch.float64)*17
    for name in a.HEADS:
        model = m.build(100, name).double()
        counts[name] = m.count(model)
        model.eval()
        expected = model(x, bits)
        assert bool(torch.isfinite(expected).all())
        pd = close(model(x[:, permutation], bits[:, permutation]), expected)
        padding = close(model(torch.cat([x, extra], 1), nn.functional.pad(bits, (0,3))), expected)
        batch = close(torch.cat([model(x[i:i+1], bits[i:i+1]) for i in range(5)]), expected)
        compact = []
        for i in range(5):
            active = torch.where(bits[i])[0]
            compact.append(model(x[i:i+1, active] if len(active) else x.new_zeros(1,1,12),
                torch.ones(1,len(active),dtype=torch.bool) if len(active) else torch.zeros(1,1,dtype=torch.bool)))
        compact_delta = close(torch.cat(compact), expected)
        # Activate every order for meaningful formula/gradient testing; no fitted model is changed.
        if hasattr(model.head, "layers"):
            with torch.no_grad():
                for layer in model.head.layers:
                    layer.order_gates.copy_(torch.linspace(.7, 1.1, layer.k, dtype=torch.float64))
        model.train()
        left, right = model(x, bits), independent_model(model, name, x, bits)
        fd = close(left, right)
        gd = gradients(model, left, right)
        checks.append(dict(head=name, permutation_max_delta=pd, padding_max_delta=padding,
            batch_max_delta=batch, compact_empty_singleton_max_delta=compact_delta,
            independent_formula_max_delta=fd, independent_all_parameter_gradient_max_delta=gd))
    for seed in a.SEEDS:
        reference = m.template(seed)
        ref = reference.state_dict()
        for name in a.HEADS:
            model = m.build(seed, name)
            for key, tensor in model.encoder.state_dict().items():
                assert torch.equal(tensor, reference.encoder.state_dict()[key])
            if name.startswith("lma") or name.startswith("additive"):
                for key, tensor in model.state_dict().items():
                    expected = ref[key][:int(name[-1])] if key.endswith("order_gates") else ref[key]
                    assert torch.equal(tensor, expected), (seed, name, key)
        ln, plain = m.build(seed,"deepsets_ln"), m.build(seed,"deepsets_plain")
        for key, tensor in plain.state_dict().items():
            assert torch.equal(tensor, ln.state_dict()[key])
        for k in (2,3):
            left, right = m.build(seed,f"lma{k}"), m.build(seed,f"additive{k}")
            assert all(torch.equal(v, right.state_dict()[key]) for key,v in left.state_dict().items())
    specs = engine.specifications(a)
    assert len(specs)==400 and len({s["id"] for s in specs})==400
    gen = torch.Generator().manual_seed(20100)
    expected_order = torch.randperm(2048,generator=gen)
    for name in a.HEADS:
        m.build(100,name)
        assert torch.equal(expected_order, torch.randperm(2048,generator=torch.Generator().manual_seed(20100)))
    for y,p in [(np.array([0.,1.,-2.]), np.array([.5,1.5,-1.])),
                (np.array([0.,0.]),np.array([0.,0.]))]:
        direct, reference = engine.metrics(a,y,p), engine.independent_metrics(a,y,p)
        assert max(abs(direct[k]-reference[k]) for k in direct)<1e-14
    u.write_json(HERE/"cpu_audit.json", dict(completed_utc=u.now(),passed=True,parameter_counts=counts,
        sizes=a.SIZES,model_checks=checks,parameter_pairing_seeds=a.SEEDS,candidate_count=400,
        sources=[dict(path=str(p),sha256=u.sha(p)) for p in engine.sources(a)]))
    print(json.dumps(dict(stage="first_cubic_cpu_audit_passed", models=len(checks), parameters=counts)),flush=True)


def audit_cuda():
    cpu = engine.read(HERE/"cpu_audit.json")
    assert cpu["passed"]
    for item in cpu["sources"]:
        assert u.sha(item["path"]) == item["sha256"]
    checks=[]
    for name in a.HEADS:
        spec=dict(head=name,seed=100,task="cubic")
        data=a.load(spec,"train","cuda")
        model=a.build(spec).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started=time.perf_counter()
        model.train()
        prediction=model(*(t[:a.BATCH] for t in data["inputs"]))
        loss=engine.loss_value(a,prediction,data["y"][:a.BATCH])
        loss.backward()
        assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm=nn.utils.clip_grad_norm_(model.parameters(),a.CLIP)
        optimizer.step()
        prediction=engine.predict(a,model,a.load(spec,"val","cuda"))
        assert np.isfinite(prediction).all()
        torch.cuda.synchronize()
        checks.append(dict(head=name,discarded_update=True,batch_size=a.BATCH,loss=float(loss.detach()),
            preclip_gradient_norm=float(norm),wall_seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer,data
        torch.cuda.empty_cache()
    u.write_json(HERE/"prefit_audit.json",dict(passed=True,completed_utc=u.now(),
        cpu_audit_sha256=u.sha(HERE/"cpu_audit.json"),discarded_updates=checks))
    print(json.dumps(dict(stage="first_cubic_cuda_audit_passed",updates=len(checks))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["cpu","cuda"])
    args=parser.parse_args()
    u.configure()
    {"cpu":audit_cpu,"cuda":audit_cuda}[args.stage]()
