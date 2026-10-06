"""Discarded, manufactured-input CPU checks; no dataset or retained checkpoint."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
import models as candidate

HERE = Path(__file__).resolve().parent
COMP = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear(x, weight, bias):
    return [sum(a*b for a, b in zip(row, x))+c for row, c in zip(weight, bias)]


def gelu(x):
    return [0.5*v*(1+math.erf(v/math.sqrt(2))) for v in x]


def scalar_reference(model, z):
    p = {name: value.detach().tolist() for name, value in model.named_parameters()}
    result = []
    # Independently enumerate j<l, without reading the implementation's pairs.
    for record in z.tolist():
        values = []
        for j in range(len(record)):
            for l in range(j+1, len(record)):
                if model.arm == "pair_mlp":
                    u = gelu(linear(record[j]+record[l], p["pair_map.0.weight"], p["pair_map.0.bias"]))
                    feature = linear(u, p["pair_map.2.weight"], p["pair_map.2.bias"])
                else:
                    left = linear(record[j], p["legs.weight"], p["legs.bias"])
                    right = linear(record[l], p["legs.weight"], p["legs.bias"])
                    w = len(left)//2
                    feature = [a*b if model.arm == "product" else (a+b)/2
                               for a, b in zip(left[:w], right[w:])]
                mean = sum(feature)/len(feature)
                variance = sum((v-mean)**2 for v in feature)/len(feature)
                normed = [(v-mean)/math.sqrt(variance+1e-5)*s+b
                          for v, s, b in zip(feature, p["norm.weight"], p["norm.bias"])]
                values.append(gelu(linear(normed, p["post.weight"], p["post.bias"])))
        pooled = [sum(row[j] for row in values)/len(values) for j in range(len(values[0]))]
        result.append(linear(pooled, p["output.weight"], p["output.bias"])[0])
    return torch.tensor(result, dtype=z.dtype)


def tensor_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal_nested(left, right)
    else:
        assert a == b


def kernel_checks():
    rng = torch.Generator().manual_seed(2026090843)
    rows = []
    for d in (8, 12):
        z = torch.randn(3, 8, d, generator=rng, dtype=torch.float64)
        y = torch.tensor([.72, -.41, 1.37], dtype=torch.float64)
        baseline = torch.tensor([-.2, .6, .3], dtype=torch.float64)
        reference_common = None
        reference_legs = None
        for arm in candidate.ARMS:
            state_before_build = torch.random.get_rng_state().clone()
            model = candidate.PairResidual(arm, d, seed=43).double()
            assert torch.equal(state_before_build, torch.random.get_rng_state())
            assert torch.count_nonzero(model(z)) == 0
            assert torch.equal(candidate.predict_cached(baseline, z, model), baseline)
            common = {k: v for k, v in model.state_dict().items()
                      if k.startswith(("norm.", "post.", "output."))}
            if reference_common is None:
                reference_common = deepcopy(common)
            else:
                equal_nested(common, reference_common)
            if arm in ("product", "additive"):
                if reference_legs is None:
                    reference_legs = tensor_state(model.legs)
                else:
                    equal_nested(tensor_state(model.legs), reference_legs)
            loss = (candidate.predict_cached(baseline, z, model)-y).square().mean()
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if not name.startswith("output."):
                    assert torch.count_nonzero(parameter.grad) == 0, name
                else:
                    assert float(parameter.grad.norm()) > 0, name
            # Only the final readout is changed here to test gradient release.
            with torch.no_grad():
                for p in model.output.parameters():
                    p.add_(p.grad, alpha=-.01)
            model.zero_grad(set_to_none=True)
            (candidate.predict_cached(baseline, z, model)-y).square().mean().backward()
            gradient_norms = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                gradient_norms[name] = float(parameter.grad.norm())
                assert gradient_norms[name] > 1e-13, (arm, name)

            manual = scalar_reference(model, z)
            with torch.no_grad():
                value = model(z)
            difference = float((manual-value).abs().max())
            assert difference < 3e-12, (arm, difference)
            derivative_errors = []
            names = [next(iter(dict(model.named_parameters()))), "norm.weight", "post.weight", "output.weight"]
            for name in names:
                p = dict(model.named_parameters())[name]
                index = (0,)*p.ndim
                analytic = float(p.grad[index])
                original = float(p.detach()[index])
                losses = []
                for delta in (1e-5, -1e-5):
                    with torch.no_grad():
                        p[index] = original+delta
                    prediction = baseline+scalar_reference(model, z)
                    losses.append(float((prediction-y).square().mean()))
                with torch.no_grad():
                    p[index] = original
                finite = (losses[0]-losses[1])/2e-5
                assert math.isclose(analytic, finite, rel_tol=8e-5, abs_tol=2e-9), (arm, name, analytic, finite)
                derivative_errors.append(abs(analytic-finite))
            with torch.no_grad():
                together = model(z)
                separate = torch.cat([model(z[i:i+1]) for i in range(len(z))])
            torch.testing.assert_close(together, separate, rtol=1e-12, atol=1e-13)
            # Serialization/resume of actual AdamW state, without disk data.
            continuous = deepcopy(model)
            interrupted = deepcopy(model)
            opt_a = torch.optim.AdamW(continuous.parameters(), lr=.001, weight_decay=.0001)
            opt_b = torch.optim.AdamW(interrupted.parameters(), lr=.001, weight_decay=.0001)
            scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=100)
            scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=100)

            def step(module, optimizer, scheduler, chosen):
                optimizer.zero_grad(set_to_none=True)
                value = candidate.predict_cached(baseline[chosen], z[chosen], module)
                (value-y[chosen]).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
                optimizer.step()
                scheduler.step()

            step(continuous, opt_a, scheduler_a, torch.tensor([2, 0]))
            step(interrupted, opt_b, scheduler_b, torch.tensor([2, 0]))
            stream = io.BytesIO()
            torch.save(dict(model=interrupted.state_dict(), optimizer=opt_b.state_dict(),
                            scheduler=scheduler_b.state_dict()), stream)
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
            rebuilt = candidate.PairResidual(arm, d, seed=1043).double()
            rebuilt.load_state_dict(checkpoint["model"], strict=True)
            opt_c = torch.optim.AdamW(rebuilt.parameters(), lr=.001, weight_decay=.0001)
            scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=100)
            opt_c.load_state_dict(checkpoint["optimizer"])
            scheduler_c.load_state_dict(checkpoint["scheduler"])
            step(continuous, opt_a, scheduler_a, torch.tensor([1]))
            step(rebuilt, opt_c, scheduler_c, torch.tensor([1]))
            equal_nested(tensor_state(continuous), tensor_state(rebuilt))
            equal_nested(opt_a.state_dict(), opt_c.state_dict())
            equal_nested(scheduler_a.state_dict(), scheduler_c.state_dict())
            with torch.no_grad():
                assert torch.equal(continuous(z), rebuilt(z))
            rows.append(dict(arm=arm, dimension=d, parameters=model.parameter_count(),
                             exact_initial_baseline=True, shared_initial_parameters=True,
                             scalar_reference_max_delta=difference,
                             finite_difference_max_error=max(derivative_errors),
                             all_parameter_gradient_norms_after_readout_change=gradient_norms,
                             exact_serialized_optimizer_continuation=True))
    return rows


def native_checks():
    synthetic = import_file("conditional_synthetic_models", COMP/"first_cubic/neural_models.py")
    pocket_path = COMP/"receptor_context/pocket_reconstruction"
    sys.path.insert(0, str(pocket_path))
    import matched_models as matched
    rows = []
    gen = torch.Generator().manual_seed(2026090844)
    for domain in ("cubic", "ligand_contact"):
        molecular = domain == "ligand_contact"
        baseline = (matched.build_model(42, "ligand_contact", "lma1") if molecular
                    else synthetic.build(100, "lma1")).double().eval()
        layer_path = "pool.layers.0" if molecular else "head.layers.0"
        n, d = (7, 8) if molecular else (12, 12)
        x = torch.randn(4, n, 53 if molecular else 12, generator=gen, dtype=torch.float64)
        lengths = torch.tensor([n, n-2, 3, 1 if molecular else 0])
        mask = torch.arange(n)[None] < lengths[:, None]
        args = dict(x=x, mask=mask)
        if molecular:
            args.update(adj=torch.eye(n, dtype=torch.float64).expand(4, n, n).clone(),
                        contact=torch.rand(4, n, 216, generator=gen, dtype=torch.float64))
        saved = tensor_state(baseline)
        independent = []
        layer = baseline.get_submodule(layer_path)
        observed = []
        handle = layer.register_forward_pre_hook(lambda _, inputs: observed.append(inputs))
        with torch.no_grad():
            native_prediction = baseline(**args)
        handle.remove()
        h, original_mask = observed[0]
        with torch.no_grad():
            # Independent bucket equation, separate from the native capture hook.
            pi = F.softmax(layer.W_H(layer.W_k(h)), -1)
            reference_z = pi.transpose(1, 2) @ (layer.W_v(h)*original_mask.unsqueeze(-1))
        frozen = candidate.FrozenLmaFeatures(baseline, layer_path).train()
        assert not baseline.training and all(not m.training for m in baseline.modules())
        base_prediction, z = frozen(**args)
        assert torch.equal(native_prediction, base_prediction)
        assert torch.equal(reference_z, z)
        assert not z.requires_grad and not base_prediction.requires_grad
        order = torch.randperm(n, generator=gen)
        permuted = dict(x=x[:, order], mask=mask[:, order])
        padded = dict(x=F.pad(x, (0, 0, 0, 3), value=73.1), mask=F.pad(mask, (0, 3)))
        if molecular:
            permuted.update(adj=args["adj"][:, order][:, :, order], contact=args["contact"][:, order])
            padded.update(adj=F.pad(args["adj"], (0, 3, 0, 3)),
                          contact=F.pad(args["contact"], (0, 0, 0, 3), value=927.3))
        perm_prediction, perm_z = frozen(**permuted)
        padded_prediction, padded_z = frozen(**padded)
        for first, second in ((base_prediction, perm_prediction), (base_prediction, padded_prediction),
                              (z, perm_z), (z, padded_z)):
            torch.testing.assert_close(first, second, rtol=1e-11, atol=3e-12)
        losses = []
        targets = torch.tensor([1.2, -.5, .1, 2.], dtype=torch.float64)
        for arm in candidate.ARMS:
            residual = candidate.PairResidual(arm, d, seed=47).double().train()
            assert torch.equal(candidate.predict_cached(base_prediction, z, residual), base_prediction)
            optimizer = torch.optim.AdamW(residual.parameters(), lr=.001, weight_decay=.0001)
            assert not ({id(p) for p in baseline.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]})
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = (candidate.predict_cached(base_prediction, z, residual)-targets).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.)
                optimizer.step()
            with torch.no_grad():
                trained = candidate.predict_cached(base_prediction, z, residual)
                p = candidate.predict_cached(perm_prediction, perm_z, residual)
                q = candidate.predict_cached(padded_prediction, padded_z, residual)
            torch.testing.assert_close(trained, p, rtol=1e-10, atol=3e-12)
            torch.testing.assert_close(trained, q, rtol=1e-10, atol=3e-12)
            equal_nested(saved, tensor_state(baseline))
            assert all(p.grad is None and not p.requires_grad for p in baseline.parameters())
            f_after, z_after = frozen(**args)
            assert torch.equal(base_prediction, f_after) and torch.equal(z, z_after)
            losses.append(dict(arm=arm, frozen_state_exact=True, native_cache_exact=True,
                               fitted_prediction_changed=bool((trained != base_prediction).any())))
            assert losses[-1]["fitted_prediction_changed"]
        # A failing native forward cannot leak the temporary capture hook.
        count_before = len(layer.interaction_projs[0]._forward_pre_hooks)
        try:
            frozen(x=x)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("The malformed native call should fail")
        assert len(layer.interaction_projs[0]._forward_pre_hooks) == count_before
        rows.append(dict(domain=domain, dimension=d, native_prediction_max_delta=0.,
                         native_bucket_max_delta=0., input_permutation_max_delta=float((z-perm_z).abs().max()),
                         padding_max_delta=float((z-padded_z).abs().max()),
                         empty_set_included=not molecular, arms=losses,
                         capture_hook_removed_after_failure=True))
    sources = synthetic.sources()+[pocket_path/"matched_models.py", pocket_path.parent/"sequence_models.py",
                                   pocket_path.parent/"sequence_encoder.py"]
    return rows, sources


def main():
    destination = HERE/"models_cpu_audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed preparation audit")
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kernels = kernel_checks()
    native, sources = native_checks()
    assert not torch.cuda.is_initialized()
    sources = sorted(set(sources+[Path(__file__), HERE/"models.py", HERE/"protocol_draft.md"]))
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  torch_version=torch.__version__, python_version=sys.version,
                  cpu_threads=1, cuda_initialized=False, retained_or_dataset_inputs_loaded=False,
                  dimensions=[candidate.dimensions(d) for d in (8, 12)],
                  kernel_checks=kernels, native_checks=native,
                  sources=[dict(path=str(path), sha256=sha(path)) for path in sources],
                  scope="Manufactured CPU scalar/gradient/native-cache/frozen-state/serialization checks only. "
                        "No real cache, candidate lifecycle, validation selection, test scoring or retained fitting.")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, kernels=len(kernels), native_domains=len(native),
                         dimensions=result["dimensions"], cuda_initialized=False)))


if __name__ == "__main__":
    main()
