"""Checks of the unchanged native CE/autograd chain, without optimizer steps."""
from pathlib import Path
import math, json, time
import numpy as np
import torch
from torch import nn


def native_chain_check(model, x, y, folder):
    # A double-precision diagnostic copy; no trained state is overwritten.
    model = model.double().cuda().train()
    x, y = x.double().cuda(), y.long().cuda()
    assert x.ndim == 2 and x.shape[1] == 20
    assert bool(((x == 0) | (x == 1)).all())
    assert torch.equal(x.sum(1).long() % 2, y)
    assert torch.is_grad_enabled() and model.training
    assert all(p.requires_grad for p in model.parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    connected_modules = []
    handles = []
    def inspect(name, out):
        assert torch.is_tensor(out) and out.requires_grad and out.grad_fn is not None
        assert bool(torch.isfinite(out).all())
        connected_modules.append(name)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.GELU)):
            handles.append(module.register_forward_hook(lambda m, i, o, n=name: inspect(n, o)))
    logits = model(x)
    for h in handles:
        h.remove()
    assert logits.shape == (len(y), 2) and logits.requires_grad
    loss = nn.functional.cross_entropy(logits, y)
    manual_logsoftmax = -nn.functional.log_softmax(logits, dim=1)[torch.arange(len(y), device=x.device), y].mean()
    signed = (2*y-1).to(logits.dtype) * (logits[:, 1]-logits[:, 0])
    manual_margin_ce = nn.functional.softplus(-signed).mean()
    assert torch.allclose(loss, manual_logsoftmax, atol=1e-13, rtol=1e-13)
    assert torch.allclose(loss, manual_margin_ce, atol=1e-13, rtol=1e-13)
    loss.backward()
    params = dict(model.named_parameters())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values())
    gradients = {n: p.grad.detach().clone() for n, p in params.items()}
    names = ["embedding.weight", "layers.0.W_H.weight", "layers.0.W_k.weight",
             "layers.0.W_v.weight", "layers.0.W_q.weight", "layers.0.W_out.weight",
             "layers.0.interaction_projs.0.weight", "layers.0.interaction_projs.1.weight",
             "layers.0.interaction_projs.2.weight", "layers.0.interaction_mlps.0.1.weight",
             "layers.0.layer_norm.weight", "layers.0.order_gates",
             "ffn.0.weight", "ffn.2.weight", "classifier.weight"]
    finite_difference = []
    with torch.no_grad():
        for name in names:
            p, g = params[name], gradients[name]
            norm = float(g.norm())
            if norm > 1e-12:
                direction = g / norm
            else:
                direction = torch.ones_like(g) / math.sqrt(g.numel())
            exact = float((g * direction).sum())
            for epsilon in [1e-4, 1e-5]:
                p.copy_(before[name] + epsilon * direction)
                plus = float(nn.functional.cross_entropy(model(x), y))
                p.copy_(before[name] - epsilon * direction)
                minus = float(nn.functional.cross_entropy(model(x), y))
                p.copy_(before[name])
                numeric = (plus-minus)/(2*epsilon)
                error = abs(numeric-exact)
                tolerance = 1e-7 + .005*abs(exact)
                row = {"parameter": name, "epsilon": epsilon,
                       "direction": "normalized_current_gradient" if norm > 1e-12 else "fixed_unit_zero_gradient_control",
                       "autograd_directional_derivative": exact,
                       "centered_finite_difference": numeric, "absolute_error": error,
                       "allowed_error": tolerance, "passed": error <= tolerance}
                finite_difference.append(row)
                assert row["passed"], "Finite-difference discrepancy: " + name
        total_norm = math.sqrt(sum(float(g.square().sum()) for g in gradients.values()))
        assert total_norm > 0 and math.isfinite(total_norm)
        for name, p in params.items():
            p.copy_(before[name] - 1e-4*gradients[name]/total_norm)
        descent_loss = float(nn.functional.cross_entropy(model(x), y))
        for name, p in params.items():
            p.copy_(before[name])
        assert descent_loss < float(loss.detach()), "Negative-gradient control failed"
    assert all(torch.equal(p.detach(), before[n]) for n, p in params.items())
    return {
        "status": "PASS", "dtype": "float64", "binary_length": 20,
        "rows": len(y), "optimizer_steps": 0, "native_forward_no_detach": True,
        "training_mode": True, "grad_enabled": True,
        "connected_linear_layernorm_gelu_modules": connected_modules,
        "labels_equal_input_count_mod2": True,
        "ce": float(loss.detach()),
        "ce_manual_logsoftmax_error": float(abs(loss-manual_logsoftmax)),
        "ce_manual_signed_margin_error": float(abs(loss-manual_margin_ce)),
        "all_parameter_gradients_present_and_finite": True,
        "parameter_gradient_norms": {n: float(g.norm()) for n, g in gradients.items()},
        "finite_difference_checks": finite_difference,
        "negative_gradient_control_loss": descent_loss,
        "negative_gradient_control_epsilon": 1e-4,
        "state_restored_without_optimizer_updates": True}


def optimizer_coverage(model, optimizer):
    model_ids = {id(p) for p in model.parameters()}
    listed = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(listed) == len({id(p) for p in listed})
    assert {id(p) for p in listed} == model_ids
    assert all(p.requires_grad for p in listed)
    assert sum(p.numel() for p in listed) == 12525
    return {"all_trainable_parameters_covered_once": True,
            "parameter_tensors": len(listed), "parameter_scalars": 12525,
            "param_groups": len(optimizer.param_groups)}


def capture_first_update(model, optimizer):
    coverage = optimizer_coverage(model, optimizer)
    assert model.training and torch.is_grad_enabled()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    return coverage, {n: p.detach().cpu().clone() for n, p in model.named_parameters()}, {
        n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}


def check_first_update(model, optimizer, captured):
    coverage, before, gradients = captured
    group = optimizer.param_groups[0]
    assert group["betas"] == (.9, .999)
    assert group["lr"] == .001 and group["weight_decay"] == 1e-4
    rows = []
    for name, p in model.named_parameters():
        actual = p.detach().cpu()
        g = gradients[name]
        expected = before[name]*(1-group["lr"]*group["weight_decay"]) - group["lr"]*g/(g.abs()+group["eps"])
        error = float((expected-actual).abs().max())
        passed = bool(torch.allclose(expected, actual, atol=2e-7, rtol=2e-6))
        assert passed, "Actual AdamW first update differs from formula: " + name
        assert float(optimizer.state[p]["step"]) == 1.
        rows.append({"parameter": name, "updated_l2": float((actual-before[name]).norm()),
                     "gradient_l2_after_clip": float(g.norm()),
                     "adamw_first_step_maximum_absolute_error": error,
                     "formula_check_passed": passed})
    assert any(row["updated_l2"] > 0 for row in rows)
    return {"status": "PASS", "coverage": coverage, "committed_step": 1,
            "learning_rate": group["lr"], "weight_decay": group["weight_decay"],
            "rows": rows, "no_step_discarded_or_repeated": True,
            "note": "Zero-gradient invariant branches in the full witness can remain unchanged; coverage is distinct from nonzero gradient"}

