"""Finite-weight witness for the unchanged native N20 LMANetwork.

This module has deliberately NOT been executed against a model by its author.
It neither imports nor constructs the source model, loads a checkpoint, trains,
nor runs a forward pass. The authorized local owner supplies the unchanged
native model and separately verifies the assigned witness.

Source lma.py SHA256:
5f62f77ef974c007ac837b427ec1fe1fa0600cf84de6d7fa4fcf44cfa3970b53
"""

import math


def assign_native_n20_witness(model, logit_scale=10.0):
    """Overwrite only existing trainable parameters; return algebraic metadata.

    The caller must pass source.LMANetwork(20, 32, 8, 3, 16, 1, False), or the
    source-identical parity_models.build_model(seed, 'lma3', 20) / 'lma3_clip1'.
    This is a hand-assigned representability witness, never a trained result.
    The proof is in native_n20_derivation.md alongside this file.
    """
    import torch

    assert len(model.layers) == 1 and model.use_positional is False
    assert tuple(model.embedding.weight.shape) == (32, 1)
    assert tuple(model.ffn[0].weight.shape) == (64, 32)
    assert tuple(model.ffn[2].weight.shape) == (32, 64)
    assert tuple(model.classifier.weight.shape) == (2, 32)
    layer = model.layers[0]
    assert (layer.d_model, layer.M, layer.k, layer.d_latent) == (32, 8, 3, 16)
    assert len(layer.interaction_projs) == len(layer.interaction_mlps) == 3
    assert all(p.requires_grad for p in model.parameters())
    assert model.ffn[1].approximate == 'none'
    assert layer.interaction_mlps[0][2].approximate == 'none'
    assert math.isfinite(logit_scale) and logit_scale > 0
    eps16 = float(layer.interaction_mlps[0][0].eps)
    eps32 = float(layer.layer_norm.eps)
    assert math.isfinite(eps16) and eps16 > 0
    assert math.isfinite(eps32) and eps32 > 0

    # These are closed-form ideal-real count states, not model evaluations.
    def count_state(c):
        u = (c - 10.0) / 10.0
        t = u / math.sqrt((1.0 + u*u)/8.0 + eps16)
        return t / math.sqrt((1.0 + t*t)/16.0 + eps32)

    states = [count_state(c) for c in range(21)]
    delta = min(states[c+1] - states[c] for c in range(20))
    assert delta > 0 and math.isfinite(delta)
    A = 12.0 / delta
    thresholds = [(states[j-1] + states[j])/2.0 for j in range(1, 21)]

    with torch.no_grad():
        # All source modules, combinatorial buffers, LayerNorm epsilon values,
        # activation choices, pooling, and requires_grad flags remain intact.
        for parameter in model.parameters():
            parameter.zero_()

        # h_i = b_i * 1_32 + e_0 - e_1. Zero Q/K/hash maps are intentional.
        model.embedding.weight[:, 0] = 1.0
        model.embedding.bias[0] = 1.0
        model.embedding.bias[1] = -1.0

        # V_i = (8*b_i - 4)*e_0, pi_ij = 1/8, hence every Z_j=(c-10)*e_0.
        layer.W_v.weight[0, :] = 0.25
        layer.W_v.bias[0] = -4.0

        # First-order product is (u,-u,1,-1,0,...), u=(c-10)/10.
        # The +/-1 biases are essential: they preserve count information at LN.
        layer.interaction_projs[0].weight[0, 0] = 0.1
        layer.interaction_projs[0].weight[1, 0] = -0.1
        layer.interaction_projs[0].bias[2] = 1.0
        layer.interaction_projs[0].bias[3] = -1.0
        mlp = layer.interaction_mlps[0]
        mlp[0].weight.fill_(1.0)
        mlp[1].weight[0, 0] = 1.0
        mlp[1].weight[1, 0] = -1.0
        # GELU(t)-GELU(-t)=t exactly in real arithmetic.
        mlp[3].weight[2, 0] = 1.0
        mlp[3].weight[2, 1] = -1.0
        mlp[3].weight[3, 0] = -1.0
        mlp[3].weight[3, 1] = 1.0
        layer.order_gates[0] = 1.0  # Others stay zero, with zero finite branches.

        # Eight nonzero first-order memories among 92 concatenated memories.
        # Q=0 gives 1/92 weights, so W_out cancels the 8/92 dilution.
        layer.W_out.weight[2, 2] = 92.0 / 8.0
        layer.W_out.weight[3, 3] = 92.0 / 8.0
        layer.layer_norm.weight.fill_(1.0)
        layer.layer_norm.weight[4] = 0.0  # Force the score residual coordinate 0.

        # 20 soft steps, two native GELU units apiece: 40 of available 64 units.
        for j, threshold in enumerate(thresholds, start=1):
            left, right = 2*(j-1), 2*(j-1)+1
            sigma = 1.0 if j % 2 else -1.0
            model.ffn[0].weight[left, 2] = A
            model.ffn[0].weight[right, 2] = A
            model.ffn[0].bias[left] = -A*threshold + 1.0
            model.ffn[0].bias[right] = -A*threshold - 1.0
            model.ffn[2].weight[4, left] = sigma
            model.ffn[2].weight[4, right] = -sigma
        model.ffn[2].bias[4] = -1.0
        # Native residual, mean pool, and native classifier remain in use.
        # logits=(0, logit_scale*f), positive f = odd parity/class 1.
        model.classifier.weight[1, 4] = logit_scale

    phi5 = math.exp(-12.5)/math.sqrt(2.0*math.pi)
    score_error_bound = 40.0*phi5
    return {
        'witness_type': 'hand_assigned_native_parameters_not_trained',
        'n': 20,
        'D': 32,
        'DL': 16,
        'M': 8,
        'k': 3,
        'depth': 1,
        'eps16': eps16,
        'eps32': eps32,
        'count_states_ideal_real': states,
        'minimum_count_gap_ideal_real': delta,
        'thresholds_ideal_real': thresholds,
        'ffn_scale_A': A,
        'used_ffn_hidden_units': 40,
        'logit_scale': logit_scale,
        'score_error_bound_ideal_real': score_error_bound,
        'signed_binary_margin_lower_bound_ideal_real': logit_scale*(1.0-score_error_bound),
        'higher_order_gates': [0.0, 0.0],
        'finite_precision_verification_required': True,
    }
