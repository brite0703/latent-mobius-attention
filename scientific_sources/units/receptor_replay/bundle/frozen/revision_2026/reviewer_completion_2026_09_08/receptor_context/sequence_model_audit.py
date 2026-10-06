"""CPU equation, all-parameter gradient, invariance and accumulation checks."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import sequence_models as models
from sequence_encoder import encode_strings
from sequence_encoder_audit import independent as reference_sequence
from sequence_data import SequenceStore

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layer(value, module):
    if isinstance(module, nn.Linear):
        result = torch.einsum("...i,oi->...o", value, module.weight)
        return result if module.bias is None else result+module.bias
    if isinstance(module, nn.LayerNorm):
        centered = value-value.mean(-1, keepdim=True)
        return centered/(centered.square().mean(-1, keepdim=True)+module.eps).sqrt()*module.weight+module.bias
    if isinstance(module, nn.GELU):
        return value*.5*(1+torch.erf(value/math.sqrt(2)))
    if isinstance(module, nn.Sequential):
        for operation in module:
            value = layer(value, operation)
        return value
    raise TypeError(type(module))


def softmax(value):
    positive = torch.exp(value-value.amax(-1, keepdim=True))
    return positive/positive.sum(-1, keepdim=True)


def reference_pool(model, h, mask):
    pool = model.pool
    valid = mask.unsqueeze(-1)
    if model.head_name == "deepsets_plain70":
        return layer((layer(h, pool.phi)*valid).sum(1), pool.project)
    if model.head_name == "cp_pool":
        factors = layer(h, pool.factor)
        # Explicit per-record valid-atom multiplication; padded factors never enter.
        products = []
        for b in range(len(h)):
            product = h.new_ones(models.CP_RANK)
            for i in torch.where(mask[b])[0]:
                product = product*factors[b, i]
            products.append(product)
        high = layer(torch.tanh(torch.stack(products)), pool.mix).relu()
        low = layer((h*valid).sum(1), pool.low_order).relu()
        return layer(high+low, pool.norm)
    for operation in pool.layers:
        q = layer(h, operation.W_q)
        routes = softmax(layer(layer(h, operation.W_k), operation.W_H))
        values = layer(h, operation.W_v)*valid
        z = torch.einsum("bnm,bnv->bmv", routes, values)
        memories = []
        for order in range(1, operation.k+1):
            projection = layer(z, operation.interaction_projs[order-1]).reshape(len(h), 8, order, 8)
            features = []
            for indices in itertools.combinations(range(8), order):
                terms = [projection[:, bucket, leg, :] for leg, bucket in enumerate(indices)]
                result = terms[0]
                for term in terms[1:]:
                    result = result+term if operation.feature_mode == "additive" else result*term
                features.append(result/order if operation.feature_mode == "additive" else result)
            memories.append(operation.order_gates[order-1]*layer(torch.stack(features, 1), operation.interaction_mlps[order-1]))
        memory = torch.cat(memories, 1)
        weights = softmax(torch.einsum("bnd,bld->bnl", q, memory)/4)
        retrieved = torch.einsum("bnl,bld->bnd", weights, memory)
        h = layer(h+layer(retrieved, operation.W_out), operation.layer_norm)*valid
        h = (h+layer(h, pool.ffn))*valid
    return layer(h.sum(1)/(mask.sum(1, keepdim=True)+1e-6), pool.norm)


def reference(model, x, mask, adj, sequence_batch):
    records = sequence_batch[3] if model.encoder is None else len(x)
    fused = model.fusion_bias[None].expand(records, -1)
    if model.encoder is not None:
        h = x
        for linear, norm in zip(model.encoder.layers, model.encoder.norms):
            h = layer(layer(layer(torch.einsum("bij,bjf->bif", adj, h), linear), norm), nn.GELU())*mask.unsqueeze(-1)
        pooled = reference_pool(model, h, mask)
        context = torch.cat([pooled, mask.sum(1, keepdim=True).to(h.dtype).log1p()], -1)
        fused = fused+layer(context, model.ligand_projection)
    if model.sequence is not None:
        fused = fused+layer(reference_sequence(model.sequence, *sequence_batch), model.sequence_projection)
    return layer(layer(fused, nn.GELU()), model.regression).squeeze(-1)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(2026090834)
    x = torch.randn(3, 4, 53, dtype=torch.float64)
    mask = torch.arange(4)[None] < torch.tensor([4, 3, 2])[:, None]
    x *= mask.unsqueeze(-1)
    adjacency = torch.zeros(3, 4, 4, dtype=torch.float64)
    for b, n in enumerate([4, 3, 2]):
        adjacency[b, :n, :n] = torch.eye(n)
        for i in range(n-1):
            adjacency[b, i, i+1] = adjacency[b, i+1, i] = 1
    scale = adjacency.sum(-1).clamp_min(1).rsqrt()
    adj = adjacency*scale[:, :, None]*scale[:, None, :]
    strings = ["ACDX", "W:AG", "GGT:KX:ACDEFGHIKLMNPQ"]
    sequence = encode_strings(strings)
    targets = torch.tensor([-.7, .2, 1.1], dtype=torch.float64)
    checks, counts = [], {}
    for setting, head in models.configurations():
        model = models.build_model(42, setting, head).double()
        counts[setting+"/"+head] = model.parameter_counts()
        actual = model(x, mask, adj, sequence)
        expected = reference(model, x, mask, adj, sequence)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        params = tuple(model.parameters())
        lhs = torch.autograd.grad((actual-targets).square().mean(), params)
        rhs = torch.autograd.grad((expected-targets).square().mean(), params)
        for a, b in zip(lhs, rhs):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
            assert bool(torch.isfinite(a).all())
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model(x[:, permutation], mask[:, permutation], adj[:, permutation][:, :, permutation], sequence)
        padded_seq = (F.pad(sequence[0], (0, 13)), F.pad(sequence[1], (0, 13)), sequence[2], sequence[3])
        padded = model(F.pad(x, (0, 0, 0, 3)), F.pad(mask, (0, 3)), F.pad(adj, (0, 3, 0, 3)), padded_seq)
        reordered_seq = encode_strings([":".join(reversed(s.split(":"))) for s in strings])
        reordered = model(x, mask, adj, reordered_seq)
        individual = torch.cat([model(x[i:i+1], mask[i:i+1], adj[i:i+1], encode_strings([strings[i]])) for i in range(3)])
        for value in (permuted, padded, reordered, individual):
            torch.testing.assert_close(actual, value, atol=1e-12, rtol=1e-10)
        # Padded graph data may contain arbitrary values; zero adjacency rows/columns remove them.
        noise = torch.where(mask.unsqueeze(-1), x, torch.full_like(x, 99))
        torch.testing.assert_close(actual, model(noise, mask, adj, sequence), atol=1e-12, rtol=1e-10)
        model.zero_grad(set_to_none=True)
        for start, stop in [(0, 2), (2, 3)]:
            output = model(x[start:stop], mask[start:stop], adj[start:stop], encode_strings(strings[start:stop]))
            ((output-targets[start:stop]).square().sum()/3).backward()
        gradients = [p.grad.clone() for p in params]
        for a, b in zip(lhs, gradients):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
        if model.sequence is not None:
            assert bool(model.sequence.embedding.weight.grad[0].eq(0).all())
            for block in (model.sequence.embedding, *model.sequence.convolutions, model.sequence_projection):
                assert sum(float(p.grad.abs().sum()) for p in block.parameters()) > 0
            changed = model(x, mask, adj, encode_strings(["YYYY", "XX:CC", "WWW:YY:VV"]))
            assert float((actual-changed).detach().abs().max()) > 1e-7
        state = {key: value.clone() for key, value in model.state_dict().items()}
        reloaded = models.build_model(42, setting, head).double()
        reloaded.load_state_dict(state, strict=True)
        torch.testing.assert_close(reloaded(x, mask, adj, sequence), actual, atol=0, rtol=0)
        checks.append(dict(setting=setting, head=head,
            equation_delta=float((actual-expected).detach().abs().max()),
            all_parameter_gradient_delta=max(float((a-b).abs().max()) for a, b in zip(lhs, rhs)),
            accumulated_gradient_delta=max(float((a-b).abs().max()) for a, b in zip(lhs, gradients)),
            node_permutation_delta=float((actual-permuted).detach().abs().max()),
            padding_delta=float((actual-padded).detach().abs().max()),
            chain_order_delta=float((actual-reordered).detach().abs().max()),
            separate_record_delta=float((actual-individual).detach().abs().max()),
            exact_state_reload=True))
    # Shared construction is checked at every prescribed seed and includes absent modalities.
    paired_tensors = 0
    for seed in range(42, 47):
        templates = {setting: models.build_model(seed, setting, "lma2").state_dict() for setting in models.SETTINGS[:2]}
        seq_reference = models.build_model(seed, "sequence_only", "none").state_dict()
        for setting, head in models.configurations():
            state = models.build_model(seed, setting, head).state_dict()
            for key, value in state.items():
                if key.startswith("sequence.") or key.startswith("sequence_projection."):
                    other = seq_reference[key]
                elif key.startswith("pool."):
                    if head not in ("lma1", "lma2", "additive2"):
                        continue
                    other = templates[setting][key]
                    if key == "pool.layers.0.order_gates":
                        other = other[:len(value)]
                else:
                    other = templates["ligand_sequence"][key]
                torch.testing.assert_close(value, other, atol=0, rtol=0)
                paired_tensors += 1
    store = SequenceStore()
    ids = [r["pdbid"] for r in store.rows[:80]]
    for seed in range(42, 47):
        generator = torch.Generator().manual_seed(20000+seed)
        order = [ids[i] for i in torch.randperm(len(ids), generator=generator)]
        chunks = list(store.chunks(order, max_records=7, max_tokens=32768))
        assert [key for group in chunks for key in group] == order
        for group in chunks:
            collated = store.collate(group)
            expected = encode_strings([store.by_id[key]["sequence"] for key in group])
            for left, right in zip(collated[:3], expected[:3]):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            assert collated[3] == len(group) and collated[0].numel() <= 32768
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True, configurations=len(checks),
        checks=checks, parameter_counts=counts, cp_rank=models.CP_RANK, paired_tensor_checks=paired_tensors,
        batching_seeds=list(range(42, 47)), exact_token_collation=True,
        sources=[dict(path=str(p), sha256=sha(p)) for p in [HERE/"sequence_models.py", HERE/"sequence_data.py",
            HERE/"sequence_encoder.py", HERE/"sequence_encoder_audit.py", Path(__file__),
            models.REVISION/"lma_revision.py", models.REVISION/"pdbbind_rerun.py",
            models.REVISION/"neural_reviewer_study_2026_09_07/models.py", models.REVISION.parent/"pdbbind_tensors_experiment.py"]],
        qualification="Independent CPU equations and all-parameter gradients, invariance, collation and accumulation are checked. GPU feasibility, training-engine checks, retained fitting and scientific performance are not certified here.")
    (HERE/"sequence_model_cpu_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "sources")}, indent=2))


if __name__ == "__main__":
    run()
