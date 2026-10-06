"""Profiler dense arithmetic subtotal checked against independent shape formulas."""
import json
import math
import torch
import component_models as gm
import study as base

HERE = gm.HERE


def expected_macs(config, n, batch, d_in=53, d=16, v=8):
    m, k = config["M"], config["k"]
    memory = sum(math.comb(m, r) for r in range(1, k+1))
    graph_encoder = n*n*d_in+n*d_in*d
    head = (7*n*d*d+n*d*v+n*d*m+n*m*v+m*v*v*k*(k+1)//2+
            memory*(v*v+v*d)+2*n*memory*d+d*d+d)
    if config["routing"] == "uniform":
        head -= n*d*d+n*d*m
    if not config["query"]:
        head -= n*d*d+2*n*memory*d
    if config["routing"] == "fixed":
        graph_encoder *= 2
    return batch*(graph_encoder+head)


def main():
    torch.set_num_threads(4)
    blob = torch.load(base.DATA/"pdbbind_train.pt", weights_only=True, map_location="cpu")
    n = int(blob["mask"][:64].sum(1).max())
    x, mask, adj = blob["X"][:64, :n], blob["mask"][:64, :n], blob["adj"][:64, :n, :n]
    rows = []
    operators = {"aten::mm", "aten::bmm", "aten::addmm", "aten::addbmm"}
    for name, config in gm.CONFIGS.items():
        for batch in (1, 64):
            model = gm.build_model(42, name, 1).cpu().eval()
            with torch.no_grad():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            record_shapes=True, with_flops=True) as profiler:
                    output = model(x[:batch], mask[:batch], adj[:batch])
            assert bool(torch.isfinite(output).all())
            counts = {event.key: dict(calls=event.count, flops=int(event.flops or 0))
                      for event in profiler.key_averages() if event.key in operators}
            subtotal = sum(r["flops"] for r in counts.values())
            analytic_macs = expected_macs(config, n, batch)
            assert subtotal == 2*analytic_macs, (name, batch, subtotal, analytic_macs)
            m, k = config["M"], config["k"]
            memory = sum(math.comb(m, r) for r in range(1, k+1))
            rows.append(dict(head=name, batch=batch, padded_atoms=n,
                valid_atoms=mask[:batch].sum(1).tolist(), memory_tokens=memory,
                total_parameters=gm.parameter_count(model), trainable_parameters=gm.trainable_count(model),
                profiler_dense_flop_subtotal=subtotal, independently_counted_dense_macs=analytic_macs,
                hadamard_multiplications=batch*8*sum((r-1)*math.comb(m, r) for r in range(1, k+1)),
                memory_tensor_float32_mib=batch*memory*16*4/2**20,
                attention_score_tensor_float32_mib=batch*n*memory*4/2**20 if config["query"] else 0.,
                profiled_operators=counts))
    base.write_json(HERE/"arithmetic_audit.json", dict(completed_utc=base.now(), passed=True, rows=rows,
        boundary="Whole preloaded model on the same padded first-64 training-input shape as the GPU profiles. Untrained weights suffice because these dense shapes do not depend on fitted values.",
        convention="One multiply-accumulate counted as two FLOPs. Profiler mm/bmm/addmm/addbmm subtotal independently reconciled by layer dimensions.",
        exclusions="Not total FLOPs: excludes bias additions, normalization, activations, softmax, masking/reductions, optimizer/backward and memory movement. Separate Hadamard and tensor-size counts are disclosed; theoretical single tensors are not measured peak VRAM.",
        source_sha256=base.sha(__file__)))
    print(json.dumps(dict(passed=True, workloads=len(rows), dense_counts_reconciled=True)), flush=True)


if __name__ == "__main__":
    main()
