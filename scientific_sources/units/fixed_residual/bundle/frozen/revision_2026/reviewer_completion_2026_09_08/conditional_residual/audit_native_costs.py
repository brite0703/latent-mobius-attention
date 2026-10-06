"""Discarded full-predictor cost checks on both actual native architectures."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import torch
from torch.nn import functional as F

import costs
import execution
from models import ARMS, PairResidual
import native_costs

HERE = Path(__file__).resolve().parent
COMP = HERE.parent


def source(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_failure(call, kind=ValueError):
    try:
        call()
    except kind as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("An intentionally invalid native profiling request succeeded")


def independent_native(parent, layer_path, inputs):
    parent = deepcopy(parent).eval()
    layer = parent.get_submodule(layer_path)
    observed = []
    handle = layer.register_forward_pre_hook(lambda _, args: observed.append(args))
    try:
        with torch.inference_mode():
            f0 = parent(**inputs)
            assert len(observed) == 1
            h, mask = observed[0]
            probability = F.softmax(F.linear(F.linear(h, layer.W_k.weight, layer.W_k.bias),
                                              layer.W_H.weight, layer.W_H.bias), dim=-1)
            values = F.linear(h, layer.W_v.weight, layer.W_v.bias)
            z = probability.transpose(1,2) @ (values*mask.unsqueeze(-1))
        return f0.clone(), z.clone()
    finally:
        handle.remove()


def main():
    destination = HERE/"native_costs_cpu_audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing complete-predictor audit")
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synthetic = import_file("native_cost_synthetic_models", COMP/"first_cubic/neural_models.py")
    pocket = COMP/"receptor_context/pocket_reconstruction"
    sys.path.insert(0, str(pocket))
    import matched_models
    sources = synthetic.sources()+[pocket/"matched_models.py", pocket.parent/"sequence_models.py", pocket.parent/"sequence_encoder.py"]
    sources += [HERE/name for name in ("native_cost_specification.md", "native_costs.py", "audit_native_costs.py", "models.py", "execution.py", "selection.py", "costs.py", "models_cpu_audit.json")]
    identities = [source(path) for path in sorted(set(sources))]
    rows, cases, unit_errors = [], [], []
    generator = torch.Generator().manual_seed(2026090846)
    for domain in ("cubic", "ligand_contact"):
        molecular = domain == "ligand_contact"
        parent = matched_models.build_model(42,"ligand_contact","lma1") if molecular else synthetic.build(100,"lma1")
        layer_path, nodes, feature_count, dimension = ("pool.layers.0",57,53,8) if molecular else ("head.layers.0",12,12,12)
        inputs = dict(x=torch.randn(256,nodes,feature_count,generator=generator),
                      mask=torch.arange(nodes)[None] < (torch.arange(256)%nodes+1)[:,None])
        if molecular:
            inputs.update(adj=torch.eye(nodes).expand(256,nodes,nodes).clone(),
                          contact=torch.rand(256,nodes,216,generator=generator))
        else:
            inputs["mask"][0] = False
        parent.train()
        next(m for m in parent.modules() if isinstance(m,torch.nn.LayerNorm)).eval()
        first_parameter = next(parent.parameters())
        first_parameter.grad = torch.full_like(first_parameter,.0125)
        list(parent.parameters())[1].requires_grad_(False)
        f0, z = independent_native(parent,layer_path,inputs)
        ids = [f"manufactured:{domain}:train:{index}" for index in range(256)]
        mean, sd = (6.35,1.73) if molecular else (0.,1.)
        truth = torch.arange(256,dtype=torch.float64)/73+mean
        parent_sha = hashlib.sha256(("manufactured complete parent "+domain).encode()).hexdigest()
        cache = execution.make_cache("train",ids,f0,z,truth,mean,sd,parent_sha)
        batch = dict(split="train",ids=ids,inputs=inputs)
        residuals = {}
        for arm in ARMS:
            residual = PairResidual(arm,dimension,seed=42).train()
            residual.norm.eval()
            with torch.no_grad():
                residual.output.weight.copy_(torch.linspace(-.15,.19,residual.sizes["width"])[None])
                residual.output.bias.fill_(.037)
            next(residual.parameters()).grad = torch.full_like(next(residual.parameters()),.021)
            residuals[arm] = residual
        before_parent, before_batch, before_rng = native_costs.module_snapshot(parent), deepcopy(batch), torch.get_rng_state().clone()
        for workload in [w for w in native_costs.workload_plan() if w["domain"] == domain]:
            residual = residuals.get(workload["procedure"])
            result = native_costs.profile(parent,layer_path,residual,cache,batch,workload,warmups=1,repeats=2)
            size = workload["batch_size"]
            reference_f0, reference_z = independent_native(parent,layer_path,{k:v[:size] for k,v in inputs.items()})
            with torch.inference_mode():
                correction = torch.zeros_like(reference_f0) if residual is None else deepcopy(residual).eval()(reference_z)
            # Python scalars independently enforce float32 addition, then one
            # double-precision inverse transformation. No reporting helper used.
            expected = [(struct.unpack("f",struct.pack("f",float(a)+float(b)))[0]*sd)+mean
                        for a,b in zip(reference_f0,correction)]
            error = max(abs(a-b) for a,b in zip(expected,result["original_unit_predictions"]))
            assert error < 1e-12, (domain,workload,error)
            unit_errors.append(error)
            assert result["parameters"]["parent"] == sum(p.numel() for p in parent.parameters())
            assert result["parameters"]["residual"] == (0 if residual is None else sum(p.numel() for p in residual.parameters()))
            assert result["tensor_storage"]["native_input_bytes"] == sum(v[:size].numel()*v.element_size() for v in inputs.values())
            assert len(result["repetitions_ms"]) == 2
            rows.append(result)
        workload = next(w for w in native_costs.workload_plan() if w["domain"]==domain and w["procedure"]=="product" and w["batch_size"]==32)
        residual = residuals["product"]
        run = lambda changed_cache=cache,changed_batch=batch,changed_residual=residual: native_costs.profile(
            parent,layer_path,changed_residual,changed_cache,changed_batch,workload,warmups=1,repeats=2)
        changed = deepcopy(batch); changed["split"]="test"
        cases.append(require_failure(lambda:run(changed_batch=changed)))
        changed = deepcopy(batch); changed["ids"][0],changed["ids"][1] = changed["ids"][1],changed["ids"][0]
        cases.append(require_failure(lambda:run(changed_batch=changed)))
        changed = deepcopy(batch); changed["inputs"]["x"] = changed["inputs"]["x"].double()
        cases.append(require_failure(lambda:run(changed_batch=changed)))
        changed = deepcopy(batch); changed["inputs"]["x"][1,0,0] = float("nan")
        cases.append(require_failure(lambda:run(changed_batch=changed)))
        altered_cache = execution.make_cache("train",ids,f0+.1,z,truth,mean,sd,parent_sha)
        cases.append(require_failure(lambda:run(changed_cache=altered_cache)))
        altered_z = execution.make_cache("train",ids,f0,z+.1,truth,mean,sd,parent_sha)
        cases.append(require_failure(lambda:run(changed_cache=altered_z)))
        cases.append(require_failure(lambda:run(changed_residual=residuals["additive"])))
        invalid_cache = deepcopy(cache); invalid_cache.f0[0] += .1
        cases.append(require_failure(lambda:run(changed_cache=invalid_cache)))
        original_prediction = native_costs.native_prediction
        def injected_failure(*args,**kwargs):
            raise RuntimeError("Injected native-inference failure after copies and hooks are constructed")
        native_costs.native_prediction = injected_failure
        try:
            cases.append(require_failure(run,RuntimeError))
        finally:
            native_costs.native_prediction = original_prediction
        assert costs.same(before_parent,native_costs.module_snapshot(parent))
        assert costs.same(before_batch,batch) and torch.equal(before_rng,torch.get_rng_state())
        cache.verify()
        print(json.dumps(dict(domain=domain,complete_workloads=12,invalid_or_exception_cases=9)),flush=True)
    assert len(rows)==24 and len(cases)==18
    for bound in identities:
        assert source(bound["path"])==bound
    receipt = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        scope="Discarded manufactured-input audit of native complete-predictor cost kernels",
        architectures=2,complete_workloads_checked=len(rows),invalid_or_exception_cases=len(cases),
        maximum_independent_original_unit_difference=max(unit_errors),
        maximum_native_cache_prediction_difference=max(r["native_cache_max_delta"]["f0"] for r in rows),
        maximum_native_cache_bucket_difference=max(r["native_cache_max_delta"]["z"] for r in rows),
        planned_complete_inference_outcomes=180,planned_cached_residual_outcomes=255,
        source_models_inputs_cache_rng_and_gradients_preserved=True,
        retained_checkpoints_loaded=False,datasets_loaded=False,retained_timing_results=False,
        cuda_initialized=torch.cuda.is_initialized(),sources=identities,cases=cases,discarded_workloads=rows,
        remaining="Actual source-bound fitting input/checkpoint adapter, complete parent bundle, preceding studies, any retained activation, final selections and uncontended scientific timings")
    destination.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("sources","cases","discarded_workloads")}),flush=True)


if __name__=="__main__":
    main()
