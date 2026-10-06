"""Follow-up to the preserved fitting-only entropy observation."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, math
import numpy as np
import torch
import diagnose as d

HERE = Path(__file__).resolve().parent


def mean_reference(model, x, mask, adj):
    layer = model.head.layers[0]
    h = model.encoder(x, mask, adj)
    pi = (layer.W_H(layer.W_k(h))).softmax(-1)
    z = pi.transpose(1,2) @ (layer.W_v(h)*mask.unsqueeze(-1))
    memories = []
    for order in range(1,layer.k+1):
        legs = layer.interaction_projs[order-1](z).chunk(order,-1)
        tuples = getattr(layer,f"combos_{order}")
        result = legs[0][:,tuples[:,0]]
        for j in range(1,order):
            value = legs[j][:,tuples[:,j]]
            result = result*value if layer.feature_mode=="product" else result+value
        if layer.feature_mode=="additive":
            result = result/order
        memories.append(layer.order_gates[order-1]*layer.interaction_mlps[order-1](result))
    if isinstance(layer,d.gm.SeparatedOrders):
        readout = torch.stack([memory.mean(1) for memory in memories]).sum(0)
    else:
        readout = torch.cat(memories,1).mean(1)
    return d.finish(model,h,mask,readout[:,None].expand(-1,x.shape[1],-1))


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = d.read(HERE/"diagnosis.json")
    for item in original["sources"]:
        assert d.sha(item["path"])==item["sha256"]
    destination = HERE/"uniform_query_diagnosis.json"
    if destination.exists():
        old=d.read(destination)
        for item in old["sources"]:
            assert d.sha(item["path"])==item["sha256"]
        for item in old["artifacts"]:
            assert d.sha(HERE/item["path"])==item["sha256"]
        print(json.dumps({"preserved_existing_uniform_diagnosis":True}))
        return
    blob=torch.load(d.DATA,map_location="cpu",weights_only=True)
    by_id={key:i for i,key in enumerate(blob["ids"])}
    indices=torch.tensor([by_id[key] for key in original["input_ids"]])
    mask=blob["mask"][indices]
    end=int(mask.sum(1).max())
    x,mask,adj=blob["X"][indices,:end],mask[:,:end],blob["adj"][indices,:end,:end]
    truth=blob["y"][indices].double().numpy()
    rows,per_graph=[],[]
    for previous in original["rows"]:
        candidate=d.read(d.GATE/"candidates"/(previous["selected_id"]+".json"))
        path=d.GATE/candidate["checkpoint"]
        assert d.sha(path)==previous["checkpoint_sha256"]
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        model=d.gm.build(previous["seed"],previous["head"],1).eval()
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        layer=model.head.layers[0]
        with torch.no_grad():
            baseline=model(x,mask,adj)
            layer.W_q.weight.zero_()
            layer.W_q.bias.zero_()
            uniform=model(x,mask,adj)
            reference=mean_reference(model,x,mask,adj)
            torch.testing.assert_close(uniform,reference,rtol=3e-5,atol=3e-5)
            model.load_state_dict(checkpoint["state_dict"],strict=True)
            restored=model(x,mask,adj)
        torch.testing.assert_close(restored,baseline,rtol=0,atol=0)
        assert all(torch.equal(value,checkpoint["state_dict"][key]) for key,value in model.state_dict().items())
        def scale(value):
            return value.double().numpy()*checkpoint["target_sd"]+checkpoint["target_mean"]
        before,after=scale(baseline),scale(uniform)
        assert np.isfinite(after).all()
        change=float(np.sqrt(np.mean((after-before)**2)))
        removed=previous["first_only_prediction_change_rms"]
        row=dict(head=previous["head"],seed=previous["seed"],selected_id=previous["selected_id"],
            uniform_prediction_change_rms=change,
            uniform_to_branch_removal_rms_ratio=change/removed if removed else None,
            uniform_minus_full_fitting_mse=float(np.mean((after-truth)**2-(before-truth)**2)),
            native_mean_reference_max_normalized_delta=float((uniform-reference).abs().max()),
            restored_max_prediction_delta=float((restored-baseline).abs().max()))
        rows.append(row)
        per_graph.extend(dict(head=row["head"],seed=row["seed"],id=key,truth=float(truth[i]),
            full_prediction=float(before[i]),uniform_prediction=float(after[i]))
            for i,key in enumerate(original["input_ids"]))
        print(json.dumps({"checked":len(rows),"total":35,"head":row["head"],"seed":row["seed"],
                          "uniform_change_rms":change}),flush=True)
    assert not torch.cuda.is_initialized()
    groups=[]
    for head in d.gm.HEADS:
        subset=[row for row in rows if row["head"]==head]
        groups.append(dict(head=head,uniform_prediction_change_rms=d.describe([r["uniform_prediction_change_rms"] for r in subset]),
            uniform_minus_full_fitting_mse=d.describe([r["uniform_minus_full_fitting_mse"] for r in subset]),
            fitting_mse_increases=sum(r["uniform_minus_full_fitting_mse"]>0 for r in subset),
            fitting_mse_decreases=sum(r["uniform_minus_full_fitting_mse"]<0 for r in subset)))
    path=HERE/"uniform_query_records.csv"
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_graph[0]))
        writer.writeheader();writer.writerows(per_graph)
    sources=[Path(__file__),HERE/"uniform_query_plan.md",HERE/"diagnosis.json",HERE/"diagnose.py"]
    result=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,groups=groups,
        fitting_records=256,selected_predictors=35,per_record_rows=len(per_graph),cuda_initialized=False,
        cpu_threads=1,optimizer_created=False,validation_or_test_tensors_loaded=False,
        sources=[dict(path=str(p),sha256=d.sha(p)) for p in sources],
        artifacts=[dict(path=path.name,sha256=d.sha(path))],
        qualification="Exploratory fixed-checkpoint query perturbation motivated by the preceding fitting-only entropy result. No refitting, model selection, held-out performance claim or cause of generalization is established.")
    d.write(destination,result)
    print(json.dumps({"complete":True,"predictors":35,"fitting_records":256,"cuda_initialized":False}),flush=True)


if __name__=="__main__":
    main()
