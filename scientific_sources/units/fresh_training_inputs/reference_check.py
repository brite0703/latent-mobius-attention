"""Compare the new loop with a declared CPU adaptation, then audit fresh outputs."""
from pathlib import Path
import argparse, ast, copy, hashlib, math, types
import training as t


class CPUPlacement(ast.NodeTransformer):
    def __init__(self):
        self.changes=[]

    def visit_Call(self,node):
        f=node.func
        if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Attribute) and isinstance(f.value.value,ast.Name):
            if f.value.value.id=="torch" and f.value.attr=="cuda":
                assert f.attr in {"synchronize","reset_peak_memory_stats","max_memory_allocated","empty_cache"}
                assert not node.args and not node.keywords
                self.changes.append(dict(line=node.lineno,kind="cuda_bookkeeping",before=ast.unparse(node),after="0" if f.attr=="max_memory_allocated" else "None"))
                return ast.copy_location(ast.Constant(0 if f.attr=="max_memory_allocated" else None),node)
        if isinstance(f,ast.Attribute) and f.attr=="cuda":
            assert not node.args and not node.keywords
            before=ast.unparse(node)
            f.attr="cpu"
            self.changes.append(dict(line=node.lineno,kind="tensor_placement",before=before,after=ast.unparse(node)))
        return self.generic_visit(node)

    def visit_Constant(self,node):
        if node.value=="cuda":
            self.changes.append(dict(line=node.lineno,kind="device_literal",before="cuda",after="cpu"))
            return ast.copy_location(ast.Constant("cpu"),node)
        return node


def original_reference(study,spec,folder):
    folder.mkdir(parents=True,exist_ok=False)
    path=study.folder/"campaign.py" if study.parity else study.folder.parent/"finite_neural_engine.py"
    name="train_candidate" if study.parity else "fit"
    module=t.import_at("reference_original_"+study.name,path)
    tree=ast.parse(path.read_text(encoding="utf-8"))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    transform=CPUPlacement()
    function=transform.visit(copy.deepcopy(function))
    assert transform.changes and {x["kind"] for x in transform.changes}=={"cuda_bookkeeping","tensor_placement","device_literal"}
    for node in ast.walk(function):
        assert not (isinstance(node,ast.Attribute) and node.attr=="cuda")
    adapted=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    code_text=ast.unparse(adapted)
    adapted_path=folder/"cpu_reference_function.py"
    adapted_path.write_text(code_text+"\n",encoding="utf-8")
    namespace=dict(vars(module))
    if study.parity:
        namespace["HERE"]=folder
        namespace["c"]=types.SimpleNamespace(**vars(module.c))
        namespace["c"].PATIENCE=0
    exec(compile(adapted,str(adapted_path),"exec"),namespace)
    lock_sha=t.sha(study.folder/"implementation_lock.json")
    if study.parity:
        record=namespace[name](spec,lock_sha)
    else:
        adapter=types.SimpleNamespace(**vars(study.adapter))
        adapter.HERE,adapter.PATIENCE=folder,0
        record=namespace[name](adapter,spec,lock_sha)
    assert record["status"]=="valid" and record["epochs_run"]==record["best_epoch"]==1
    return record,dict(source=path.relative_to(t.HERE).as_posix(),source_sha256=t.sha(path),
        adapted_source=str(adapted_path.relative_to(t.HERE)),adapted_source_sha256=t.sha(adapted_path),
        changes=transform.changes,fixture="Original 100-epoch cosine horizon; zero-patience cutoff after epoch one")


def preflight(studies,plan):
    comparisons=[]
    for c in plan["candidates"]:
        study,spec=studies[c["study"]],c["spec"]
        base=t.HERE/"preflight"/c["study"]/spec["id"]
        ref,adaptation=original_reference(study,spec,base/"reference")
        new=t.fit(study,spec,base/"portable",fixture=True)
        assert new["status"]=="valid" and new["epochs_run"]==new["best_epoch"]==1
        original=t.torch.load(base/"reference"/ref["checkpoint"],map_location="cpu",weights_only=True)["state_dict"]
        portable=t.torch.load(base/"portable/checkpoint.pt",map_location="cpu",weights_only=True)["state_dict"]
        assert set(original)==set(portable)
        assert all(t.torch.equal(original[k],portable[k]) for k in original)
        with t.np.load(base/"reference"/ref["validation_prediction"],allow_pickle=False) as a,t.np.load(base/"portable/validation.npz",allow_pickle=False) as b:
            t.np.testing.assert_array_equal(a["logits" if study.parity else "prediction"],b["prediction"])
            t.np.testing.assert_array_equal(a["truth"],b["truth"])
        rh,nh=ref["history"][0],new["history"][0]
        assert rh["training_cross_entropy" if study.parity else "training_loss"]==nh["training_loss"]
        assert rh["maximum_preclip_gradient_norm"]==nh["maximum_preclip_gradient_norm"]
        assert ref["best_validation_cross_entropy" if study.parity else "best_validation_loss"]==new["best_validation_loss"]
        if study.parity:
            assert rh["training_accuracy"]==nh["training_accuracy"]
        p=base/"portable/record.json"
        r=base/"reference/candidates"/(spec["id"]+".json")
        comparisons.append(dict(study=study.name,spec=spec,state_tensors=len(original),exact_state_and_validation_agreement=True,
            original_loop_adaptation=adaptation,portable_record=str(p.relative_to(t.HERE)),portable_record_sha256=t.sha(p),
            reference_record=str(r.relative_to(t.HERE)),reference_record_sha256=t.sha(r)))
    t.verify_package()
    t.write(t.HERE/"preflight/report.json",dict(passed=True,completed_utc=t.now(),binding=t.binding(),
            comparisons=comparisons,scope="Six one-epoch CPU conformance fixtures; not fresh-training efficacy evidence"))
    print(t.json.dumps(dict(preflight_passed=True,comparisons=len(comparisons))),flush=True)


def independent_metrics(kind,y,pred):
    if kind=="regression":
        delta=[float(p)-float(v) for p,v in zip(pred,y)]
        return dict(mse=math.fsum(x*x for x in delta)/len(delta),mae=math.fsum(abs(x) for x in delta)/len(delta))
    margins=[float(row[1])-float(row[0]) for row in pred]
    positive=[s for s,v in zip(margins,y) if int(v)==1]
    negative=[s for s,v in zip(margins,y) if int(v)==0]
    numerator=sum(2*int(a>b)+int(a==b) for a in positive for b in negative)
    losses=[]
    for s,v in zip(margins,y):
        z=(1-2*int(v))*s
        losses.append(max(z,0)+math.log1p(math.exp(-abs(z))))
    return dict(accuracy=sum(int((s>0)==int(v)) for s,v in zip(margins,y))/len(y),
                cross_entropy=math.fsum(losses)/len(y),auc=numerator/(2*len(positive)*len(negative)) if positive and negative else None)


def audit(studies,plan):
    selection=t.read(t.HERE/"selection.json")
    evaluation=t.read(t.HERE/"evaluation.json")
    assert selection["binding"]==t.binding() and evaluation["selection_sha256"]==t.sha(t.HERE/"selection.json")
    assert [(r["study"],r["spec"]) for r in selection["candidates"]]==[(c["study"],c["spec"]) for c in plan["candidates"]]
    records={}
    maximum=0.
    for entry in selection["candidates"]:
        path=t.HERE/entry["record"]
        assert t.sha(path)==entry["sha256"]
        r=t.verified_record(path.parent,t.binding())
        assert not r["fixture"]
        if r["status"]!="valid":
            records[(entry["study"],entry["spec"]["id"])]=r
            continue
        history=r["history"]
        assert [h["epoch"] for h in history]==list(range(1,len(history)+1))
        assert r["epochs_run"]==len(history)<=100
        minimum=float("inf"); best_epoch=0; bad=0
        for index,h in enumerate(history):
            expected_lr=entry["spec"]["lr"]*.5*(1+math.cos(math.pi*h["epoch"]/100))
            assert abs(h["next_learning_rate"]-expected_lr)<=1e-15
            scheduled=h["epoch"]==1 or h["epoch"]%5==0 or h["epoch"]==100
            assert ("validation_loss" in h)==scheduled
            if scheduled:
                if h["validation_loss"]<minimum:
                    minimum,best_epoch,bad=h["validation_loss"],h["epoch"],0
                else:
                    bad+=1
            if index<len(history)-1:
                assert bad<8
        assert len(history)==100 or bad>=8
        assert best_epoch==r["best_epoch"] and minimum==r["best_validation_loss"]
        study=studies[entry["study"]]
        model=study.build(entry["spec"])
        initial=t.torch.load(path.parent/"initial_state.pt",map_location="cpu",weights_only=True)
        assert all(t.torch.equal(v,initial[k]) for k,v in model.state_dict().items())
        checkpoint=t.torch.load(path.parent/"checkpoint.pt",map_location="cpu",weights_only=True)
        assert checkpoint["binding"]==t.binding() and checkpoint["spec"]==entry["spec"]
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        data=study.load(entry["spec"],"val")
        with t.np.load(path.parent/"validation.npz",allow_pickle=False) as saved:
            t.np.testing.assert_array_equal(saved["ids"],data["ids"])
            t.np.testing.assert_array_equal(saved["truth"],study.truth(data))
            t.np.testing.assert_array_equal(saved["prediction"],study.predict(model,entry["spec"],data))
            independent=independent_metrics(study.kind,saved["truth"],saved["prediction"])
        maximum=max(maximum,abs(study.score(independent)-minimum))
        assert abs(study.score(independent)-minimum)<=1e-11
        records[(entry["study"],entry["spec"]["id"])]=r
    assert len(evaluation["rows"])==len(selection["choices"])
    outcomes=[]
    for choice,row in zip(selection["choices"],evaluation["rows"]):
        valid=[c for c in choice["candidates"] if records[(c["study"],c["spec"]["id"])]["status"]=="valid"]
        expected=min(valid,key=lambda c:(records[(c["study"],c["spec"]["id"])]["best_validation_loss"],c["spec"]["lr"])) if valid else None
        assert choice["selected"]==expected
        if expected is None:
            assert row["status"]=="unavailable"
            continue
        assert row["spec"]==expected["spec"] and row["study"]==expected["study"]
        study=studies[row["study"]]
        folder=(t.HERE/expected["record"]).parent
        model=study.build(row["spec"])
        checkpoint=t.torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        data=study.load(row["spec"],"test")
        assert t.sha(t.HERE/row["prediction_file"])==row["prediction_sha256"]
        with t.np.load(t.HERE/row["prediction_file"],allow_pickle=False) as saved:
            t.np.testing.assert_array_equal(saved["ids"],data["ids"])
            t.np.testing.assert_array_equal(saved["truth"],study.truth(data))
            t.np.testing.assert_array_equal(saved["prediction"],study.predict(model,row["spec"],data))
            independent=independent_metrics(study.kind,saved["truth"],saved["prediction"])
        for key,value in independent.items():
            if value is None:
                assert row["metrics"][key] is None
            else:
                delta=abs(value-row["metrics"][key])
                assert delta<=1e-11
                maximum=max(maximum,delta)
        outcomes.append(dict(study=study.name,spec=row["spec"],independent_test_metrics=independent))
    t.verify_package()
    report=dict(passed=True,completed_utc=t.now(),binding=t.binding(),candidates=len(records),selected_predictors=len(outcomes),
        all_initial_states_reconstructed=True,all_saved_predictions_exactly_reloaded=True,
        maximum_independent_metric_difference=maximum,original_scientific_files_unchanged=True,
        imported_historical_model_states=0,environment=t.ENVIRONMENT,outcomes=outcomes,
        selection_sha256=t.sha(t.HERE/"selection.json"),evaluation_sha256=t.sha(t.HERE/"evaluation.json"),
        scope="Fresh CPU training pilot for three specified cases only; no manuscript efficacy update or complete-GPU-training reproduction")
    t.write(t.HERE/"final_audit.json",report)
    t.write(t.HERE/"run_control.json",dict(status="complete",finished_utc=t.now(),audit_sha256=t.sha(t.HERE/"final_audit.json")))
    print(t.json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["preflight","audit"])
    args=parser.parse_args()
    t.initialize()
    studies,plan=t.studies_and_plan()
    (preflight if args.mode=="preflight" else audit)(studies,plan)
