"""Separate locked component campaign; imported core training source stays unchanged."""
import argparse
import json
import platform
import sys
import numpy as np
import torch
import component_models as gm
import study as base
import fit

HERE=gm.HERE


def specifications():
    rows=[dict(id=f"gcn1_{head}_seed{seed}_lr{j}",head=head,depth=1,seed=seed,lr=lr,lr_index=j)
          for head in gm.HEADS for seed in base.SEEDS for j,lr in enumerate(base.LRS)]
    return [rows[i] for i in np.random.default_rng(2026090802).permutation(len(rows))]


def lock():
    base.verify_lock()
    assert not (HERE/"implementation_lock.json").exists()
    audit=json.loads((HERE/"prefit_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    for item in audit["sources"]:
        assert base.sha(item["path"])==item["sha256"],item["path"]
    files=[HERE/"protocol.md",HERE/"prefit_audit.json",HERE/"web_review_25_disposition.md",
           gm.CORE/"implementation_lock.json",gm.CORE/"study_manifest.json"]
    files += [__import__("pathlib").Path(item["path"]) for item in audit["sources"]]
    record=dict(locked_utc=base.now(),purpose="Reviewer-directed development-informed routing/query/width comparison",
        candidates=specifications(),candidate_count=190,selection_count=95,
        heads=gm.HEADS,configurations=gm.CONFIGS,seeds=base.SEEDS,learning_rates=base.LRS,
        training="Separate fit.py copied from prior candidate loop, with explicit trainable-parameter optimizer and prediction/penalty objective and records",
        optimizer="AdamW",weight_decay=1e-4,batch_size=256,predict_batch_size=64,
        maximum_epochs=150,cosine_T_max=150,gradient_clip_norm=10.,balance_loss_weights="Per configuration: 0, 0.01 or 0.1",
        target_scaling="train-only float64 mean and population SD; standardized float32 targets",
        validation="epoch 1, every 5, final; stop after 8 unimproved checks; earliest strict within-candidate minimum",
        selection="Minimum finite validation RMSE; lower numerical LR on ties; no replacement for failed families/seeds",
        test_status="Repeatedly development-used ligand-only split; exploratory, no confirmatory inference",
        pairing="Seed-specific untrained M16 k3 template; retained tensors, first k gates, first M routing rows; own combination-index buffers",
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
                     gpu=torch.cuda.get_device_name(),cpu_threads=4,tf32=False),
        files=[dict(path=str(p),sha256=base.sha(p)) for p in files])
    base.write_json(HERE/"implementation_lock.json",record)
    print(json.dumps(dict(stage="component_implementation_locked",candidates=190,selections=95,
                         sha256=base.sha(HERE/"implementation_lock.json"))),flush=True)


def verify():
    old=base.HERE
    base.HERE=gm.CORE
    try:
        base.verify_lock()
    finally:
        base.HERE=old
    record=json.loads((HERE/"implementation_lock.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert base.sha(item["path"])==item["sha256"],item["path"]
    assert record["candidates"]==specifications()
    return record


def train():
    record=verify()
    assert not (HERE/"selection_lock.json").exists()
    assert not (HERE/"evaluation.json").exists()
    base.HERE=HERE
    training,validation=base.load_split("train"),base.load_split("val")
    assert not set(training["ids"]) & set(validation["ids"])
    mean=float(training["y"].double().mean())
    sd=float(training["y"].double().std(unbiased=False))
    lock_sha=base.sha(HERE/"implementation_lock.json")
    for spec in record["candidates"]:
        fit.train_candidate(spec,training,validation,mean,sd,lock_sha)
        done=[json.loads(p.read_text(encoding="utf-8")) for p in (HERE/"candidates").glob("*.json")]
        base.write_json(HERE/"progress.json",dict(updated_utc=base.now(),completed=len(done),total=190,
            valid=sum(r["status"]=="valid" for r in done),failed=sum(r["status"]=="failed" for r in done),
            aggregate_fit_seconds=sum(r["wall_seconds"] for r in done),last_id=spec["id"]))
    selections,files=[],[]
    assert len(done)==190
    for head in gm.HEADS:
        for seed in base.SEEDS:
            rows=[r for r in done if r["spec"]["head"]==head and r["spec"]["seed"]==seed]
            assert len(rows)==2 and all(r["implementation_lock_sha256"]==lock_sha for r in rows)
            valid=[r for r in rows if r["status"]=="valid"]
            chosen=min(valid,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if valid else None
            selections.append(dict(depth=1,head=head,seed=seed,selected_id=chosen["spec"]["id"] if chosen else None,
                validation_rmse=chosen["best_validation_rmse"] if chosen else None,
                candidate_ids=[r["spec"]["id"] for r in rows],
                checkpoint_sha256=chosen["checkpoint_sha256"] if chosen else None))
            files += [dict(path=f'candidates/{r["spec"]["id"]}.json',
                           sha256=base.sha(HERE/"candidates"/(r["spec"]["id"]+".json"))) for r in rows]
    base.write_json(HERE/"selection_lock.json",dict(locked_utc=base.now(),selections=selections,
        candidate_records=files,implementation_lock_sha256=lock_sha,
        qualification="All 95 within-procedure choices fixed before this campaign's test scoring; prior test reuse remains"))
    print(json.dumps(dict(stage="component_all_selections_locked",count=len(selections))),flush=True)


def verify_selection():
    verify()
    record=json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    assert record["implementation_lock_sha256"]==base.sha(HERE/"implementation_lock.json")
    assert len(record["selections"])==95 and len(record["candidate_records"])==190
    for item in record["candidate_records"]:
        assert base.sha(HERE/item["path"])==item["sha256"]
    return record


def evaluate():
    selections=verify_selection()
    assert not (HERE/"evaluation.json").exists()
    test=base.load_split("test")
    y=test["y"].double().cpu().numpy()
    output=HERE/"test_predictions"
    output.mkdir(exist_ok=True)
    rows=[]
    for choice in selections["selections"]:
        if choice["selected_id"] is None:
            rows.append(dict(**choice,status="all_candidates_failed",test=None))
            continue
        candidate=json.loads((HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
        path=HERE/candidate["checkpoint"]
        assert base.sha(path)==choice["checkpoint_sha256"]
        checkpoint=torch.load(path,weights_only=True,map_location="cpu")
        assert checkpoint["implementation_lock_sha256"]==base.sha(HERE/"implementation_lock.json")
        model=gm.build_model(choice["seed"],choice["head"],1).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        prediction=base.predict(model,test,checkpoint["target_mean"],checkpoint["target_sd"])
        destination=output/(choice["selected_id"]+".npz")
        np.savez_compressed(destination,ids=np.asarray(test["ids"]),truth=y,prediction=prediction)
        rows.append(dict(**choice,status="valid",test=base.metrics(y,prediction),
            total_parameters=gm.original_models.parameter_count(model),best_epoch=candidate["best_epoch"],
            trainable_parameters=gm.trainable_count(model),
            frozen_parameters=gm.parameter_count(model)-gm.trainable_count(model),
            selected_epochs=candidate["epochs_run"],selected_lr=candidate["spec"]["lr"],
            selected_gate_coefficients=model.head.layers[0].order_gates.detach().cpu().tolist(),
            prediction_file=str(destination.relative_to(HERE)),prediction_sha256=base.sha(destination)))
        del model
    base.write_json(HERE/"evaluation.json",dict(evaluated_utc=base.now(),rows=rows,
        selection_lock_sha256=base.sha(HERE/"selection_lock.json")))
    print(json.dumps(dict(stage="component_evaluation_complete",selections=len(rows))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["lock","train","evaluate"])
    args=parser.parse_args()
    base.configure()
    {"lock":lock,"train":train,"evaluate":evaluate}[args.stage]()

