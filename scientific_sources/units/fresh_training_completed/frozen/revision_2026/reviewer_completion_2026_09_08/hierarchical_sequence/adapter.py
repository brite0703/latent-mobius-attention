"""Fixed hierarchy datasets, ordinary sequence inference and train-only count controls."""
from pathlib import Path
import importlib.util
import json
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def local(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g=local("hierarchy_generator","generator.py")
m=local("hierarchy_neural_models","neural_models.py")
u=m.pm.import_file("hierarchy_utilities",HERE.parent/"synthetic_parity/parity_common.py")
TASKS,SEEDS,HEADS=["depth2","depth3","depth4"],list(range(200,210)),m.HEADS
KIND,LRS="classification",[.0003,.001]
EPOCHS,BATCH,PATIENCE,CLIP=100,256,8,10.
ORDER_SEED=2026090808
SIZES=m.SIZES
EVALUATION_SPLITS=["test"]
PAIRS=[("lma2","lma1"),("lma3","lma1"),("lma3","lma2"),
       ("lma2","transformer"),("lma3","transformer"),
       ("lma2","deepsets_wide"),("lma3","deepsets_wide"),
       ("lma2","cp_pool"),("lma3","cp_pool")]
SLICES=dict(train=slice(0,4800),val=slice(4800,5400),test=slice(5400,6000))


def data_path(task,seed):
    return HERE/"data"/f"{task}_seed{seed}.npz"


def load(spec,split,device="cpu"):
    sl=SLICES[split]
    with np.load(data_path(spec["task"],spec["seed"])) as z:
        x=torch.as_tensor(z["x"][sl].copy(),dtype=torch.float32,device=device)
        y=torch.as_tensor(z["y"][sl].copy(),dtype=torch.long,device=device)
        count=z["count"][sl].copy()
    return dict(inputs=(x,),y=y,count=count,ids=np.arange(6000)[sl])


def build(spec):
    return m.build(spec["seed"],spec["head"],3**int(spec["task"][-1]))


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(),HERE/"generator.py",HERE/"neural_audit.py",
        HERE.parent/"synthetic_parity/audit.py",HERE.parent/"synthetic_parity/campaign.py"]+m.sources()))


def lock_files():
    return [HERE/name for name in ["protocol.md","neural_implementation.md","generator_audit.json","data_manifest.json"]]+[
        data_path(task,seed) for task in TASKS for seed in SEEDS]


def prepare():
    assert json.loads((HERE/"generator_audit.json").read_text())["passed"]
    (HERE/"data").mkdir(exist_ok=True)
    rows=[]
    for task in TASKS:
        depth=int(task[-1])
        for seed in SEEDS:
            path=data_path(task,seed)
            expected=g.generate(depth,seed)
            if not path.exists():
                np.savez_compressed(path,**expected)
            with np.load(path) as z:
                for key,value in expected.items():
                    np.testing.assert_array_equal(z[key],value)
            keys={split:[bytes(x) for x in expected["x"][sl]] for split,sl in SLICES.items()}
            rows.append(dict(task=task,depth=depth,n=3**depth,seed=seed,file=str(path.relative_to(HERE)),sha256=u.sha(path),
                split_sizes={key:len(value) for key,value in keys.items()},
                unique_sequences={key:len(set(value)) for key,value in keys.items()},
                validation_sequences_seen_in_training=sum(key in set(keys["train"]) for key in keys["val"]),
                test_sequences_seen_in_training=sum(key in set(keys["train"]) for key in keys["test"]),
                test_sequences_seen_in_training_or_validation=sum(key in set(keys["train"]+keys["val"]) for key in keys["test"]),
                class_counts={key:np.bincount(expected["y"][sl],minlength=2).tolist() for key,sl in SLICES.items()},
                count_histograms={key:np.bincount(expected["count"][sl],minlength=3**depth+1).tolist() for key,sl in SLICES.items()}))
    u.write_json(HERE/"data_manifest.json",dict(passed=True,prepared_utc=u.now(),datasets=rows,
        qualification="IID rows may repeat. All models receive leaf positions. No population neural evaluation or count canonicalization."))


def diagnostics(model,data):
    if not hasattr(model,"head") or not hasattr(model.head,"factor"):
        return None
    counts=dict(product_entries=0,finite_entries=0,exact_zero_products=0,tanh_derivative_below_1e_6=0,
                tanh_exactly_abs_one=0,exact_zero_factors=0)
    maximum=0.
    model.eval()
    with torch.no_grad():
        for start in range(0,len(data["y"]),BATCH):
            h=model.encode(data["inputs"][0][start:start+BATCH])
            factors=model.head.factor(h)
            product=factors.double().prod(1)
            bounded=product.tanh()
            finite=torch.isfinite(product)
            counts["product_entries"]+=product.numel()
            counts["finite_entries"]+=int(finite.sum())
            counts["exact_zero_products"]+=int((product==0).sum())
            counts["tanh_derivative_below_1e_6"]+=int(((1-bounded.square())<1e-6).sum())
            counts["tanh_exactly_abs_one"]+=int((bounded.abs()==1).sum())
            counts["exact_zero_factors"]+=int((factors==0).sum())
            if bool(finite.any()):
                maximum=max(maximum,float(product[finite].abs().max()))
    return dict(**counts,maximum_absolute_product=maximum)


def reference(spec):
    train=load(spec,"train")
    n=3**int(spec["task"][-1])
    totals=np.bincount(train["count"],minlength=n+1)
    positives=np.bincount(train["count"],weights=train["y"].numpy(),minlength=n+1)
    probability=(positives+1)/(totals+2)
    logits=np.column_stack([np.log1p(-probability),np.log(probability)])
    return logits,totals


def evaluate_references():
    rows=[]
    for task in TASKS:
        for seed in SEEDS:
            spec=dict(task=task,seed=seed)
            logits,totals=reference(spec)
            test=load(spec,"test")
            prediction=logits[test["count"]]
            path=HERE/"test_predictions"/f"{task}_count_lookup_seed{seed}.npz"
            np.savez_compressed(path,ids=test["ids"],truth=test["y"].numpy(),prediction=prediction,
                                logits_by_count=logits,training_count_frequency=totals)
            rows.append(dict(**spec,head="count_lookup",test=u.metrics(test["y"].numpy(),prediction),
                prediction_file=str(path.relative_to(HERE)),prediction_sha256=u.sha(path)))
    return rows


def audit_references(rows):
    from sklearn.metrics import roc_auc_score
    assert len(rows)==30
    for row in rows:
        path=HERE/row["prediction_file"]
        assert u.sha(path)==row["prediction_sha256"]
        logits,totals=reference(row)
        test=load(row,"test")
        with np.load(path) as z:
            np.testing.assert_array_equal(z["prediction"],logits[test["count"]])
            np.testing.assert_array_equal(z["training_count_frequency"],totals)
            score=z["prediction"][:,1]-z["prediction"][:,0]
            ce=math.fsum(float(np.logaddexp(0.,(1-2*int(y))*s)) for y,s in zip(z["truth"],score))/len(score)
            assert abs(ce-row["test"]["cross_entropy"])<1e-12
            assert abs(float(roc_auc_score(z["truth"],score))-row["test"]["auc"])<1e-12
            assert sum(int((s>0)==y) for s,y in zip(score,z["truth"]))/len(score)==row["test"]["accuracy"]
