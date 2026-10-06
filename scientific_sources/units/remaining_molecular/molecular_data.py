"""Immutable molecular rows, metadata-only exact-label filter and scoped batches."""
from pathlib import Path
import csv
import json
import re
import numpy as np
import torch
import molecular_models as m

HERE=m.HERE
DATA=m.REVISION/"data/lp_pdbbind/tensors_reconstructed"
base=m.imported("remaining_core_study",m.CORE/"study.py")
EXACT=re.compile(r"(?:Kd|Ki|IC50)=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(mM|uM|nM|pM|fM|M)")


def exact_ids():
    with (DATA/"sample_manifest.csv").open(encoding="utf-8",newline="") as f:
        records=list(csv.DictReader(f))
    flags={}
    for row in records:
        match=EXACT.fullmatch(row["affinity_record"])
        flags[row["pdbid"]]=bool(match and float(match[1])>0)
    assert len(flags)==len(records)
    return flags,records


def load(split,block="full",device="cpu"):
    assert split in ("train","val","test")
    blob=torch.load(DATA/f"pdbbind_{split}.pt",map_location="cpu",weights_only=True)
    counts=blob["mask"].sum(1)
    assert torch.equal(blob["mask"],torch.arange(blob["mask"].shape[1])[None]<counts[:,None])
    assert bool((counts>0).all()) and not bool(blob["X"][~blob["mask"]].any())
    if split=="train" and block=="exact_labels":
        flags,_=exact_ids()
        keep=torch.tensor([flags[i] for i in blob["ids"]],dtype=torch.bool)
        assert int(keep.sum())==7143 and int((~keep).sum())==241
        blob={key:value[keep] if torch.is_tensor(value) else [item for item,flag in zip(value,keep) if flag]
              for key,value in blob.items()}
    return {key:value.to(device) if torch.is_tensor(value) else value for key,value in blob.items()}


def batch(blob,idx):
    mask=blob["mask"][idx]
    n=int(mask.sum(1).max())
    return blob["X"][idx,:n],mask[:,:n],blob["adj"][idx,:n,:n]


@torch.no_grad()
def predict(model,blob,mean,sd):
    model.eval()
    out=[]
    for start in range(0,len(blob["y"]),64):
        idx=torch.arange(start,min(start+64,len(blob["y"])),device=blob["y"].device)
        value=model(*batch(blob,idx))
        if value.shape!=idx.shape or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite or malformed molecular prediction")
        out.append(value.double()*sd+mean)
    return torch.cat(out).cpu().numpy()


def audit_data():
    flags,records=exact_ids()
    by_id={r["pdbid"]:r for r in records}
    all_ids={}
    for split,size in [("train",7384),("val",958),("test",2171)]:
        blob=load(split)
        assert len(blob["y"])==size and len(set(blob["ids"]))==size
        all_ids[split]=set(blob["ids"])
        assert all(by_id[key]["split"]==split for key in blob["ids"])
        if split!="train":
            assert all(flags[key] for key in blob["ids"])
    assert not all_ids["train"]&all_ids["val"] and not all_ids["train"]&all_ids["test"] and not all_ids["val"]&all_ids["test"]
    full,reduced=load("train"),load("train","exact_labels")
    keep=torch.tensor([flags[key] for key in full["ids"]],dtype=torch.bool)
    for key,value in reduced.items():
        if torch.is_tensor(value):
            assert torch.equal(value,full[key][keep])
        else:
            assert value==[item for item,flag in zip(full[key],keep) if flag]
    exclusions=[by_id[key] for key in full["ids"] if not flags[key]]
    reasons=dict(inequality=sum(any(char in r["affinity_record"] for char in ("<",">","≤","≥")) for r in exclusions),
                 approximate=sum(any(char in r["affinity_record"] for char in ("~","≈")) for r in exclusions))
    assert reasons==dict(inequality=172,approximate=69)
    statistics={}
    for label,blob in [("full",full),("exact_labels",reduced)]:
        values=blob["y"].double()
        statistics[label]=dict(rows=len(values),mean=float(values.mean()),population_sd=float(values.std(unbiased=False)))
    output=dict(passed=True,audited_utc=base.now(),excluded_training_ids=[r["pdbid"] for r in exclusions],
        retained_training_ids=reduced["ids"],exclusion_reasons=reasons,target_statistics=statistics,
        mask_convention="Verified prefix masks, retained order; effective batches trimmed by valid-atom maximum",
        sources=[dict(path=str(p),sha256=base.sha(p)) for p in [DATA/"sample_manifest.csv",DATA/"data_audit.json",
            m.REVISION/"data_interpretation_audit.py"]+[DATA/f"pdbbind_{s}.pt" for s in ("train","val","test")]])
    base.write_json(HERE/"data_prefit_audit.json",output)
    return output
