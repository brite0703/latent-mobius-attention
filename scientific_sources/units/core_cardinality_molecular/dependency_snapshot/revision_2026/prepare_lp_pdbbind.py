"""Build a documented ligand-only reconstruction from official LP-PDBBind CSV.

This does not recreate the missing remote feature pipeline. No 3D structures or
protein features are used. Splits and labels are read, never inferred or refit.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

ELEMENTS=[1,5,6,7,8,9,11,12,14,15,16,17,19,20,26,30,35,53]
HYBRIDS=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,
         Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]


def onehot(value, choices):
    return [float(value==v) for v in choices]+[float(value not in choices)]


def atom_features(atom):
    return (onehot(atom.GetAtomicNum(),ELEMENTS)+onehot(atom.GetDegree(),list(range(7)))+
            onehot(atom.GetFormalCharge(),list(range(-3,4)))+onehot(atom.GetTotalNumHs(),list(range(5)))+
            onehot(atom.GetHybridization(),HYBRIDS)+onehot(atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",["R","S"])+
            [float(atom.GetIsAromatic()),float(atom.IsInRing()),atom.GetMass()/100])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv,index_col=0)
    assert df.index.is_unique
    for col in ("CL1","CL2","covalent","remove_for_balancing_val"):
        assert df[col].dtype==bool, f"Non-Boolean column: {col}"
    selected=df[(~df.covalent)&(((df.new_split=="train")&df.CL1)|
                  (df.new_split.isin(["val","test"])&df.CL2))].copy()
    # The optional validation rebalancing flag is retained and reported, not
    # silently applied: its purpose is not established by the downloaded README.
    records=[]
    failures=[]
    for pdbid,row in selected.iterrows():
        smiles=row.smiles
        if not isinstance(smiles,str) or not smiles:
            failures.append(dict(pdbid=pdbid,split=row.new_split,reason="missing_smiles"));continue
        mol=Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms()==0:
            failures.append(dict(pdbid=pdbid,split=row.new_split,reason="rdkit_parse_failed"));continue
        Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
        x=np.asarray([atom_features(atom) for atom in mol.GetAtoms()],dtype=np.float32)
        a=Chem.GetAdjacencyMatrix(mol).astype(np.float32)+np.eye(len(x),dtype=np.float32)
        inv=1/np.sqrt(a.sum(axis=1))
        normalized=a*inv[:,None]*inv[None,:]
        records.append(dict(pdbid=pdbid,split=row.new_split,x=x,adj=normalized,y=float(row.value),
            canonical_smiles=Chem.MolToSmiles(mol,isomericSmiles=True),seq=row.seq,
            remove_for_balancing_val=bool(row.remove_for_balancing_val),affinity_record=row["kd/ki"]))
    assert records and all(np.isfinite(r["y"]) for r in records)
    max_atoms=max(len(r["x"]) for r in records)
    d_in=records[0]["x"].shape[1]
    inventory=[]
    split_sets={}
    summary={}
    for split in ("train","val","test"):
        rows=[r for r in records if r["split"]==split]
        count=len(rows)
        split_sets[split]={r["pdbid"] for r in rows}
        x=np.zeros((count,max_atoms,d_in),dtype=np.float32)
        mask=np.zeros((count,max_atoms),dtype=bool)
        adj=np.zeros((count,max_atoms,max_atoms),dtype=np.float32)
        for i,row in enumerate(rows):
            n=len(row["x"])
            x[i,:n]=row["x"];mask[i,:n]=True;adj[i,:n,:n]=row["adj"]
            inventory.append({key:row[key] for key in ("pdbid","split","y","canonical_smiles","remove_for_balancing_val","affinity_record")}|dict(atoms=n))
        blob=dict(X=torch.from_numpy(x),mask=torch.from_numpy(mask),adj=torch.from_numpy(adj),
                  y=torch.tensor([r["y"] for r in rows],dtype=torch.float32),ids=[r["pdbid"] for r in rows])
        assert blob["y"].ndim==1 and bool(blob["mask"].any(dim=1).all())
        torch.save(blob,args.output_dir/f"pdbbind_{split}.pt")
        summary[split]=dict(n=count,label_mean=float(blob["y"].mean()),label_sd=float(blob["y"].std()),
            mean_atoms=float(mask.sum(axis=1).mean()),min_atoms=int(mask.sum(axis=1).min()),max_atoms=int(mask.sum(axis=1).max()),
            flagged_for_val_rebalancing=sum(r["remove_for_balancing_val"] for r in rows))
    for first,second in (("train","val"),("train","test"),("val","test")):
        assert not split_sets[first]&split_sets[second]
    overlap={}
    for first,second in (("train","val"),("train","test"),("val","test")):
        sets={s:{r["canonical_smiles"] for r in records if r["split"]==s} for s in (first,second)}
        seqs={s:{r["seq"] for r in records if r["split"]==s} for s in (first,second)}
        overlap[f"{first}_{second}"]=dict(identical_ligand_smiles=len(sets[first]&sets[second]),
                                         identical_protein_sequences=len(seqs[first]&seqs[second]))
    pd.DataFrame(inventory).to_csv(args.output_dir/"sample_manifest.csv",index=False)
    (args.output_dir/"exclusions.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    unit_scale={"M":1,"mM":1e-3,"uM":1e-6,"nM":1e-9,"pM":1e-12,"fM":1e-15}
    label_checks=[]
    for row in records:
        match=re.fullmatch(r"(?:Kd|Ki|IC50)=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(mM|uM|nM|pM|fM|M)",row["affinity_record"])
        if match and float(match[1])>0:
            label_checks.append(abs(row["y"]+math.log10(float(match[1])*unit_scale[match[2]])))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rdkit=rdBase.rdkitVersion,
        source_rows=len(df),selected_before_parsing=len(selected),excluded=len(failures),
        split_protocol="published new_split; noncovalent CL1 train; noncovalent CL2 validation and test; validation rebalancing flag retained",
        label="unaltered numeric value column in pK units; no second log transform",input="ligand atom features from supplied SMILES only",
        no_protein_or_3d=True,max_atoms=max_atoms,feature_dimension=d_in,
        atom_features=dict(elements=ELEMENTS,degree=list(range(7)),formal_charge=list(range(-3,4)),hydrogen_count=list(range(5)),
            hybridization=[str(h) for h in HYBRIDS],chirality="CIP R/S/other; avoids atom-order-dependent CW/CCW tags",other="each categorical field has an other bin; aromatic, ring, mass/100"),
        graph="unweighted undirected chemical bonds plus self loops; symmetric degree normalization; padding has zero adjacency",
        splits=summary,exact_cross_split_overlap=overlap,
        affinity_unit_checks=dict(parseable_exact_records=len(label_checks),max_absolute_pK_discrepancy=max(label_checks),
                                  discrepancies_exceeding_001=sum(v>.01 for v in label_checks)),
        limitation="Reconstructed ligand-only benchmark. Original remote filters and features are unavailable; historical RMSE is not an acceptance target. Exact-identity overlap does not measure all similarity leakage.")
    (args.output_dir/"data_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
