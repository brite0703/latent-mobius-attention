"""Independent checks of reconstructed chemistry, tensor schemas, and GPU masks."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem
from prepare_lp_pdbbind import atom_features
from pdbbind_rerun import build_model,validate_blob


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    chemical=[]
    for smiles in ("C[C@H](O)C(=O)O","N[C@@H](Cc1ccccc1)C(=O)O","F[C@](Cl)(Br)I","c1ccncc1"):
        mol=Chem.MolFromSmiles(smiles);Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
        permutation=list(reversed(range(mol.GetNumAtoms())))
        changed=Chem.RenumberAtoms(mol,permutation);Chem.AssignStereochemistry(changed,cleanIt=True,force=True)
        original=np.asarray([atom_features(a) for a in mol.GetAtoms()])
        moved=np.asarray([atom_features(a) for a in changed.GetAtoms()])
        assert np.array_equal(original[permutation],moved)
        assert np.array_equal(Chem.GetAdjacencyMatrix(mol)[permutation][:,permutation],Chem.GetAdjacencyMatrix(changed))
        chemical.append(dict(smiles=smiles,atom_renumbering_passed=True))
    counts={};ids={};file_hashes={}
    for split in ("train","val","test"):
        path=args.data_dir/f"pdbbind_{split}.pt"
        blob=torch.load(path,weights_only=True,map_location="cpu");validate_blob(blob)
        counts[split]=len(blob["y"]);ids[split]=set(blob["ids"])
        file_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert not ids["train"]&ids["val"] and not ids["train"]&ids["test"] and not ids["val"]&ids["test"]
    try:validate_blob(blob|{"y":blob["y"].unsqueeze(1)})
    except AssertionError:pass
    else:raise AssertionError("Unsafe label shape accepted")
    del blob
    rows=[]
    for name in ("mean","deepsets","pma","transformer","lma1","lma2","lma3"):
        opts=SimpleNamespace(d_model=16,d_latent=8,buckets=8,backbone="mlp",graph_depth=1)
        model=build_model(42,name,53,opts).cuda().eval()
        torch.manual_seed(182)
        x=torch.randn(3,5,53,device="cuda");mask=torch.ones(3,5,dtype=torch.bool,device="cuda")
        padded=torch.cat([x,torch.randn(3,8,53,device="cuda")],1).requires_grad_(True)
        padded_mask=torch.cat([mask,torch.zeros(3,8,dtype=torch.bool,device="cuda")],1)
        with torch.no_grad():expected=model(x,mask)
        actual=model(padded,padded_mask)
        delta=float((actual-expected).abs().max())
        assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-5)
        actual.square().sum().backward()
        gradient=float(padded.grad[~padded_mask].abs().max())
        assert gradient==0
        rows.append(dict(model=name,fp32_cuda_padding_delta=delta,masked_input_gradient_max=gradient))
        del model,padded,x
        torch.cuda.empty_cache()
    # Reproduce the original CPU-only checkpoint alias and the cloned fix.
    module=torch.nn.Linear(2,1)
    alias={k:v.cpu() for k,v in module.state_dict().items()}
    frozen={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
    before=frozen["weight"].clone()
    with torch.no_grad():module.weight.add_(1)
    assert torch.equal(alias["weight"],module.weight) and torch.equal(frozen["weight"],before)
    result=dict(split_counts=counts,tensor_file_sha256=file_hashes,chemistry_checks=chemical,gpu_checks=rows,
        cpu_checkpoint_alias_reproduced=True,cloned_checkpoint_preserved=True,label_broadcasting_rejected=True,
        status="All declared checks passed. No claim that the original remote pipeline was reproduced.")
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"split_counts":counts,"gpu_checks":rows,"status":result["status"]},indent=2))


if __name__=="__main__":main()
