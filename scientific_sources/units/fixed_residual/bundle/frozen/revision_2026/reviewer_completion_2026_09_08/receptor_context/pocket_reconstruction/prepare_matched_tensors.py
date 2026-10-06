"""Build common canonical ligand graphs and fixed contact inputs; no fitting."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem, rdBase

import radial_contacts as contact
from audit_radial_contacts import scalar_reference

HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
sys.path.insert(0, str(REVISION))
import prepare_lp_pdbbind as ligand_features

COHORT = HERE / "eligible_cohort_v2"
OUT = HERE / "matched_inputs/tensors_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(2)
    require(not OUT.exists(), "Preserve existing cache, including any incomplete attempt")
    inventory = read(HERE / "matched_inputs/audit.json")
    development = read(HERE / "radial_contacts_development_audit.json")
    cohort = read(COHORT / "cohort_manifest.json")
    processing = read(COHORT / "manifest.json")
    require(inventory["passed"] and development["passed"], "Input or descriptor prerequisite failed")
    require(inventory["cohort_manifest_sha256"] == sha(COHORT / "cohort_manifest.json"), "Cohort changed")
    require(inventory["source_sha256"] == sha(HERE / "audit_matched_cohort_inputs.py"), "Input audit source changed")
    require(development["source_sha256"] == sha(Path(contact.__file__)), "Descriptor audit stale")
    require(development["audit_source_sha256"] == sha(HERE / "audit_radial_contacts.py"), "Descriptor audit source changed")
    manifest_path = REVISION / "data/lp_pdbbind/tensors_reconstructed/sample_manifest.csv"
    require(sha(manifest_path) == processing["original_data_manifest_sha256"], "Original target manifest changed")
    sequence_path = HERE.parent / "published_sequences.csv"
    require(sha(sequence_path) == inventory["sequence_csv_sha256"], "Stored sequence source changed")
    source_paths = [Path(__file__), Path(contact.__file__), Path(ligand_features.__file__),
        HERE / "audit_radial_contacts.py", HERE / "radial_contacts_development_audit.json",
        HERE / "matched_inputs/audit.json", HERE / "audit_matched_cohort_inputs.py",
        COHORT / "cohort_manifest.json", COHORT / "manifest.json", COHORT / "final_audit.json",
        manifest_path, sequence_path]
    sources = [dict(path=str(p), sha256=sha(p)) for p in source_paths]
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        metadata_rows = list(csv.DictReader(stream))
    metadata = {r["pdbid"]:r for r in metadata_rows}
    frozen_metadata = {r["pdbid"]:r for r in processing["rows"]}
    require(len(metadata_rows) == len(metadata) == len(frozen_metadata) == 10513, "Original metadata inventory")
    with sequence_path.open(encoding="utf-8", newline="") as stream:
        sequence = {r["pdbid"]:r["sequence"] for r in csv.DictReader(stream)}
    rows = []
    for split in ("train", "val", "test"):
        require([r["pdbid"] for r in cohort["eligible_by_split"][split]] == inventory["ordered_eligible_ids_by_split"][split], "Eligible ordering changed")
        for receipt in cohort["eligible_by_split"][split]:
            key = receipt["pdbid"]
            meta = metadata[key]
            require(meta["split"] == frozen_metadata[key]["split"] == split and
                    meta["canonical_smiles"] == frozen_metadata[key]["canonical_smiles"], "Common ligand or split mismatch")
            payload_path = (COHORT / receipt["payload"]["path"]).resolve()
            require(payload_path.is_relative_to(COHORT.resolve()), "Payload path escaped root")
            encoded = payload_path.read_bytes()
            require(hashlib.sha256(encoded).hexdigest() == receipt["payload"]["compressed_sha256"], "Payload bytes changed")
            raw = gzip.decompress(encoded)
            require(hashlib.sha256(raw).hexdigest() == receipt["payload"]["uncompressed_sha256"], "Payload content changed")
            payload = json.loads(raw)
            require(payload["pdbid"] == key and payload["no_affinity_fields"], "Payload record identity")
            mol = Chem.RemoveHs(Chem.MolFromSmiles(meta["canonical_smiles"]))
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            lig = sorted(payload["ligand_atoms"], key=lambda a:a["reference_atom_index"])
            require([a["reference_atom_index"] for a in lig] == list(range(mol.GetNumAtoms())), "Canonical atom bijection")
            require([a.GetSymbol().upper() for a in mol.GetAtoms()] == [a["element"].upper() for a in lig], "Coordinate/feature element alignment")
            x = np.asarray([ligand_features.atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
            adjacency = Chem.GetAdjacencyMatrix(mol).astype(np.float32)+np.eye(len(x), dtype=np.float32)
            degree_inv = 1/np.sqrt(adjacency.sum(1))
            adj = adjacency*degree_inv[:,None]*degree_inv[None,:]
            features = contact.from_payload(payload)
            require(x.shape == (len(lig),53) and features.shape == (len(lig),216), "Aligned feature dimensions")
            require(np.isfinite(x).all() and np.isfinite(adj).all() and np.isfinite(features).all(), "Nonfinite common input")
            require(np.array_equal(adjacency,adjacency.T) and np.allclose(adj,adj.T,rtol=0,atol=1e-7), "Asymmetric graph")
            require(np.isfinite(float(meta["y"])), "Nonfinite unchanged target")
            rows.append(dict(pdbid=key,split=split,x=x,adj=adj,contact=features,y=np.float32(float(meta["y"])),
                canonical_smiles=meta["canonical_smiles"],sequence=sequence[key],payload=receipt["payload"],
                ligand_xyz=np.array([a["xyz"] for a in lig]), protein_xyz=np.array([a["xyz"] for a in payload["protein_atoms"]]),
                protein_types=contact.protein_types(payload["protein_atoms"])))
            if len(rows)%200 == 0:
                print(json.dumps(dict(prepared=len(rows),total=1612)),flush=True)
    require(len(rows) == len({r["pdbid"] for r in rows}) == 1612, "Prepared inventory mismatch")
    # Deterministic fitting-only ordinary/largest-ligand/largest-pocket references.
    fitting = [r for r in rows if r["split"] == "train"]
    references = [min(fitting,key=lambda r:r["pdbid"]),
                  max(fitting,key=lambda r:(len(r["x"]),r["pdbid"])),
                  max(fitting,key=lambda r:(len(r["protein_xyz"]),r["pdbid"]))]
    reference_checks = []
    for row in references:
        expected = scalar_reference(row["ligand_xyz"],row["protein_xyz"],row["protein_types"]).astype(np.float32)
        require(np.array_equal(expected,row["contact"]), "Independent real-input contact reference mismatch: "+row["pdbid"])
        reference_checks.append(dict(pdbid=row["pdbid"],float32_reference_exact=True))
    max_atoms = max(len(r["x"]) for r in rows)
    require(max_atoms == 57, "Input envelope changed")
    OUT.mkdir(parents=True)
    blobs = {}; entries = []; files = []
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        n = len(selected)
        blob = dict(X=torch.zeros(n,max_atoms,53), mask=torch.zeros(n,max_atoms,dtype=torch.bool),
            adj=torch.zeros(n,max_atoms,max_atoms), contact=torch.zeros(n,max_atoms,216),
            y=torch.from_numpy(np.array([r["y"] for r in selected],dtype=np.float32)),
            ids=[r["pdbid"] for r in selected], sequences=[r["sequence"] for r in selected])
        for i,row in enumerate(selected):
            size=len(row["x"])
            blob["X"][i,:size]=torch.from_numpy(row["x"])
            blob["adj"][i,:size,:size]=torch.from_numpy(row["adj"])
            blob["contact"][i,:size]=torch.from_numpy(row["contact"])
            blob["mask"][i,:size]=True
            entries.append(dict(pdbid=row["pdbid"],split=split,atom_count=size,canonical_smiles=row["canonical_smiles"],
                payload_uncompressed_sha256=row["payload"]["uncompressed_sha256"],
                graph_X_sha256=hashlib.sha256(row["x"].tobytes()).hexdigest(),
                graph_adjacency_sha256=hashlib.sha256(row["adj"].tobytes()).hexdigest(),
                contact_sha256=hashlib.sha256(row["contact"].tobytes()).hexdigest()))
        for key in ("X","adj","contact","y"):
            require(bool(torch.isfinite(blob[key]).all()), "Invalid cached tensor")
        require(not bool(blob["X"][~blob["mask"]].any()) and not bool(blob["contact"][~blob["mask"]].any()), "Nonzero atom padding")
        require(not bool((blob["adj"]*(~blob["mask"])[:,:,None]).any()) and
                not bool((blob["adj"]*(~blob["mask"])[:,None,:]).any()), "Nonzero graph padding")
        path=OUT/(split+".pt")
        torch.save(blob,path)
        restored=torch.load(path,map_location="cpu",weights_only=True)
        require(restored.keys()==blob.keys(), "Cache fields changed after reload")
        for key,value in blob.items():
            require(torch.equal(value,restored[key]) if torch.is_tensor(value) else value==restored[key], "Cache reload mismatch: "+key)
        files.append(dict(split=split,path=str(path),sha256=sha(path),bytes=path.stat().st_size,records=n))
        blobs[split]=blob
    training_targets=blobs["train"]["y"].double()
    mean=float(training_targets.mean()); sd=float(training_targets.std(unbiased=False))
    require(np.isfinite(mean) and np.isfinite(sd) and sd>0, "Invalid fitting target scale")
    for item in sources:
        require(sha(item["path"])==item["sha256"], "Source changed during preparation")
    result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,files=files,
        input_records=1612,split_counts={s:len(b["ids"]) for s,b in blobs.items()},max_atoms=max_atoms,
        target_scale=dict(mean=mean,population_sd=sd,computed_from="fitting-only float32 stored targets accumulated in float64"),
        all_tensor_reload_checks_exact=True,reference_checks=reference_checks,rows=entries,
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,rdkit=rdBase.rdkitVersion),
        fitted_model=False,model_outcomes_used=False,test_targets_used_for_selection=False,
        qualifications=["Canonical-reference atom order, not the old upstream-SMILES tensor order.",
            "Every arm receives the same newly reconstructed ligand graph and unchanged float32 target.",
            "Only deterministic input features are cached; learned embeddings must be recomputed with gradients.",
            "This verifies preparation and serialization, not full-model correctness, GPU feasibility or predictive performance."])
    (OUT/"manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("passed","created_utc","files","split_counts","max_atoms","target_scale","reference_checks")}),flush=True)


if __name__ == "__main__":
    main()
