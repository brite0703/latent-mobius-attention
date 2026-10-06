"""Audit information visible to ligand-only models and label-check coverage."""
import collections
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parent
DATA=ROOT/"data"/"lp_pdbbind"/"tensors_reconstructed"


def main():
    torch.set_num_threads(2)
    manifest=pd.read_csv(DATA/"sample_manifest.csv",keep_default_na=False)
    exact=re.compile(r"(?:Kd|Ki|IC50)=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(mM|uM|nM|pM|fM|M)")
    remaining=[]
    for r in manifest.to_dict("records"):
        match=exact.fullmatch(r["affinity_record"])
        if match and float(match[1])>0:continue
        raw=r["affinity_record"]
        reason=("censored_inequality" if any(c in raw for c in ("<",">","≤","≥")) else
                "approximate_annotation" if any(c in raw for c in ("~","≈")) else
                "not_matched_by_exact_parser")
        remaining.append({k:r[k] for k in ("pdbid","split","affinity_record","y")}|{"reason":reason})
    grouped={}
    smi=dict(zip(manifest.pdbid,manifest.canonical_smiles))
    for split in ("train","val","test"):
        blob=torch.load(DATA/f"pdbbind_{split}.pt",weights_only=True,map_location="cpu")
        groups=collections.defaultdict(list)
        for i,pdbid in enumerate(blob["ids"]):
            rows=blob["X"][i][blob["mask"][i]].numpy()
            ordered=sorted(row.astype("<f4",copy=False).tobytes() for row in rows)
            key=hashlib.sha256(b"".join(ordered)).hexdigest()
            groups[key].append(pdbid)
        grouped[split]=dict(groups)
    overlaps={}
    for a,b in itertools.combinations(grouped,2):
        common=set(grouped[a])&set(grouped[b])
        distinct=[h for h in common if len({smi[p] for p in grouped[a][h]+grouped[b][h]})>1]
        overlaps[f"{a}_{b}"]=dict(shared_atom_feature_multisets=len(common),
            records_in_first=sum(len(grouped[a][h]) for h in common),
            records_in_second=sum(len(grouped[b][h]) for h in common),
            shared_multisets_with_distinct_canonical_smiles=len(distinct),
            examples=[dict(first_ids=grouped[a][h],second_ids=grouped[b][h],
                canonical_smiles=sorted({smi[p] for p in grouped[a][h]+grouped[b][h]})) for h in sorted(distinct)[:5]])
    result=dict(retained_records=len(manifest),exact_parser_verified_records=len(manifest)-len(remaining),
        remaining_affinity_annotations=len(remaining),remaining_by_reason=dict(collections.Counter(r["reason"] for r in remaining)),
        remaining_by_split=dict(collections.Counter(r["split"] for r in remaining)),
        label_policy="The raw published numeric value remains the target. Unmatched or approximate annotations are not certified exact affinities.",
        feature_collision_definition="SHA256 of lexicographically sorted exact float32 atom-feature rows, retaining multiplicities; chemical adjacency is not included",
        atom_feature_multiset_overlap=overlaps,
        interpretation="Different molecules can yield identical MLP set inputs. This is an input-information audit, not proof that the published joint protein/ligand split is violated.")
    (ROOT/"results"/"data_interpretation_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (ROOT/"results"/"affinity_annotations_not_exactly_verified.json").write_text(json.dumps(remaining,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
