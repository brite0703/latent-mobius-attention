"""All archived selected predictors on manifest-defined ligand-overlap subsets."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
REVISION = COMPLETION.parent
CORE = REVISION/"neural_reviewer_study_2026_09_07"
MANIFEST = REVISION/"data/lp_pdbbind/tensors_reconstructed/sample_manifest.csv"
CAMPAIGNS = dict(core=CORE, cardinality=CORE/"cardinality_controls",
                 gate=COMPLETION/"gate_readout", components=COMPLETION/"components_width")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    assert len(y) and y.shape == p.shape
    error = y-p
    mean_y, mean_p = math.fsum(y)/len(y), math.fsum(p)/len(p)
    a, b = y-mean_y, p-mean_p
    den = math.sqrt(math.fsum(a*a)*math.fsum(b*b))
    return dict(n=len(y), rmse=math.sqrt(math.fsum(error*error)/len(y)),
                mae=math.fsum(np.abs(error))/len(y), pearson=math.fsum(a*b)/den if den else None)


def run():
    # All four completed campaigns are required; no partial/favorable subset report.
    for folder in CAMPAIGNS.values():
        audit = json.loads((folder/"final_audit.json").read_text(encoding="utf-8"))
        assert audit["passed"]
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        entries = list(csv.DictReader(f))
    by_id = {row["pdbid"]: row for row in entries}
    assert len(entries) == len(by_id)
    fitting = {r["canonical_smiles"] for r in entries if r["split"] == "train"}
    validation = {r["canonical_smiles"] for r in entries if r["split"] == "val"}
    test = [r for r in entries if r["split"] == "test"]
    ids = [r["pdbid"] for r in test]
    groups = dict(full=set(ids), absent_from_train_and_val={r["pdbid"] for r in test if r["canonical_smiles"] not in fitting | validation},
                  shared_with_train_or_val={r["pdbid"] for r in test if r["canonical_smiles"] in fitting | validation},
                  shared_with_val={r["pdbid"] for r in test if r["canonical_smiles"] in validation})
    assert groups["absent_from_train_and_val"] | groups["shared_with_train_or_val"] == groups["full"]
    assert not groups["absent_from_train_and_val"] & groups["shared_with_train_or_val"]
    records, sources = [], []
    for campaign, folder in CAMPAIGNS.items():
        evaluation_path = folder/"evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        sources.append(dict(campaign=campaign, evaluation_sha256=sha(evaluation_path), audit_sha256=sha(folder/"final_audit.json")))
        for row in evaluation["rows"]:
            if row["status"] != "valid":
                records.append(dict(campaign=campaign, head=row["head"], depth=row["depth"], seed=row["seed"],
                                    status=row["status"], groups=None))
                continue
            prediction_path = folder/row["prediction_file"]
            assert sha(prediction_path) == row["prediction_sha256"]
            with np.load(prediction_path) as z:
                assert set(z["ids"].tolist()) == groups["full"]
                assert len(z["ids"]) == len(groups["full"])
                # Saved truths are float32 tensor targets, compared in those same units.
                expected_y = np.asarray([float(by_id[str(i)]["y"]) for i in z["ids"]], dtype=np.float32).astype(float)
                np.testing.assert_array_equal(expected_y, z["truth"])
                result = {}
                for name, members in groups.items():
                    mask = np.asarray([str(i) in members for i in z["ids"]])
                    result[name] = metrics(z["truth"][mask], z["prediction"][mask]) if mask.any() else None
                assert abs(result["full"]["rmse"]-row["test"]["rmse"]) < 1e-12
                records.append(dict(campaign=campaign, head=row["head"], depth=row["depth"], seed=row["seed"],
                    selected_id=row["selected_id"], status="valid", groups=result, prediction_sha256=row["prediction_sha256"]))
    assert len(records) == 240
    summaries = []
    configs = sorted({(r["campaign"], r["depth"], r["head"]) for r in records})
    for campaign, depth, head in configs:
        values = [r for r in records if (r["campaign"], r["depth"], r["head"]) == (campaign, depth, head)]
        assert len(values) == 5
        for group in groups:
            scores = [r["groups"][group]["rmse"] for r in values if r["status"] == "valid" and r["groups"][group] is not None]
            summaries.append(dict(campaign=campaign, depth=depth, head=head, group=group, test_records=len(groups[group]),
                valid=len(scores), mean_rmse=float(np.mean(scores)) if scores else None,
                sample_sd_rmse=float(np.std(scores, ddof=1)) if len(scores)>1 else None))
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        manifest_sha256=sha(MANIFEST), analysis_sha256=sha(__file__), protocol_sha256=sha(HERE/"protocol.md"),
        subsets={name:sorted(value) for name, value in groups.items()}, records=records, summaries=summaries, sources=sources,
        qualification="Prespecified metadata-only subset of a reused test; different evaluation population, no reselection, no independent replication or protein-level decontamination.")
    (HERE/"overlap_sensitivity.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (HERE/"overlap_sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(dict(passed=True, predictors=len(records), subset_sizes={name:len(value) for name,value in groups.items()})))


if __name__ == "__main__":
    run()
