"""Inspect frozen eligible inputs without using affinities or model outcomes."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

HERE = Path(__file__).resolve().parent
COHORT = HERE / "eligible_cohort_v2"
OUT = HERE / "matched_inputs"
STANDARD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
ELEMENTS = {"C", "N", "O", "S"}
ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
ATOMIC_NUMBERS = {Chem.GetPeriodicTable().GetElementSymbol(i).upper(): i for i in range(1, 119)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def distance_squared(x, z):
    require(x.ndim == z.ndim == 2 and x.shape[1] == z.shape[1] == 3, "Coordinate dimensions")
    require(np.isfinite(x).all() and np.isfinite(z).all(), "Nonfinite coordinates")
    return np.sum((x[:, None, :] - z[None, :, :]) ** 2, axis=-1)


def geometry_checks():
    require([ATOMIC_NUMBERS[s.upper()] for s in ("CL", "Br", "SE", "Ca")] == [17, 35, 34, 20], "mmCIF element-symbol case")
    x = np.array([[0., 0., 0.], [1., 2., 3.]])
    z = np.array([[3., 4., 0.], [0., 0., 10.], [0., 0., 10.000000001]])
    d = distance_squared(x, z)
    require(d[0, 0] == 25 and d[0, 1] == 100 and d[0, 2] > 100, "Literal distance boundary")
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    t = np.array([13., -5., 8.])
    require(np.allclose(d, distance_squared(x@rotation+t, z@rotation+t), rtol=0, atol=1e-12), "Rigid-motion invariance")
    require(np.array_equal(d[::-1, ::-1], distance_squared(x[::-1], z[::-1])), "Atom-permutation covariance")
    try:
        distance_squared(np.array([[np.nan, 0., 0.]]), z)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite coordinates were accepted")
    return ["mmCIF element-symbol case", "independent 3-4-5 distance", "literal radius boundary", "rigid-motion invariance",
            "atom-permutation covariance", "nonfinite rejection"]


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    require(len(a) > 0 and np.isfinite(a).all(), "Empty or nonfinite inventory")
    return dict(minimum=float(a.min()), median=float(np.median(a)),
                percentile95=float(np.quantile(a, .95)), maximum=float(a.max()))


def main():
    require(not (OUT / "audit.json").exists(), "Preserve the completed input audit; use an explicit amendment")
    development = geometry_checks()
    final = read(COHORT / "final_audit.json")
    manifest = read(COHORT / "cohort_manifest.json")
    source = read(COHORT / "manifest.json")
    require(final["passed"] and final["processing_errors"] == 0 and final["completed"] == 10513, "Final cohort not complete")
    require(final["cohort_manifest_sha256"] == sha(COHORT / "cohort_manifest.json"), "Cohort hash changed")
    require(final["processing_manifest_sha256"] == manifest["processing_manifest_sha256"] == sha(COHORT / "manifest.json"), "Processing hash changed")
    for name, expected in source["source_closure"].items():
        require(sha(HERE / name) == expected, "Frozen processor changed: " + name)
    metadata = {r["pdbid"]: r for r in source["rows"]}
    require(len(metadata) == 10513 and all(set(r) == {"pdbid", "split", "canonical_smiles"}
            for r in metadata.values()), "Unexpected input fields")
    require(len(manifest["all_outcomes"]) == 10513 and
            {r["pdbid"] for r in manifest["all_outcomes"]} == set(metadata), "Original committed ID inventory")
    seq_path = HERE.parent / "published_sequences.csv"
    with seq_path.open(encoding="utf-8", newline="") as stream:
        sequences = {r["pdbid"]: {k: r[k] for k in ("pdbid", "split", "sequence")} for r in csv.DictReader(stream)}
    require(set(sequences) == set(metadata), "Sequence inventory mismatch")
    eligible = {r["pdbid"]: (split, r["payload"]) for split, rows in manifest["eligible_by_split"].items() for r in rows}
    require(len(eligible) == final["eligible"] == 1612, "Eligible inventory mismatch")
    # Reconcile all original committed categories without rerunning scientific eligibility.
    status_counts = {s: Counter() for s in ("train", "val", "test")}
    for count, entry in enumerate(manifest["all_outcomes"], 1):
        key = entry["pdbid"]
        r = read(COHORT / "records" / key[1:3] / (key + ".json"))
        body = {k: v for k, v in r.items() if k != "content_sha256"}
        require(hashlib.sha256(canonical_bytes(body)).hexdigest() == r["content_sha256"] == entry["record_content_sha256"], "Record seal mismatch: " + key)
        require(r["pdbid"] == key and r["split"] == entry["split"] == metadata[key]["split"], "Split mismatch: " + key)
        require(r["status"] == entry["status"] and bool(r["eligible"]) == (key in eligible), "Category mismatch: " + key)
        require(r["manifest_sha256"] == manifest["processing_manifest_sha256"], "Foreign cohort record")
        require(r["input_metadata_sha256"] == hashlib.sha256(canonical_bytes(metadata[key])).hexdigest(), "Metadata changed")
        if key in eligible:
            require(r["payload"] == eligible[key][1], "Payload receipt mismatch")
        status_counts[r["split"]][r["status"]] += 1
        if count % 1000 == 0:
            print(json.dumps(dict(stage="record_seals", checked=count, total=10513)), flush=True)
    for split, counts in status_counts.items():
        require(dict(counts) == {k:v for k,v in final["by_split"][split].items() if k != "completed"}, "Final status counts changed")
    ordered_ids = [entry["pdbid"] for entry in manifest["all_outcomes"] if entry["pdbid"] in eligible]
    require(len(ordered_ids) == len(set(ordered_ids)) == 1612, "Ordered ID inventory")
    expected_paths = {(COHORT / receipt["path"]).resolve() for _, receipt in eligible.values()}
    require(expected_paths == {p.resolve() for p in (COHORT / "payloads").rglob("*.json.gz")}, "Orphan or missing payload")
    rows = []
    total_compressed = total_raw = 0
    ligand_elements, pocket_elements = Counter(), Counter()
    for i, key in enumerate(ordered_ids):
        split, receipt = eligible[key]
        path = (COHORT / receipt["path"]).resolve()
        require(path.is_relative_to(COHORT.resolve()), "Payload path escaped cohort")
        encoded = path.read_bytes()
        require(hashlib.sha256(encoded).hexdigest() == receipt["compressed_sha256"], "Compressed payload changed")
        raw = gzip.decompress(encoded)
        require(hashlib.sha256(raw).hexdigest() == receipt["uncompressed_sha256"], "Raw payload changed")
        payload = json.loads(raw)
        require(payload["pdbid"] == key and payload["no_affinity_fields"], "Payload identity")
        lig, pro = payload["ligand_atoms"], payload["protein_atoms"]
        require(len(lig) == receipt["ligand_atoms"] > 0 and len(pro) == receipt["protein_atoms"] > 0, "Atom inventory")
        mol = Chem.RemoveHs(Chem.MolFromSmiles(metadata[key]["canonical_smiles"]))
        require(mol.GetNumAtoms() == len(lig), "Reference atom count: " + key)
        require(sorted(a["reference_atom_index"] for a in lig) == list(range(len(lig))), "Reference atom bijection")
        require(all(mol.GetAtomWithIdx(a["reference_atom_index"]).GetAtomicNum() == ATOMIC_NUMBERS[a["element"].upper()] for a in lig), "Reference element alignment")
        require(all(a["residue"] in STANDARD and a["element"] in ELEMENTS for a in pro), "Unsupported retained pocket atom")
        for atoms in (lig, pro):
            require(len({(a["operator"], a["source_atom_id"]) for a in atoms}) == len(atoms), "Duplicate atom identity")
            require(all(np.isfinite(a["occupancy"]) and a["occupancy"] > 0 for a in atoms), "Nonpositive occupancy")
        x, z = np.array([a["xyz"] for a in lig]), np.array([a["xyz"] for a in pro])
        distances = distance_squared(x, z)
        residue_groups = {}
        for j, a in enumerate(pro):
            rk = (a["operator"], a["label_chain"], a["label_position"], a["residue"])
            require(bool(a["label_position"]), "Missing residue position")
            residue_groups.setdefault(rk, []).append(j)
        require(all(float(distances[:, indices].min()) <= 100.0 for indices in residue_groups.values()), "Retained residue outside literal 10-Angstrom crop: " + key)
        sequence = sequences[key]
        require(sequence["split"] == split, "Sequence split mismatch")
        chains = sequence["sequence"].split(":")
        require(all(c and set(c) <= ALPHABET for c in chains), "Invalid stored chain")
        ligand_elements.update(a["element"] for a in lig)
        pocket_elements.update(a["element"] for a in pro)
        rows.append(dict(pdbid=key, split=split, ligand_atoms=len(lig), pocket_atoms=len(pro),
            pocket_residues=len(residue_groups), pocket_chain_instances=len({rk[:2] for rk in residue_groups}),
            ligand_pocket_pairs=len(lig)*len(pro), pairs_within6=int((distances <= 36).sum()),
            pairs_within8=int((distances <= 64).sum()), pairs_within10=int((distances <= 100).sum()),
            minimum_interatomic_distance=float(np.sqrt(distances.min())),
            supplied_chains=len(chains), supplied_residues=sum(map(len, chains)),
            longest_chain=max(map(len, chains)), single_record_padded_tokens=len(chains)*max(map(len, chains)),
            payload_compressed_bytes=len(encoded), payload_uncompressed_bytes=len(raw)))
        total_compressed += len(encoded); total_raw += len(raw)
        if (i+1) % 200 == 0:
            print(json.dumps(dict(checked=i+1, total=len(ordered_ids))), flush=True)
    fitting_reference = [key for key, (split, _) in eligible.items() if split in ("train", "val")]
    ref_ligands = {metadata[key]["canonical_smiles"] for key in fitting_reference}
    ref_strings = {sequences[key]["sequence"] for key in fitting_reference}
    ref_chains = {c for s in ref_strings for c in s.split(":")}
    test_ids = [key for key in ordered_ids if eligible[key][0] == "test"]
    full = set(test_ids)
    absent = {"canonical_ligand": {k for k in full if metadata[k]["canonical_smiles"] not in ref_ligands},
              "complete_stored_string": {k for k in full if sequences[k]["sequence"] not in ref_strings},
              "any_supplied_chain": {k for k in full if not set(sequences[k]["sequence"].split(":")) & ref_chains}}
    absent["both_ligand_and_any_chain"] = absent["canonical_ligand"] & absent["any_supplied_chain"]
    groups = {"full": test_ids}
    for name, keys in absent.items():
        groups[name+"_absent_from_train_and_val"] = [key for key in test_ids if key in keys]
        complement = "either_ligand_or_any_supplied_chain" if name == "both_ligand_and_any_chain" else name
        groups[complement+"_overlap_with_train_or_val"] = [key for key in test_ids if key not in keys]
    fields = [k for k in rows[0] if k not in ("pdbid", "split")]
    summaries = {split: {key: distribution([row[key] for row in rows if row["split"] == split]) for key in fields}
                 for split in ("train", "val", "test")}
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "input_inventory.csv"
    require(not csv_path.exists(), "Preserve existing inventory")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(passed=True, audited_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
        cohort_manifest_sha256=sha(COHORT / "cohort_manifest.json"), processing_manifest_sha256=sha(COHORT / "manifest.json"),
        final_cohort_audit_sha256=sha(COHORT / "final_audit.json"), sequence_csv_sha256=sha(seq_path),
        original_records_reconciled=10513, eligible_payloads_reconciled=1612,
        numerical_targets_analyzed=False, model_predictions_analyzed=False, scientific_eligibility_changed=False,
        ordered_eligible_ids_by_split={split:[k for k in ordered_ids if eligible[k][0] == split] for split in summaries},
        status_counts={s:dict(c) for s,c in status_counts.items()}, input_distributions=summaries,
        ligand_element_counts=dict(ligand_elements), pocket_element_counts=dict(pocket_elements),
        payload_bytes=dict(compressed=total_compressed, uncompressed=total_raw),
        test_groups=groups, test_subset_sizes={key:len(value) for key,value in groups.items()},
        geometry_checks=development, inventory_csv_sha256=sha(csv_path),
        qualifications=["Input inventory only: no model has been trained or selected here.",
            "All test shapes enter resource accounting; test affinities and predictions do not.",
            "Radius counts at 6/8/10 Angstrom are resource diagnostics, not selected model settings.",
            "The crop is conditioned on a located ligand and selected source/chemical rules.",
            "Exact-overlap subsets use this common fitting/validation cohort and do not establish similarity separation.",
            "Existing chemistry and structure audits are prerequisites, not replaced by this inventory."])
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Matched public-cohort input audit", "", result["audited_utc"], "",
        "All 10,513 committed outcomes and all 1,612 eligible coordinate payloads reconcile with their frozen hashes. The audit checks the ligand-reference atom bijection and elements, finite coordinates, distinct atom identities, standard pocket chemistry, the literal residue crop and unchanged sequence strings. No affinity or prediction determines these checks; no scientific exclusion is changed.", "",
        "| Split | Records | Ligand atoms, median / max | Pocket atoms, median / max | Pocket residues, median / max | Largest ligand-pocket pair matrix |", "|---|---:|---:|---:|---:|---:|"]
    for split, stats in summaries.items():
        n = len(result["ordered_eligible_ids_by_split"][split])
        cells = [f"{stats[k]['median']:g} / {stats[k]['maximum']:g}" for k in ("ligand_atoms", "pocket_atoms", "pocket_residues")]
        lines.append(f"| {split} | {n} | "+" | ".join(cells)+f" | {stats['ligand_pocket_pairs']['maximum']:g} |")
    lines += ["", "The complete per-entry inventory and all percentile summaries remain in input_inventory.csv and audit.json. Input sizes can inform memory limits before fitting; they do not establish GPU feasibility. The retained cohort has changed substantially from the full dataset, so every direct ligand/sequence/pocket comparison must refit on its common fitting and validation records and retain its common test records.", "",
        "| Fixed test subset | Records |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k,v in result["test_subset_sizes"].items()]
    lines += ["", "These are exact canonical-ligand and supplied-string/chain comparisons against the eligible fitting and validation rows. They describe reused test subpopulations; they neither isolate leakage effects nor establish sequence-similarity separation. Atom-distance counts are descriptive resource measurements, and the model interface still requires its own audit."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("passed", "audited_utc", "original_records_reconciled", "eligible_payloads_reconciled", "input_distributions", "payload_bytes", "test_subset_sizes")}), flush=True)


if __name__ == "__main__":
    main()
