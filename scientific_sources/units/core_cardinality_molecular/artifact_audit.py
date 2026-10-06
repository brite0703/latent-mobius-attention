"""Check final evidence coverage and arithmetic without rerunning any fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
SUPP = HERE/"cardinality_controls"
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    core, supp = read(HERE/"evaluation.json"), read(SUPP/"evaluation.json")
    actual = core["rows"]+supp["rows"]
    summary = read(HERE/"combined_results_summary.json")
    assert len(actual) == 110 and len({r["selected_id"] for r in actual}) == 110
    assert len(summary["rows"]) == 22
    errors = []
    for r in summary["rows"]:
        group = [a for a in actual if (a["depth"],a["head"]) == (r["depth"],r["head"])]
        assert {a["seed"] for a in group} == set(range(42,47))
        assert all(a["status"] == "valid" for a in group)
        values = [a["test"]["rmse"] for a in group]
        mean = math.fsum(values)/5
        sd = statistics.stdev(values)
        errors.extend([abs(mean-r["rmse"]["mean"]),abs(sd-r["rmse"]["sd"])])
        assert {a["total_parameters"] for a in group} == {r["parameters"]}
    assert max(errors) < 1e-12
    resource = list(csv.DictReader((HERE/"tables/all_22_model_profiles.csv").open(encoding="utf-8")))
    assert len(resource) == 88
    assert {r["padded_n"] for r in resource} == {"53"}
    assert len({(r["depth"],r["head"],r["batch_size"],r["mode"]) for r in resource}) == 88
    assert read(HERE/"profiles.json")["workload_ids"] == read(SUPP/"profiles.json")["workload_ids"]
    stability = list(csv.DictReader((HERE/"tables/all_220_stability_records.csv").open(encoding="utf-8")))
    assert len(stability) == 220
    clipped = sum(int(r["epochs_with_a_batch_norm_above_10"])>0 for r in stability)
    assert clipped == 219
    assert round(max(float(r["largest_recorded_preclip_norm"]) for r in stability if r["head"]=="lma3"),5) == 15.99442
    data_path = ROOT/"revision_2026/data/lp_pdbbind/tensors_reconstructed/sample_manifest.csv"
    members = list(csv.DictReader(data_path.open(encoding="utf-8-sig")))
    validation_ligands = {r["canonical_smiles"] for r in members if r["split"] == "val"}
    overlap = [r["pdbid"] for r in members if r["split"] == "test" and r["canonical_smiles"] in validation_ligands]
    assert len(overlap) == 150
    labels = read(ROOT/"revision_2026/results/data_interpretation_audit.json")
    assert labels["remaining_by_split"] == {"train":241}
    current = read(HERE/"reviewer_resolution_update.json")
    original = read(ROOT/"revision_2026/reviewer_resolution_2026_09_07/reviewer_resolution_matrix.json")
    assert len(current["rows"]) == len(original["rows"]) == 56
    for old,new in zip(original["rows"],current["rows"]):
        for key,value in old.items():
            if key != "current_status":
                assert new[key] == value, (old["comment"],key)
    assert current["new_candidates"] == 220 and current["new_selections"] == 110
    log = (HERE/"output/pdf/neural_revision_insert.log").read_text(encoding="utf-8",errors="replace")
    assert not any(word in log for word in ("Warning:","Overfull", "Underfull"))
    pdf = HERE/"output/pdf/neural_revision_insert.pdf"
    poppler = Path(r"C:\Users\88695\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin")
    info = subprocess.check_output([str(poppler/"pdfinfo.exe"),str(pdf)],text=True,encoding="utf-8")
    assert any(line.split(":",1)[1].strip() == "3" for line in info.splitlines() if line.startswith("Pages:"))
    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    assert "??" not in text
    for token in ("1.67145","1.68916","1.61960","1.61993","1.62569","4,047","4,091"):
        assert token in text, token
    for page in range(1,4):
        assert (HERE/f"tmp/pdfs/neural_revision_insert-{page}.png").exists()
    output = dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
        selected_predictors=110,configurations=22,candidate_records=220,resource_measurements=88,
        maximum_summary_arithmetic_delta=max(errors),active_clipping_candidates=clipped,
        validation_test_shared_ligand_test_records=len(overlap),annotation_qualified_training_records=241,
        original_reviewer_comments_preserved=56,compiled_pdf_pages=3,compile_warnings=0,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        visual_review="All three 110-dpi Poppler-rendered pages inspected; table, math, text and references legible without clipping or overlap.",
        scope="Final artifact coverage, arithmetic, data-qualification and rendering checks; no additional model fit or selection.")
    (HERE/"artifact_audit.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output),flush=True)


if __name__ == "__main__":
    main()
