"""Disposable edge cases for candidate completeness, failures and nomination."""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sequence_selection as selection
import sequence_models as models

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    plan = selection.candidate_plan()
    assert len(plan) == len({row["id"] for row in plan}) == 110
    configs = list(models.configurations())
    assert len(configs) == 11 and configs[-1] == ("sequence_only", "none")
    assert selection.candidate_plan() == plan
    positions = {pair: index for index, pair in enumerate(configs)}
    records = [row | dict(status="valid", best_epoch=5,
        best_validation_rmse=float(positions[(row["setting"], row["head"])]+2)) for row in plan]
    # The lowest apparent partial procedure must be ineligible after both rates fail.
    for row in records:
        if (row["setting"], row["head"]) == configs[0]:
            row["best_validation_rmse"] = 0.
            if row["seed"] == 42:
                row.update(status="failed", best_validation_rmse=None, best_epoch=None)
        # Complete procedures 1 and 2 tie; the fixed configuration order must win.
        elif (row["setting"], row["head"]) in configs[1:3]:
            row["best_validation_rmse"] = 2.
        # A real validation improvement must override the lower-rate tie rule.
        elif (row["setting"], row["head"], row["seed"], row["lr_index"]) == (*configs[3], 44, 1):
            row["best_validation_rmse"] = 1.
    result = selection.select(records)
    assert len(result["selections"]) == 55
    assert result == selection.select(list(reversed(records)))
    assert result["nominated_procedure"] == selection.key(*configs[1])
    assert result["procedure_validation"][0] == dict(configuration=selection.key(*configs[0]),
        prescribed_seeds=5, valid_seeds=4, eligible=False, mean_selected_validation_rmse=None)
    assert sum(row["status"] == "all_candidates_failed" for row in result["selections"]) == 1
    for row in result["selections"]:
        if row["status"] != "valid":
            assert row["selected_id"] is None
        elif (row["setting"], row["head"], row["seed"]) == (*configs[3], 44):
            assert row["learning_rate"] == .001 and row["validation_rmse"] == 1.
        else:
            assert row["learning_rate"] == .0003
    all_failed = [row | dict(status="failed", best_validation_rmse=None, best_epoch=None) for row in records]
    assert selection.select(all_failed)["nominated_procedure"] is None
    bad_cases = [records[:-1], records+[records[0]]]
    for changes in (dict(lr=.003), dict(status="running"), dict(best_validation_rmse=float("nan")),
                    dict(best_validation_rmse=-1.), dict(best_validation_rmse=True), dict(best_epoch=0)):
        invalid = copy.deepcopy(records)
        target = next(row for row in invalid if row["status"] == "valid")
        target.update(changes)
        bad_cases.append(invalid)
    rejected = 0
    for invalid in bad_cases:
        try:
            selection.select(invalid)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("An invalid candidate set was accepted")
    assert rejected == 8
    contrasts = selection.planned_contrasts()
    assert len(contrasts) == len({row["id"] for row in contrasts}) == 19
    valid_keys = {selection.key(*pair) for pair in configs}
    assert all(sum(t["weight"] for t in row["terms"]) == 0 and
               all(t["configuration"] in valid_keys for t in row["terms"]) for row in contrasts)
    assert contrasts[-1]["terms"] == [
        dict(configuration="ligand_sequence/lma2", weight=1),
        dict(configuration="ligand_sequence/lma1", weight=-1),
        dict(configuration="ligand_only/lma2", weight=-1),
        dict(configuration="ligand_only/lma1", weight=1)]
    metadata = json.loads((HERE/"sequence_metadata.json").read_text())
    assert metadata["passed"]
    for source in metadata["sources"]:
        assert sha(source["path"]) == source["sha256"]
    specification = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        retained_fitting_started=False, candidate_count=110, selection_count=55,
        candidate_order_seed=selection.CANDIDATE_ORDER_SEED, candidates=plan,
        configuration_order=[selection.key(*pair) for pair in configs],
        primary_information_setting="ligand_sequence",
        primary_contrast="ligand_sequence_lma2_minus_lma1", contrasts=contrasts,
        selection="Require all 110 terminal candidate records; each seed selects the minimum valid validation RMSE, lower learning rate on exact tie; an all-rates-failed seed remains failed.",
        nomination="Only complete five-seed procedures are eligible; minimum arithmetic mean selected validation RMSE, fixed configuration order on exact tie. No test outcome is used.",
        metadata_sha256=sha(HERE/"sequence_metadata.json"),
        sources=[dict(path=str(HERE/name), sha256=sha(HERE/name)) for name in
            ("sequence_selection.py", "sequence_models.py", "sequence_protocol.md")],
        qualification="Pre-fit candidate/analysis specification only. It is not the GPU-verified retained implementation lock or a record of actual model selection.")
    destination = HERE/"sequence_execution_plan.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        assert {k:v for k,v in old.items() if k != "created_utc"} == {k:v for k,v in specification.items() if k != "created_utc"}
    else:
        destination.write_text(json.dumps(specification, indent=2, allow_nan=False), encoding="utf-8")
    audit = dict(completed_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        candidates=110, choices=55, contrasts=19, invalid_cases_rejected=rejected,
        checks=["complete candidate identity", "both-rate failure retained", "incomplete procedure excluded",
                "lower-rate exact tie", "higher-rate validation improvement", "fixed-order nomination tie",
                "input record-order invariance", "all procedures failed", "signed difference-of-differences"],
        sources=[dict(path=str(HERE/name), sha256=sha(HERE/name)) for name in
            ("sequence_selection.py", "sequence_selection_audit.py", "sequence_execution_plan.json", "sequence_metadata.json")],
        qualification="Manufactured scalar validation records; no affinity target, retained prediction or test tensor was read, and no GPU computation was performed. Candidate orchestration, actual checkpoints and GPU execution need their separate checks.")
    (HERE/"sequence_selection_cpu_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    run()
