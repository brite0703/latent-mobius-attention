"""Fixed candidate enumeration and validation-only sequence-study selection."""
import math
import random
from sequence_models import configurations, HEADS

SEEDS = tuple(range(42, 47))
RATES = (.0003, .001)
CANDIDATE_ORDER_SEED = 2026090834


def candidate_plan():
    rows = [dict(id=f"{setting}_{head}_seed{seed}_lr{index}", setting=setting,
                 head=head, seed=seed, lr=lr, lr_index=index)
            for setting, head in configurations() for seed in SEEDS
            for index, lr in enumerate(RATES)]
    random.Random(CANDIDATE_ORDER_SEED).shuffle(rows)
    return rows


def key(setting, head):
    return setting+"/"+head


def planned_contrasts():
    rows = []
    for setting in ("ligand_only", "ligand_sequence"):
        for right in ("lma1", "deepsets_plain70", "cp_pool", "additive2"):
            rows.append(dict(id=f"{setting}_lma2_minus_{right}",
                terms=[dict(configuration=key(setting, "lma2"), weight=1),
                       dict(configuration=key(setting, right), weight=-1)]))
    for head in HEADS:
        rows.append(dict(id=f"{head}_joint_minus_ligand",
            terms=[dict(configuration=key("ligand_sequence", head), weight=1),
                   dict(configuration=key("ligand_only", head), weight=-1)]))
    for head in HEADS:
        rows.append(dict(id=f"{head}_joint_minus_sequence",
            terms=[dict(configuration=key("ligand_sequence", head), weight=1),
                   dict(configuration=key("sequence_only", "none"), weight=-1)]))
    rows.append(dict(id="joint_order_difference_minus_ligand_order_difference",
        terms=[dict(configuration=key("ligand_sequence", "lma2"), weight=1),
               dict(configuration=key("ligand_sequence", "lma1"), weight=-1),
               dict(configuration=key("ligand_only", "lma2"), weight=-1),
               dict(configuration=key("ligand_only", "lma1"), weight=1)]))
    return rows


def select(records):
    """Require every prescribed terminal record; never inspect test outcomes."""
    plan = candidate_plan()
    expected = {row["id"]: row for row in plan}
    actual = {row["id"]: row for row in records}
    if len(records) != len(actual) or set(actual) != set(expected):
        raise ValueError("Selection requires exactly the 110 unique prescribed candidates")
    for identifier, row in actual.items():
        if any(row.get(field) != value for field, value in expected[identifier].items()):
            raise ValueError("Candidate identity differs from the fixed plan")
        if row.get("status") not in ("valid", "failed"):
            raise ValueError("Every prescribed candidate needs a terminal outcome")
        if row["status"] == "valid":
            score, epoch = row.get("best_validation_rmse"), row.get("best_epoch")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
                raise ValueError("A valid candidate needs a finite nonnegative validation RMSE")
            if type(epoch) is not int or not 1 <= epoch <= 100:
                raise ValueError("A valid candidate needs a selected epoch from the declared horizon")
    selections, nomination_rows = [], []
    for setting, head in configurations():
        procedure_rows = []
        for seed in SEEDS:
            candidates = sorted([row for row in actual.values()
                if (row["setting"], row["head"], row["seed"]) == (setting, head, seed)], key=lambda row: row["lr"])
            assert len(candidates) == 2
            valid = [row for row in candidates if row["status"] == "valid"]
            base = dict(configuration=key(setting, head), setting=setting, head=head, seed=seed,
                        candidate_ids=[row["id"] for row in candidates])
            if valid:
                chosen = min(valid, key=lambda row: (row["best_validation_rmse"], row["lr"]))
                base.update(status="valid", selected_id=chosen["id"], learning_rate=chosen["lr"],
                    best_epoch=chosen["best_epoch"], validation_rmse=chosen["best_validation_rmse"])
            else:
                base.update(status="all_candidates_failed", selected_id=None, learning_rate=None,
                            best_epoch=None, validation_rmse=None)
            selections.append(base)
            procedure_rows.append(base)
        valid_scores = [row["validation_rmse"] for row in procedure_rows if row["status"] == "valid"]
        nomination_rows.append(dict(configuration=key(setting, head), prescribed_seeds=5,
            valid_seeds=len(valid_scores), eligible=len(valid_scores) == 5,
            mean_selected_validation_rmse=math.fsum(valid_scores)/5 if len(valid_scores) == 5 else None))
    eligible = [(i, row) for i, row in enumerate(nomination_rows) if row["eligible"]]
    nominated = min(eligible, key=lambda pair: (pair[1]["mean_selected_validation_rmse"], pair[0]))[1]["configuration"] if eligible else None
    return dict(selections=selections, procedure_validation=nomination_rows,
        nominated_procedure=nominated,
        nomination_scope="Five selected validation scores nominate a complete procedure, not an ensemble, refit, or retrospective change in the primary information setting.",
        primary_information_setting="ligand_sequence",
        primary_contrast="ligand_sequence_lma2_minus_lma1", contrasts=planned_contrasts())
