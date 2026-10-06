"""Proposed finite allocation and validation-only choices; no experiment I/O."""
import math
import random
from models import ARMS

RATES = (.0003, .001)
DOMAINS = {"cubic": tuple(range(100, 110)), "ligand_contact": tuple(range(42, 47))}


def candidate_plan():
    rows = []
    for domain, seeds in DOMAINS.items():
        for seed in seeds:
            for arm in ARMS:
                for rate_index, rate in enumerate(RATES):
                    rows.append(dict(id=f"{domain}_{arm}_seed{seed}_lr{rate_index}",
                                     domain=domain, seed=seed, arm=arm, rate_index=rate_index, rate=rate))
    random.Random(2026090843).shuffle(rows)
    return rows


def validate_spec(spec):
    if spec not in candidate_plan():
        raise ValueError("Candidate must equal an entry in the declared finite allocation")


def select_all(records):
    """No test metric, test prediction, or failure's earlier score is consulted.

    Actual file/checkpoint/parent verification belongs to the campaign layer,
    which does not yet exist. This function cannot certify those artifacts.
    """
    plan = candidate_plan()
    by_id = {r["spec"]["id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {s["id"] for s in plan}:
        raise ValueError("Every declared candidate needs exactly one terminal record")
    for spec in plan:
        row = by_id[spec["id"]]
        if row["spec"] != spec:
            raise ValueError("Candidate specification changed")
        if row["status"] not in ("valid", "failed_numerical", "missing_parent"):
            raise ValueError("Only terminal declared outcomes may reach selection")
        if row["status"] != "missing_parent":
            binding = row.get("parent_checkpoint_sha256")
            if not isinstance(binding, str) or len(binding) != 64 or any(c not in '0123456789abcdef' for c in binding):
                raise ValueError("Missing parent checkpoint binding")
        if row["status"] == "valid":
            score, initial, epoch = row["best_validation_mse"], row["epoch_zero_validation_mse"], row["best_epoch"]
            if not (type(epoch) is int and type(row["completed_epochs"]) is int
                    and 0 <= epoch <= row["completed_epochs"] <= 100 and row["completed_epochs"] >= 1):
                raise ValueError("Invalid checkpoint or completed epoch")
            if not (math.isfinite(score) and math.isfinite(initial) and 0 <= score <= initial):
                raise ValueError("A valid choice must include its finite epoch-zero option")
    choices = []
    for domain, seeds in DOMAINS.items():
        for seed in seeds:
            parent_rows = [r for r in records if r["spec"]["domain"] == domain and r["spec"]["seed"] == seed]
            absent = [r["status"] == "missing_parent" for r in parent_rows]
            if any(absent) and not all(absent):
                raise ValueError("A missing parent applies to all arms and rates for its replicate")
            bindings = {r["parent_checkpoint_sha256"] for r in parent_rows if not all(absent)}
            if len(bindings) > 1:
                raise ValueError("Arms must inherit the same parent predictor")
            for arm in ARMS:
                pair = sorted((r for r in parent_rows if r["spec"]["arm"] == arm), key=lambda r: r["spec"]["rate_index"])
                eligible = [r for r in pair if r["status"] == "valid"]
                chosen = min(eligible, key=lambda r: (r["best_validation_mse"], r["spec"]["rate_index"])) if eligible else None
                choices.append(dict(domain=domain, seed=seed, arm=arm,
                    candidate_ids=[r["spec"]["id"] for r in pair],
                    candidate_statuses=[r["status"] for r in pair],
                    selected_id=chosen["spec"]["id"] if chosen else None,
                    selected_epoch=chosen["best_epoch"] if chosen else None,
                    outcome=("selected_epoch_zero" if chosen["best_epoch"] == 0 else "selected_trained") if chosen
                            else ("missing_parent" if all(absent) else "no_valid_residual")))
    assert len(choices) == 45
    return choices
