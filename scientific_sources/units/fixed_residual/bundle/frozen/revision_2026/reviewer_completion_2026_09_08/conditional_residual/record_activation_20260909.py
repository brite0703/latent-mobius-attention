"""Record the scientific decision; activation and fitting are separate actions."""
import datetime as dt
import json
from pathlib import Path

import campaign_control as control
import campaign_lifecycle as life
import selection

HERE = Path(__file__).resolve().parent
SEQ = HERE.parent / "receptor_context"
MATCHED = SEQ / "pocket_reconstruction"


def main():
    destination = HERE / "activation_decision.json"
    if destination.exists() or (HERE / "retained_study/study_context.json").exists():
        raise FileExistsError("Preserve the original recorded activation decision/context")
    preflight_path = HERE / "native_parent_bundle_v1/activation_preflight.json"
    preflight = life.read(preflight_path)
    interpretation = life.read(MATCHED / "matched_interpretation.json")
    if not interpretation["passed"] or interpretation["contrasts_reconciled"] != 270 or interpretation["profile_workloads_reconciled"] != 60:
        raise ValueError("The completed matched interpretation is required")
    for item in interpretation["sources"] + interpretation["outputs"]:
        life.verify_artifact(item)
    sequence_interpretation = life.read(SEQ / "sequence_interpretation.json")
    if not sequence_interpretation["passed"]:
        raise ValueError("The completed sequence interpretation is required")
    for item in sequence_interpretation["sources"]:
        life.verify_artifact(item)
    evidence = [preflight_path, MATCHED / "matched_interpretation.json", MATCHED / "matched_findings.md",
        MATCHED / "web_review_53_disposition.md", SEQ / "sequence_interpretation.json",
        SEQ / "sequence_findings.md", SEQ / "web_review_51_disposition.md",
        HERE / "protocol_draft.md", HERE / "protocol_clarifications.md",
        HERE / "scoring_cost_specification.md", Path(__file__)]
    decision = dict(
        decision="activate_conditional_residual",
        recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        outcome_informed=True, tests_reused=True,
        reason=("Both fixed receptor studies and their prediction/cost checks are complete. "
                "The matched contact and sequence settings have favorable mean native order-two versus order-one comparisons, "
                "with variable paired effects and comparator-specific limitations. Universal or all-seed superiority is not required. "
                "These end-to-end fits still do not answer whether an added product learner helps at fixed native buckets beyond "
                "the specified additive and comparable-budget pair-MLP residuals. The previously specified residual grid answers "
                "that distinct question while preventing parent adaptation. Activate its unchanged ten cubic and five ligand/contact "
                "order-one parents, three arms, two rates and fitting allowance. The receptor setting is retained from the earlier "
                "specification rather than changed to the currently more favorable sequence setting. All fifteen actual parents "
                "and fitting/validation caches have passed the native/source/archived-validation prerequisites. "
                "Activation occurs after inspection of parent outcomes and remains exploratory on reused evaluation data. "
                "Report every planned outcome and close this fixed grid regardless of direction."),
        candidate_plan=selection.candidate_plan(),
        plan_counts=dict(candidates=90, validation_choices=45, test_outcomes=60, primary_contrasts=6,
                         complete_predictor_cost_outcomes=180, cached_residual_cost_outcomes=255),
        fixed_settings=dict(epoch_limit=100, batch_size=256, patience=8, cpu_threads=1),
        scientific_scope=("A comparison of the specified residual learners at frozen native representations. "
                          "No proof of multiplication necessity, arbitrary-decoder dominance, molecular information retention, "
                          "biological mechanism, causal explanation of the earlier native results, dataset-property rule, "
                          "native theorem transfer, untouched-test confirmation or theorem originality follows from this design."),
        reporting_requirements=[
            "Keep all candidate trajectories, failure denominators, selected epoch-zero outcomes and original parent choices.",
            "Use the same replayed CPU f0 for all contrasts, raw cubic targets and the inherited molecular target transformation exactly once.",
            "Keep product-minus-baseline, product-minus-additive and product-minus-pair-MLP contrasts separate for both tasks.",
            "A benefit shared by all branches supports successful residual fitting without identifying capacity as its cause.",
            "A null result is bounded to this representation, branch family and fitting allowance; no favorable-setting search follows.",
            "All forty-five common choices must precede new residual test decoding; existing tests remain development-inspected."],
        evidence=[life.artifact(path) for path in evidence],
        authorization="Author's continuing authorization for substantive local revision experiments; this is a scientific workflow decision, not a new permission request."
    )
    control.validate_activation(decision, preflight, HERE / "native_parent_bundle_v1/parent_bundle.json")
    life.atomic_json(destination, decision, immutable=True)
    print(json.dumps(dict(recorded=True, recorded_utc=decision["recorded_utc"], candidates=len(decision["candidate_plan"]),
                          residual_fitting_started=False)), flush=True)


if __name__ == "__main__":
    main()
