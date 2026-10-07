"""Plan or inspect metadata gates for the unchanged fixed-residual reporter."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_entry_support import (ROOT, configuration, check_source, source_record,
                                  external_root, relative_file, read_json, require, emit, run_cli)

def preflight(mode, evidence, config, root=ROOT):
    evidence = external_root(evidence, root)
    layout = config["fixed_report"]["evidence_layout"]
    status = read_json(relative_file(evidence, layout + "/retained_pipeline_status.json"))
    require(isinstance(status, dict) and isinstance(status.get("status"), str)
            and isinstance(status.get("stage"), str), "FIXED_STATUS_SCHEMA_MISMATCH")
    if mode == "complete":
        stages = status.get("stages_finished")
        require(status["status"] == "complete" and isinstance(stages, list)
                and all(isinstance(stage, str) for stage in stages)
                and set(stages) == {"train", "evaluate", "profile"}, "FIXED_COMPLETE_GATE_BLOCKED")
    else:
        require(status["stage"] != "profile" or status["status"] == "complete",
                "FIXED_SELECTION_TIMING_GATE_BLOCKED")
    lock_path = relative_file(evidence, layout + "/retained_study/selection_lock.json")
    require(lock_path.is_file(), "FIXED_SELECTION_LOCK_MISSING")
    lock = read_json(lock_path)
    context = read_json(relative_file(evidence, layout + "/retained_study/study_context.json"))
    require(isinstance(lock, dict) and isinstance(context, dict)
            and context.get("scope") == "retained_residual_campaign", "FIXED_CONTEXT_SCHEMA_MISMATCH")
    return {"metadata_gate": "PASS", "metadata_gate_scope": "status, lock presence and context schema only",
            "remaining_scientific_guards": "original reporter must verify all bindings, records and payloads",
            "scientific_report_executed": False, "full_workflow_validated": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("selection", "complete"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--evidence-root")
    args = parser.parse_args()
    config = configuration()
    record = source_record(config, "fixed_report")
    check_source(record)
    result = {"entry": "fixed-report", "report_mode": args.mode,
              "mode": "preflight" if args.preflight else "plan", "source": record["source"],
              "evidence_layout": config["fixed_report"]["evidence_layout"],
              "requires": ["explicit external evidence root", "original status and selection locks",
                           "complete source bindings and retained inputs", "original scientific environment"],
              "scientific_execution": False, "full_workflow_validated": False}
    if args.preflight:
        result.update(preflight(args.mode, args.evidence_root, config))
    emit(result)

if __name__ == "__main__":
    run_cli(main)
