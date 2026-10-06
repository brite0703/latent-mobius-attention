"""Explicit retained-stage entry points; no activation follows from import."""
import argparse
from pathlib import Path

import torch

import campaign_control as control
import campaign_lifecycle as life
import native_parent_bundle
import native_profiles
import native_test_producer

HERE=Path(__file__).resolve().parent
ROOT=HERE/"retained_study"


def preflight():
    bundle_path=native_parent_bundle.build()
    preceding=native_parent_bundle.preceding_studies()
    evidence=preceding|dict(parent_cache_audit=life.artifact(native_parent_bundle.DESTINATION/"parent_cache_audit.json"),
        campaign_control_audit=life.artifact(HERE/"campaign_control_cpu_audit_v2.json"),
        native_cost_adapter_audit=life.artifact(HERE/"native_delivery_cpu_audit.json"))
    for role,source in evidence.items():
        receipt=control.checked(source)
        if receipt.get("complete" if role.endswith("profiles") else "passed") is not True:
            raise ValueError("Unfinished native residual prerequisite: "+role)
    paths=[HERE/name for name in ("native_input_specification.md","native_cost_specification.md",
        "native_inputs_cpu_audit.json","native_costs_cpu_audit.json","native_delivery_cpu_audit.json")]
    record=dict(passed=True,parent_bundle=life.artifact(bundle_path),evidence=evidence,
                sources=control.source_inventory(life.read(bundle_path))+[life.artifact(p) for p in paths],
                scope="Prerequisites only; a separate dated substantive activation decision is still required")
    path=native_parent_bundle.DESTINATION/"activation_preflight.json"
    life.atomic_json(path,record,immutable=True)
    return path


def activate():
    decision_path=HERE/"activation_decision.json"
    if not decision_path.exists():
        raise ValueError("A dated substantive activation decision must be recorded after assessing the preceding receptor results")
    preflight_path=preflight()
    return control.create_context(ROOT,native_parent_bundle.DESTINATION/"parent_bundle.json",scope=control.RETAINED,
        decision_path=decision_path,preflight_path=preflight_path)


def train():
    context=control.verify_context(ROOT)
    if context["scope"]!=control.RETAINED:
        raise ValueError("The native retained entry point cannot operate on an audit context")
    control.fit_all(ROOT)
    return control.lock_selection(ROOT)


def evaluate():
    manifest=control.prepare_test_caches(ROOT,native_test_producer.produce)
    return control.score_all(ROOT,manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","preflight","activate","train","evaluate","profile"))
    args=parser.parse_args()
    if torch.cuda.is_initialized():
        raise ValueError("Residual entry points require a fresh CPU-only process")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    actions=dict(prepare=native_parent_bundle.build,preflight=preflight,activate=activate,
                 train=train,evaluate=evaluate,profile=lambda:native_profiles.profile_all(ROOT))
    result=actions[args.action]()
    if isinstance(result,Path):
        print(str(result),flush=True)
    else:
        print(dict(stage=args.action,completed=True,cuda_initialized=torch.cuda.is_initialized()),flush=True)


if __name__=="__main__":
    main()
