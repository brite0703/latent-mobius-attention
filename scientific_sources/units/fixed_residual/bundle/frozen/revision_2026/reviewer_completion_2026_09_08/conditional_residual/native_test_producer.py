"""Retained common-selection-gated scientific test-cache production."""
from pathlib import Path

import campaign_control as control
import campaign_lifecycle as life
import native_cache
import native_inputs


def produce(root,access_record,bundle_record):
    root=Path(root).resolve()
    context=control.verify_context(root)
    if context["scope"]!=control.RETAINED:
        raise ValueError("The scientific test producer requires an actual retained residual context")
    control.test_access(root)
    if bundle_record!=context["parent_bundle"] or access_record!=life.artifact(root/"test_access.json"):
        raise ValueError("Native test production has a different selection/parent authorization")
    bundle=control.checked(bundle_record)
    if life.artifact(__file__) not in bundle["producer_sources"]:
        raise ValueError("This scientific test producer was not frozen before residual fitting")
    manifest_path=root/"native_test_caches/manifest.json"
    if manifest_path.exists():
        control.validate_test_manifest(root,manifest_path)
        return manifest_path
    authorization=dict(root=str(root),test_access=access_record,parent_bundle=bundle_record)
    rows=[]
    sources={r["path"]:r for r in bundle["producer_sources"]}
    for parent_row in bundle["parents"]:
        row=dict(domain=parent_row["domain"],seed=parent_row["seed"],status=parent_row["status"],reason=parent_row["reason"])
        if parent_row["status"]=="missing_parent":
            row["cache"]=None
        else:
            parent,layer_path,payload=native_inputs.parent_and_inputs(parent_row,"test",authorization=authorization)
            cache,check=native_cache.extract(parent,layer_path,payload,parent_row["parent_checkpoint"]["sha256"])
            row.update(parent_checkpoint=parent_row["parent_checkpoint"],native_prediction_and_bucket_equal=check["native_prediction_and_bucket_equal"],
                cache=native_cache.save(cache,root/"native_test_caches"/f"{row['domain']}_seed{row['seed']}.pt"),checks=check)
            for source in payload["sources"]+[row["cache"]]:
                sources[str(Path(source["path"]).resolve())]=source
        rows.append(row)
    for source in sources.values():
        life.verify_artifact(source)
    manifest=dict(schema=1,scope=context["scope"],test_access=access_record,parent_bundle=bundle_record,
        native_cache_audit_passed=True,producer_sources=bundle["producer_sources"],parents=rows,sources=list(sources.values()),
        scope_note="Native float32 predictions and buckets; independent same-shape reconstruction and original source/target identities checked; no residual refit")
    life.atomic_json(manifest_path,manifest,immutable=True)
    control.validate_test_manifest(root,manifest_path)
    return manifest_path
