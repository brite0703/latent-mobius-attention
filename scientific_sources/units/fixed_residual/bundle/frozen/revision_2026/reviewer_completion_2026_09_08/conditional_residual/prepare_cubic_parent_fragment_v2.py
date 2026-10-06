"""Bind the ten already verified cubic parents to the campaign contract.

This produces an incomplete parent fragment, never an activated 15-parent
campaign, new native prediction or molecular substitute.
"""
from pathlib import Path

import torch

import campaign_control as control
import campaign_lifecycle as life

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/"first_cubic"


def main():
    root = HERE/"parent_fragments/cubic_v2"
    if root.exists():
        raise FileExistsError("Preserve the existing actual parent fragment")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    manifest_path = HERE/"caches/cubic_v1/manifest.json"
    audit_path = HERE/"cubic_cache_audit.json"
    manifest, audit = life.read(manifest_path), life.read(audit_path)
    assert manifest["passed"] and audit["passed"] and audit["source_manifest_sha256"] == life.sha(manifest_path)
    sources = {str(Path(s["path"]).resolve()):s for receipt in (manifest,audit) for s in receipt["sources"]+receipt.get("artifacts",[])}
    for record in list(sources.values())+manifest["files"]:
        life.verify_artifact(record)
    native = {r["seed"]:r for r in manifest["checks"]}
    independent = {(r["seed"],r["split"]):r for r in audit["checks"]}
    files = {Path(r["path"]).resolve():r for r in manifest["files"]}
    rows, records = [], 0
    for seed in range(100,110):
        check = native[seed]
        assert check["parent_state_exact"] and check["parent_validation_selection_reconciled"]
        row = dict(domain="cubic", seed=seed, status="available", reason=None,
                   parent_selection=life.artifact(PARENT/"selection_lock.json"),
                   parent_candidates=[life.artifact(PARENT/"candidates"/f"cubic_lma1_seed{seed}_lr{i}.json") for i in range(2)],
                   selected_id=check["selected_id"], parent_checkpoint=life.artifact(PARENT/"checkpoints"/(check["selected_id"]+".pt")),
                   caches={})
        assert row["parent_checkpoint"]["sha256"] == check["parent_checkpoint_sha256"]
        assert control.original_parent_choice(row) == row["selected_id"]
        split_checks = {r["split"]:r for r in check["splits"]}
        loaded = {}
        for split, count in (("train",2048),("val",1024)):
            path = HERE/"caches/cubic_v1"/f"seed{seed}_{split}.pt"
            entry = files[path.resolve()]
            row["caches"][split] = {key:entry[key] for key in ("path","sha256","content_digest")}
            a,b = split_checks[split],independent[seed,split]
            assert a["native_prediction_max_delta"] == a["independent_bucket_max_delta"] == 0.
            assert a["rows"] == b["rows"] == count
            assert all(b[key] for key in ("independent_integer_target_exact","source_row_order_exact","all_three_zero_residual_predictions_exact","cache_units_identity","content_digest_verified"))
            cache = life.load_cache(row["caches"][split],split,row["parent_checkpoint"]["sha256"])
            assert len(cache.ids) == count and cache.z.shape == (count,8,12) and cache.z.dtype == torch.float32
            assert (cache.target_mean,cache.target_sd) == (0.,1.)
            loaded[split] = cache
            records += count
        assert not set(loaded["train"].ids) & set(loaded["val"].ids)
        native_path = root/f"seed{seed}_cache_binding.json"
        life.atomic_json(native_path,dict(passed=True,domain="cubic",seed=seed,
            parent_checkpoint_sha256=row["parent_checkpoint"]["sha256"],
            cache_content_digests={split:r["content_digest"] for split,r in row["caches"].items()},
            sources=[life.artifact(Path(__file__)),life.artifact(manifest_path),life.artifact(audit_path)]+list(row["caches"].values()),
            scope="Binding of existing source-verified exact native and independent cache audits; no new native evaluation performed"),immutable=True)
        row["native_cache_audit"] = life.artifact(native_path)
        rows.append(row)
    output = dict(schema=1,prepared_utc=life.utc(),scope="unactivated actual cubic parent fragment",parents=rows,
                  parent_count=10,cached_record_parent_pairs=records,complete_fifteen_parent_bundle=False,
                  matched_parents_available=False,new_test_predictions=False,residual_fits=0,
                  sources=[life.artifact(Path(__file__)),life.artifact(HERE/"campaign_control.py"),life.artifact(HERE/"campaign_lifecycle.py"),
                           life.artifact(manifest_path),life.artifact(audit_path)]+list(sources.values()),
                  native_audit_basis="Existing twenty fitting/validation native-cache and independent-target receipts, all source hashes reverified")
    assert records == 30720
    life.atomic_json(root/"fragment.json",output,immutable=True)
    print(dict(parents=10,cached_record_parent_pairs=records,complete_fifteen_parent_bundle=False,new_test_predictions=False,residual_fits=0),flush=True)


if __name__ == "__main__":
    main()
