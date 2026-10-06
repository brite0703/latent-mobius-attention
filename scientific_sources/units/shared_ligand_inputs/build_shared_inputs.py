"""Package the exact shared ligand inputs omitted by older evidence archives."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
COMP = RELEASE.parent
ROOT = COMP.parents[1]
DATA = ROOT / "revision_2026/data/lp_pdbbind"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    status = read(COMP / "conditional_residual/retained_pipeline_status.json")
    assert status["stage"] != "profile" or status["status"] == "complete", "Wait for retained timing"
    receipt_path = HERE / "shared_inputs_receipt.json"
    assert not receipt_path.exists(), "Preserve the completed input package"
    origin = read(DATA / "source_manifest.json")
    audit = read(DATA / "tensors_reconstructed/data_audit.json")
    locked = read(COMP / "receptor_context/sequence_implementation_lock.json")
    bound = {str(Path(row["path"]).resolve()): row["sha256"] for row in locked["sources"]}
    assert origin["commit"] == "cc363f80a6a696d562f290a87aa20173e60f6c52"
    assert sha(DATA / "LP_PDBBind.csv") == origin["csv_sha256"].lower() == audit["source_csv_sha256"]
    script = ROOT / "revision_2026/prepare_lp_pdbbind.py"
    assert sha(script) == audit["script_sha256"]
    assert sha(DATA / "LICENSE.txt") == read(HERE / "upstream_notice_receipt.json")["pinned_notice_sha256"]
    paths = [DATA / name for name in ("LP_PDBBind.csv", "LICENSE.txt", "source_manifest.json", "upstream_README.md")]
    paths += [DATA / "tensors_reconstructed" / name for name in
              ("pdbbind_train.pt", "pdbbind_val.pt", "pdbbind_test.pt", "sample_manifest.csv", "data_audit.json", "exclusions.json")]
    paths.append(script)
    rows = []
    for path in paths:
        digest = sha(path)
        key = str(path.resolve())
        if path.suffix == ".pt" or path.name == "sample_manifest.csv":
            assert digest == bound[key], "Shared input differs from the completed study"
        rows.append(dict(path="LMA/"+path.relative_to(ROOT).as_posix(), source=key, sha256=digest, bytes=path.stat().st_size))
    additions = [(HERE / "shared_inputs_README.md", "README.md"), (Path(__file__), "build_shared_inputs.py")]
    rows += [dict(path=name, source=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size) for path, name in additions]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = dict(schema="lma_shared_ligand_inputs_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    source_repository=origin["repository"], source_commit=origin["commit"],
                    split_counts={name: audit["splits"][name]["n"] for name in ("train", "val", "test")},
                    feature_dimension=audit["feature_dimension"], padded_atoms=audit["max_atoms"],
                    input_scope="Retained ligand tensors; no receptor or pocket coordinates in those tensors",
                    fresh_reconstruction_verified_by_this_package=False, public_deposition=False,
                    files=[{key: row[key] for key in ("path", "sha256", "bytes")} for row in rows])
    assert manifest["split_counts"] == dict(train=7384, val=958, test=2171)
    manifest_bytes = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    archive = RELEASE / "output" / ("shared_ligand_inputs_"+stamp+".zip")
    assert not archive.exists()
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for row in rows:
            output.write(row["source"], row["path"])
        output.writestr("manifest.json", manifest_bytes)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        assert len(check.namelist()) == len(rows)+1 and len(set(check.namelist())) == len(rows)+1
        for row in rows:
            assert hashlib.sha256(check.read(row["path"])).hexdigest() == row["sha256"]
        assert check.read("manifest.json") == manifest_bytes
    for row in rows:
        assert sha(row["source"]) == row["sha256"]
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), archive=str(archive), sha256=sha(archive),
                   bytes=archive.stat().st_size, files=len(rows)+1, payload_bytes=sum(row["bytes"] for row in rows),
                   all_member_hashes_and_crc_verified=True, source_inputs_unchanged=True,
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), sources=rows,
                   scope=manifest["input_scope"], fresh_reconstruction_verified=False, public_deposition=False)
    receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "sources"}))


if __name__ == "__main__":
    main()
