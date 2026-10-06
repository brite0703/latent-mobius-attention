"""Verify packaged diagnostic bytes and the declared availability mapping."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((HERE/"manifest.json").read_text(encoding="utf-8"))
    by_path = {row["path"]:row for row in manifest["files"]}
    assert len(by_path) == len(manifest["files"])
    for relative, row in by_path.items():
        path = (HERE/relative).resolve()
        path.relative_to(HERE)
        assert path.is_file() and sha(path) == row["sha256"], relative
    counts = dict(packaged=0, shared_input_companion=0, omitted_editorial=0)
    for bound in manifest["direct_record_bindings"]:
        counts[bound["availability"]] += 1
        if bound["availability"] == "packaged":
            assert by_path[bound["package_path"]]["sha256"] == bound["sha256"]
        elif bound["availability"] == "shared_input_companion":
            assert any(r["original_path"] == bound["original_path"] and r["sha256"] == bound["sha256"]
                       for r in manifest["companion_files"])
        else:
            assert any(r["original_path"] == bound["original_path"] and r["sha256"] == bound["sha256"]
                       for r in manifest["omitted_editorial_dependencies"])
    print(json.dumps(dict(passed=True, packaged_files=len(by_path), direct_record_bindings=counts,
                          diagnostic_reexecuted=False, neural_training_or_test_scoring=False,
                          recursive_historical_workflow_closure_claimed=False)))


if __name__ == "__main__":
    main()
