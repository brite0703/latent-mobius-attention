"""Check source bytes and syntax without importing scientific modules."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "configuration/source_origins.json").read_text())
    python_files = 0
    for record in manifest["files"]:
        source = ROOT / record["path"]
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Unsafe source path")
        content = source.read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Changed source: " + record["path"])
        if source.suffix == ".py":
            ast.parse(content.decode(), filename=record["path"])
            python_files += 1
    print(json.dumps({
        "file_identity": "PASS",
        "files_checked": len(manifest["files"]),
        "AST": "PASS",
        "Python_files": python_files,
        "science_imports": False,
        "model_execution": False,
    }))


if __name__ == "__main__":
    main()
