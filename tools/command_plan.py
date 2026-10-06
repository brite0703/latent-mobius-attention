"""Print a historical command plan without launching a scientific program."""
from pathlib import Path
import argparse
import json
import shlex

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / "configuration/entrypoints.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", choices=[r["id"] for r in registry["entries"]])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    entry = next(r for r in registry["entries"] if r["id"] == args.entrypoint)
    root = args.evidence_root.resolve()
    if entry["output_mode"] == "external_argument":
        if args.output is None:
            parser.error("This entrypoint requires --output")
        output = args.output.resolve()
        if output.is_relative_to(root):
            parser.error("Output must be outside original evidence")
        scientific_args = [v.format(output=str(output)) for v in entry["arguments"]]
        output_description = str(output)
    else:
        if args.output is not None:
            parser.error("Original receptor entrypoint has no --output argument")
        scientific_args = entry["arguments"]
        output_description = str(root / entry["output_relative_to_evidence_root"])
    print(json.dumps({
        "cwd": str(root),
        "command": shlex.join(["python", "-I", "-B", entry["source"], *scientific_args]),
        "output": output_description,
        "output_mode": entry["output_mode"],
        "scope": entry["scope"],
        "requires": entry["requires"],
        "executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
