"""Plan/preflight or guarded definition imports from public, hash-bound sources."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_entry_support import (ROOT, configuration, verify_n20_sources, dependencies_present,
                                  import_n20_definitions, emit, require, run_cli)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", choices=("fp32", "warm-fp64", "random-fp64", "native"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    config = configuration()
    records = verify_n20_sources(config)
    if args.import_only:
        emit(import_n20_definitions(args.entry))
        return
    available = dependencies_present() if args.preflight else None
    if args.preflight:
        require(all(available.values()), "IMPORT_DEPENDENCY_UNAVAILABLE")
    emit({"entry": args.entry, "mode": "preflight" if args.preflight else "plan",
          "source": config["n20"]["entries"][args.entry]["source"],
          "public_sources_hash_checked": len(records), "model_helper_mappings": 11,
          "layout": "temporary public code only; no scientific input payloads",
          "dependencies_present": available, "scientific_execution": False,
          "requires_for_science": ["external inputs and locks", "new isolated outputs and protocol/time budget",
                                   "appropriate scientific environment", "separate execution authorization"],
          "full_workflow_validated": False})

if __name__ == "__main__":
    run_cli(main)
