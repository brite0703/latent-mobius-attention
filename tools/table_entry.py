"""Plan, inspect external summary metadata, or explicitly format retained tables."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_entry_support import (ROOT, configuration, check_source, source_record,
                                  external_root, relative_file, read_json, require,
                                  digest, load_definition, emit, run_cli)

TABLES = ("4", "5", "6", "7", "8", "9", "12", "13")
SPEC_FIELDS = {"4": ("heads", "settings"), "5": ("heads", "settings"), "6": ("heads",),
               "7": ("procedures",), "8": ("declared_memory_tokens",),
               "9": ("campaigns", "settings", "heads"), "12": ("scopes",), "13": ("contrasts",)}

def inspect_metadata(metadata, config, root=ROOT):
    metadata = external_root(metadata, root)
    manifest = read_json(relative_file(metadata, "retained_summary_manifest.json"))
    layout = read_json(relative_file(metadata, "presentation_spec.json"))
    require(isinstance(manifest, dict) and isinstance(manifest.get("inputs"), dict)
            and isinstance(layout, dict) and isinstance(layout.get("tables"), dict), "TABLE_METADATA_SCHEMA_MISMATCH")
    schemas = config["table_export"]["input_schemas"]
    require(set(manifest["inputs"]) == set(schemas) and set(layout["tables"]) == set(TABLES),
            "TABLE_INPUT_INVENTORY_MISMATCH")
    for name, schema in schemas.items():
        record = manifest["inputs"][name]
        require(isinstance(record, dict) and isinstance(record.get("relative_path"), str)
                and type(record.get("bytes")) is int and record["bytes"] > 0
                and isinstance(record.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "TABLE_INPUT_RECORD_INVALID")
        source = relative_file(metadata, record["relative_path"])
        require(source.suffix.lower() == ".json", "TABLE_SUMMARY_MUST_BE_JSON")
        require(source.is_file(), "TABLE_SUMMARY_MISSING")
        require(source.stat().st_size == record["bytes"] and digest(source) == record["sha256"],
                "TABLE_SUMMARY_HASH_MISMATCH")
        saved = read_json(source)
        require(isinstance(saved, dict) and isinstance(saved.get(schema["rows_key"]), list),
                "TABLE_SUMMARY_SCHEMA_MISMATCH")
        for key in schema.get("additional_list_keys", []):
            require(isinstance(saved.get(key), list), "TABLE_SUMMARY_SCHEMA_MISMATCH")
    for table, spec in layout["tables"].items():
        require(isinstance(spec, dict) and spec.get("input") in schemas
                and spec["input"] == config["table_export"]["table_inputs"][table]
                and type(spec.get("expected_output_rows")) is int and spec["expected_output_rows"] > 0
                and type(spec.get("longtable")) is bool
                and all(isinstance(spec.get(key), str) for key in ("caption", "label", "columns", "header"))
                and all(key in spec for key in SPEC_FIELDS[table]), "TABLE_PRESENTATION_SCHEMA_MISMATCH")
        if spec["longtable"]:
            require(type(spec.get("column_count")) is int and spec["column_count"] > 0,
                    "TABLE_PRESENTATION_SCHEMA_MISMATCH")
    return metadata

def export_tables(metadata, output, tables, config, root=ROOT):
    require(output is not None, "EXPLICIT_OUTPUT_ROOT_REQUIRED")
    destination = Path(output).resolve()
    require(not destination.exists(), "NEW_OUTPUT_ROOT_REQUIRED")
    for protected in (root.resolve(), metadata.resolve()):
        require(not destination.is_relative_to(protected) and not protected.is_relative_to(destination),
                "OUTPUT_ROOT_MUST_BE_SEPARATE")
    program = check_source(source_record(config, "table_formatter"), root)
    captured = io.StringIO()
    old_argv = sys.argv
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            formatter = load_definition("public_retained_table_formatter", program)
            formatter.HERE = metadata
            formatter.PACKET = metadata
            sys.argv = ["export_retained_tables.py", "--export-retained-tables",
                        "--output-root", str(destination), "--tables", *tables]
            formatter.main()
    finally:
        sys.argv = old_argv
    return {"formatting": "COMPLETED", "tables": tables, "scientific_execution": False,
            "statistics_recomputed": False, "external_paths_published": False,
            "aggregate_numerical_correctness_validated": False, "full_TeX_equivalence_validated": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root")
    parser.add_argument("--output-root")
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--export-retained-tables", action="store_true")
    args = parser.parse_args()
    config = configuration()
    record = source_record(config, "table_formatter")
    check_source(record)
    if args.preflight or args.export_retained_tables:
        metadata = inspect_metadata(args.metadata_root, config)
        if args.export_retained_tables:
            emit(export_tables(metadata, args.output_root, list(dict.fromkeys(args.tables)), config))
            return
    emit({"entry": "table-export", "mode": "preflight" if args.preflight else "plan",
          "source": record["source"], "tables": list(dict.fromkeys(args.tables)),
          "metadata_files": ["retained_summary_manifest.json", "presentation_spec.json"],
          "external_aggregate_inputs": list(config["table_export"]["input_schemas"]),
          "metadata_identity_and_schema": "PASS" if args.preflight else "NOT_READ",
          "requires_for_export": ["explicit external metadata root with six retained aggregate JSON files",
                                  "manifest byte sizes and SHA256 identities", "presentation specification",
                                  "explicit export flag and new separate output directory"],
          "statistics_recomputed": False, "scientific_execution": False,
          "aggregate_numerical_correctness_validated": False, "full_TeX_equivalence_validated": False})

if __name__ == "__main__":
    run_cli(main)
