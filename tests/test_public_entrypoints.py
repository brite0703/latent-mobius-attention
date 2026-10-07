"""Code, metadata and guarded-import checks; no scientific workflow execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import public_entry_support as support
import fixed_report_entry as fixed
import table_entry as tables

DEPENDENCIES = all(support.dependencies_present().values())

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

class EntryChecks(unittest.TestCase):
    def cli(self, script, *arguments, root=ROOT, expected=0):
        result = subprocess.run([sys.executable, "-I", "-B", "tools/" + script, *arguments],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, expected, result.stdout)
        self.assertEqual(result.stderr, "")
        parsed = json.loads(result.stdout)
        self.assertNotIn(str(root), result.stdout)
        self.assertFalse(parsed.get("scientific_execution"))
        return parsed

    def test_all_plans_use_public_sources_without_external_inputs(self):
        for entry in ("fp32", "warm-fp64", "random-fp64", "native"):
            with self.subTest(entry=entry):
                result = self.cli("n20_entry.py", entry)
                self.assertEqual(result["model_helper_mappings"], 11)
                self.assertIsNone(result["dependencies_present"])
                self.assertFalse(result["full_workflow_validated"])
        for mode in ("selection", "complete"):
            self.assertEqual(self.cli("fixed_report_entry.py", mode)["mode"], "plan")
        result = self.cli("table_entry.py")
        self.assertEqual(result["metadata_identity_and_schema"], "NOT_READ")
        self.assertEqual(len(result["external_aggregate_inputs"]), 6)

    def test_standard_library_plans_do_not_import_scientific_dependencies(self):
        for filename, arguments in [("n20_entry.py", ["fp32"]), ("fixed_report_entry.py", ["selection"]),
                                    ("table_entry.py", [])]:
            code = ("import runpy,sys; sys.argv=" + repr([filename, *arguments]) +
                    "; runpy.run_path(" + repr("tools/" + filename) + ",run_name='__main__'); " +
                    "assert not any(n=='torch' or n.startswith('torch.') or n=='numpy' "
                    "or n.startswith('numpy.') or n in ('parity_models','component_models') for n in sys.modules)")
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stderr, "")

    def test_external_roots_are_explicit_and_separate(self):
        for script, arguments in [("fixed_report_entry.py", ["complete"]), ("table_entry.py", [])]:
            result = self.cli(script, *arguments, "--preflight", expected=2)
            self.assertEqual(result["reason"], "EXPLICIT_EXTERNAL_ROOT_REQUIRED")
        with self.assertRaises(support.EntryError):
            support.external_root(str(ROOT), ROOT)
        with self.assertRaises(support.EntryError):
            support.external_root(str(ROOT.parent), ROOT)

    def test_mapping_rejects_unsafe_paths(self):
        for name in ("../escape.py", "/absolute.py", "C:/escape.py", "a\\b.py"):
            with self.subTest(name=name), self.assertRaises(support.EntryError):
                support.relative_file(ROOT, name)

    def test_public_mapping_and_original_source_hashes(self):
        config = support.configuration()
        records = support.verify_n20_sources(config)
        self.assertEqual(len(config["n20"]["model_sources"]), 11)
        self.assertEqual(len({r["logical_path"] for r in config["n20"]["model_sources"]}), 11)
        self.assertTrue(all(".." not in Path(r["layout_path"]).parts for r in records))
        for identifier in ("fixed_report", "table_formatter"):
            support.check_source(support.source_record(config, identifier))

    def test_changed_source_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="public-source-negative-") as name:
            root = Path(name) / "checkout"
            shutil.copytree(ROOT, root)
            config = support.configuration(root)
            source = root / config["n20"]["model_sources"][0]["source"]
            source.write_bytes(source.read_bytes() + b"\n")
            result = self.cli("n20_entry.py", "fp32", root=root, expected=2)
            self.assertEqual(result["reason"], "PUBLIC_SOURCE_HASH_MISMATCH")

    def test_origin_binding_is_required(self):
        config = copy.deepcopy(support.configuration())
        config["n20"]["model_sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(support.EntryError, "SOURCE_ORIGIN_BINDING_MISMATCH"):
            support.verify_n20_sources(config)

    def test_public_helper_requires_explicit_layout_and_rejects_main(self):
        path = ROOT / "scientific_sources/n20_diagnostics/fp32_recovery/support/verify_inputs.py"
        spec = importlib.util.spec_from_file_location("unconfigured_public_helper", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with self.assertRaisesRegex(ValueError, "EXPLICIT_PUBLIC_LAYOUT_REQUIRED"):
            helper.scientific_imports()
        with self.assertRaises(SystemExit):
            helper.main()

    @unittest.skipUnless(DEPENDENCIES, "Definition imports require NumPy and PyTorch")
    def test_all_four_guarded_definition_imports(self):
        for entry in ("fp32", "warm-fp64", "random-fp64", "native"):
            with self.subTest(entry=entry):
                result = self.cli("n20_entry.py", entry, "--import-only")
                self.assertEqual(result["definition_import"], "PASS")
                self.assertEqual(result["source_hashes"], "PASS")
                self.assertEqual(result["model_helper_mappings"], 11)
                self.assertEqual(len(result["import_time_calls_omitted"]), 3)
                self.assertFalse(result["captured_source_bytes_changed"])
                self.assertFalse(result["cuda_initialized"])
                self.assertFalse(result["full_workflow_validated"])
                self.assertTrue(any(n.endswith("/parity_models.py") for n in result["imported_public_sources"]))
                self.assertTrue(all(n.startswith("scientific_sources/") for n in result["imported_public_sources"]))

    @unittest.skipUnless(DEPENDENCIES, "Preflight requires scientific import dependencies")
    def test_n20_dependency_preflight(self):
        result = self.cli("n20_entry.py", "fp32", "--preflight")
        self.assertTrue(all(result["dependencies_present"].values()))
        self.assertFalse(result["full_workflow_validated"])

    @unittest.skipUnless(DEPENDENCIES, "Guard checks require NumPy and PyTorch")
    def test_guards_block_main_loads_models_and_cuda_without_execution(self):
        import numpy
        import torch
        original_load = torch.load
        with tempfile.TemporaryDirectory(prefix="public-guard-negative-") as name:
            root = Path(name)
            for function in ("main", "forward", "backward", "check_inputs"):
                namespace = {}
                exec(compile("def " + function + "():\n raise AssertionError('body must not run')\n",
                             str(root / "guarded.py"), "exec"), namespace)
                with support.definition_guard(torch, numpy, root):
                    with self.assertRaises(support.EntryError):
                        namespace[function]()
            with support.definition_guard(torch, numpy, root):
                for call in (lambda: torch.load("absent.pt"), lambda: numpy.load("absent.npz"),
                             lambda: torch.nn.Module(), lambda: torch.Tensor.backward(object()),
                             lambda: torch.cuda.init(), lambda: (root / "absent.pt").open("rb")):
                    with self.subTest(call=call.__code__.co_firstlineno), self.assertRaises(support.EntryError):
                        call()
            self.assertIs(torch.load, original_load)
            self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(DEPENDENCIES, "Definition imports require scientific import dependencies")
    def test_cached_scientific_modules_are_rejected_before_import(self):
        import types
        sys.modules["parity_models"] = types.ModuleType("parity_models")
        try:
            with self.assertRaisesRegex(support.EntryError, "SCIENTIFIC_MODULE_ALREADY_IMPORTED"):
                support.import_n20_definitions("fp32")
        finally:
            sys.modules.pop("parity_models", None)

    def fixed_fixture(self, root, status):
        config = support.configuration()
        study = root / config["fixed_report"]["evidence_layout"]
        write_json(study / "retained_pipeline_status.json", status)
        write_json(study / "retained_study/selection_lock.json", {"fixture": "schema only"})
        write_json(study / "retained_study/study_context.json", {"scope": "retained_residual_campaign"})
        return config, study

    def test_fixed_missing_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="public-fixed-negative-") as name:
            result = self.cli("fixed_report_entry.py", "complete", "--preflight",
                              "--evidence-root", name, expected=2)
            self.assertEqual(result["reason"], "JSON_INPUT_UNAVAILABLE_OR_INVALID")

    def test_fixed_original_timing_and_complete_gates(self):
        with tempfile.TemporaryDirectory(prefix="public-fixed-schema-") as name:
            root = Path(name)
            config, study = self.fixed_fixture(root, {"stage": "profile", "status": "failed",
                                                       "stages_finished": ["train", "evaluate"]})
            for mode in ("selection", "complete"):
                with self.subTest(mode=mode), self.assertRaises(support.EntryError):
                    fixed.preflight(mode, name, config)
            write_json(study / "retained_pipeline_status.json",
                       {"stage": "evaluate", "status": "complete", "stages_finished": ["train", "evaluate"]})
            with self.assertRaisesRegex(support.EntryError, "FIXED_COMPLETE_GATE_BLOCKED"):
                fixed.preflight("complete", name, config)

    def test_fixed_synthetic_metadata_gate_is_not_scientific_validation(self):
        with tempfile.TemporaryDirectory(prefix="public-fixed-schema-") as name:
            config, study = self.fixed_fixture(Path(name), {"stage": "profile", "status": "complete",
                                                           "stages_finished": ["train", "evaluate", "profile"]})
            result = fixed.preflight("complete", name, config)
            self.assertEqual(result["metadata_gate"], "PASS")
            self.assertFalse(result["scientific_report_executed"])
            self.assertFalse(result["full_workflow_validated"])
            (study / "retained_study/selection_lock.json").unlink()
            with self.assertRaisesRegex(support.EntryError, "FIXED_SELECTION_LOCK_MISSING"):
                fixed.preflight("selection", name, config)

    def table_fixture(self, root):
        config = support.configuration()
        inputs = {}
        for key, schema in config["table_export"]["input_schemas"].items():
            # Empty structural fixtures contain no retained aggregate values.
            saved = {schema["rows_key"]: []}
            for extra in schema.get("additional_list_keys", []):
                saved[extra] = []
            path = root / (key + ".json")
            write_json(path, saved)
            data = path.read_bytes()
            inputs[key] = {"relative_path": path.name, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}
        layout = {}
        for table, input_key in config["table_export"]["table_inputs"].items():
            spec = dict(input=input_key, expected_output_rows=1, longtable=False,
                        caption="Example", label="example", columns="l", header="Example")
            for field in tables.SPEC_FIELDS[table]:
                spec[field] = []
            layout[table] = spec
        write_json(root / "retained_summary_manifest.json", {"inputs": inputs})
        write_json(root / "presentation_spec.json", {"tables": layout})
        return config, inputs

    def test_table_schema_check_is_not_row_or_numerical_validation(self):
        with tempfile.TemporaryDirectory(prefix="public-table-schema-") as name:
            config, _ = self.table_fixture(Path(name))
            result = self.cli("table_entry.py", "--preflight", "--metadata-root", name)
            self.assertEqual(result["metadata_identity_and_schema"], "PASS")
            self.assertFalse(result["aggregate_numerical_correctness_validated"])
            self.assertFalse(result["full_TeX_equivalence_validated"])
            self.assertEqual(tables.inspect_metadata(name, config), Path(name))

    def test_table_missing_or_changed_input_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="public-table-negative-") as name:
            root = Path(name)
            result = self.cli("table_entry.py", "--preflight", "--metadata-root", name, expected=2)
            self.assertEqual(result["reason"], "JSON_INPUT_UNAVAILABLE_OR_INVALID")
            config, inputs = self.table_fixture(root)
            key = next(iter(inputs))
            (root / inputs[key]["relative_path"]).write_text("{}")
            with self.assertRaisesRegex(support.EntryError, "TABLE_SUMMARY_HASH_MISMATCH"):
                tables.inspect_metadata(name, config)

    def test_table_manifest_cannot_escape_external_root(self):
        with tempfile.TemporaryDirectory(prefix="public-table-negative-") as name:
            root = Path(name)
            config, inputs = self.table_fixture(root)
            inputs[next(iter(inputs))]["relative_path"] = "../escape.json"
            write_json(root / "retained_summary_manifest.json", {"inputs": inputs})
            with self.assertRaisesRegex(support.EntryError, "UNSAFE_RELATIVE_PATH"):
                tables.inspect_metadata(name, config)

    def test_table_export_requires_new_separate_output_before_loading_formatter(self):
        with tempfile.TemporaryDirectory(prefix="public-table-output-") as name:
            root = Path(name)
            config, _ = self.table_fixture(root)
            for output in (None, str(root), str(root / "new"), str(ROOT / "new")):
                with self.subTest(output=output), self.assertRaises(support.EntryError):
                    tables.export_tables(root, output, ["4"], config)
            self.assertNotIn("public_retained_table_formatter", sys.modules)

if __name__ == "__main__":
    unittest.main(verbosity=2)
