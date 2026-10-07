"""Public entry checks and guarded definition imports. Standard-library plans."""
import ast
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
class EntryError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code

def require(condition, code):
    if not condition:
        raise EntryError(code)

def relative_file(root, name):
    require(isinstance(name, str) and "\\" not in name and not re.match(r"^[A-Za-z]:", name), "UNSAFE_RELATIVE_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts), "UNSAFE_RELATIVE_PATH")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(), "UNSAFE_SOURCE_PATH")
    return result

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise EntryError("JSON_INPUT_UNAVAILABLE_OR_INVALID") from None

def configuration(root=ROOT):
    config = read_json(root / "configuration/public_entrypoints.json")
    require(config.get("schema_version") == 1, "REGISTRY_SCHEMA_MISMATCH")
    require(config.get("baseline_commit") == "50472812494054d3c8ea03efe256f3b3b2e47bdf", "BASELINE_MISMATCH")
    require(len(config["n20"]["model_sources"]) == 11, "MODEL_MAPPING_INCOMPLETE")
    return config

def check_source(record, root=ROOT):
    path = relative_file(root, record["source"])
    require(path.is_file(), "PUBLIC_SOURCE_MISSING")
    content = path.read_bytes()
    require(len(content) == record["bytes"] and hashlib.sha256(content).hexdigest() == record["sha256"], "PUBLIC_SOURCE_HASH_MISMATCH")
    if path.suffix == ".py":
        ast.parse(content.decode("utf-8-sig"), filename=record["source"])
    return path

def source_record(config, identifier):
    require(identifier in config["source_records"], "UNKNOWN_PUBLIC_SOURCE")
    return config["source_records"][identifier]

def verify_n20_sources(config, root=ROOT):
    records = config["n20"]["model_sources"] + [
        source_record(config, identifier) for identifier in config["n20"]["layout_sources"]
    ]
    origins = read_json(root / "configuration/source_origins.json")
    indexed = {r["path"]: r for r in origins["files"]}
    for record in records:
        expected = indexed.get(record["source"])
        require(expected is not None and expected["sha256"] == record["sha256"] and expected["bytes"] == record["bytes"],
                "SOURCE_ORIGIN_BINDING_MISMATCH")
        check_source(record, root)
    return records

def external_root(value, root=ROOT):
    require(value is not None, "EXPLICIT_EXTERNAL_ROOT_REQUIRED")
    path = Path(value).resolve()
    require(path.is_dir(), "EXTERNAL_ROOT_UNAVAILABLE")
    require(not path.is_relative_to(root.resolve()) and not root.resolve().is_relative_to(path), "EXTERNAL_ROOT_MUST_BE_SEPARATE")
    return path

def dependencies_present():
    return {name: importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "sklearn", "matplotlib")}

@contextmanager
def definition_guard(torch, numpy, scientific_root):
    """Block execution and data loads before importing any captured definitions."""
    require(not torch.cuda.is_initialized(), "CUDA_ALREADY_INITIALIZED")
    patches = []
    def blocked(*args, **kwargs):
        raise EntryError("SCIENTIFIC_EXECUTION_BLOCKED")
    targets = [
        (torch, "load"), (torch.jit, "load"), (numpy, "load"), (numpy, "loadtxt"),
        (numpy, "genfromtxt"), (numpy, "memmap"), (torch.nn.Module, "__init__"),
        (torch.nn.Module, "__call__"), (torch.nn.Module, "_call_impl"),
        (torch.Tensor, "backward"), (torch.autograd, "backward"), (torch.autograd, "grad"),
        (torch.cuda, "init"), (torch.cuda, "_lazy_init"), (torch.Tensor, "cuda"),
        (torch.nn.Module, "cuda"),
    ]
    state = {"active": True}
    def audit(event, args):
        if state["active"] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(args[0])).suffix.lower()
            if suffix in {".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".safetensors", ".csv", ".parquet", ".h5", ".hdf5"}:
                raise EntryError("SCIENTIFIC_DATA_READ_BLOCKED")
    sys.addaudithook(audit)
    observed = set()
    scientific_prefix = str(scientific_root.resolve()) + os.sep
    forbidden = {"main", "forward", "backward", "train", "evaluate", "fit", "metrics", "configure",
                 "check_inputs", "verify_inputs", "instantiate", "predict", "step", "head_sizes", "sizes", "set_seed"}
    def profile(frame, event, arg):
        if event != "call":
            return
        filename = frame.f_code.co_filename
        if filename.startswith(scientific_prefix):
            require(frame.f_code.co_name not in forbidden, "SCIENTIFIC_MAIN_OR_FUNCTION_BLOCKED")
            if frame.f_code.co_name == "<module>":
                observed.add(Path(filename).resolve())
        elif frame.f_code.co_name in {"forward", "backward"}:
            raise EntryError("MODEL_FORWARD_BACKWARD_BLOCKED")
    previous_profile = sys.getprofile()
    try:
        for obj, name in targets:
            patches.append((obj, name, getattr(obj, name)))
            setattr(obj, name, blocked)
        sys.setprofile(profile)
        yield observed
        require(not torch.cuda.is_initialized(), "CUDA_INITIALIZATION_BLOCKED")
    finally:
        state["active"] = False
        sys.setprofile(previous_profile)
        for obj, name, original in reversed(patches):
            setattr(obj, name, original)

def load_definition(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

@contextmanager
def definition_source_view(stage, records, omissions):
    """Compile an explicit definition view without changing captured file bytes."""
    mapped = {(stage / record["layout_path"]).resolve(): record for record in records}
    requested = {record["source"]: omission for omission in omissions
                 for record in records if record["source"] == omission["source"]}
    require(len(requested) == len(omissions), "DEFINITION_OMISSION_MAPPING_MISMATCH")
    skipped = set()
    original = importlib.machinery.SourceFileLoader.get_code
    def get_code(loader, fullname):
        path = Path(loader.path).resolve()
        if not path.is_relative_to(stage.resolve()):
            return original(loader, fullname)
        require(path in mapped, "UNMAPPED_SCIENTIFIC_IMPORT_BLOCKED")
        record = mapped[path]
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == record["sha256"], "DEFINITION_SOURCE_CHANGED")
        tree = ast.parse(content.decode("utf-8-sig"), filename=str(path))
        omission = requested.get(record["source"])
        if omission:
            matches = []
            for node in tree.body:
                call = None
                if omission["kind"] == "assignment" and isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == omission["target"]:
                        call = node.value
                elif omission["kind"] == "expression" and isinstance(node, ast.Expr):
                    call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    if call.func.id == omission["callable"] and not call.keywords and (
                        (not call.args and omission["args"] == []) or
                        (len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                         and [call.args[0].value] == omission["args"])):
                        matches.append(node)
            require(len(matches) == 1, "DEFINITION_OMISSION_NOT_FOUND")
            tree.body.remove(matches[0])
            skipped.add(record["source"])
        return compile(tree, str(path), "exec")
    importlib.machinery.SourceFileLoader.get_code = get_code
    try:
        yield skipped
    finally:
        importlib.machinery.SourceFileLoader.get_code = original

def import_n20_definitions(entry, root=ROOT):
    config = configuration(root)
    require(entry in config["n20"]["entries"], "UNKNOWN_N20_ENTRY")
    records = verify_n20_sources(config, root)
    require(all(dependencies_present().values()), "IMPORT_DEPENDENCY_UNAVAILABLE")
    require(not any(name in sys.modules for name in ("verify_inputs", "component_models", "study", "models",
                    "parity_models", "parity_common", "lma_revision", "pdbbind_rerun", "training_chain_checks")),
            "SCIENTIFIC_MODULE_ALREADY_IMPORTED")
    old_path, old_env = list(sys.path), os.environ.get("MPLCONFIGDIR")
    output = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="lboia-public-definitions-") as temporary:
        stage = Path(temporary)
        reverse = {}
        for record in records:
            target = relative_file(stage, record["layout_path"])
            require(not target.exists(), "LAYOUT_TARGET_COLLISION")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(check_source(record, root), target)
            require(digest(target) == record["sha256"], "STAGED_SOURCE_HASH_MISMATCH")
            reverse[target.resolve()] = record["source"]
        settings = stage / "library-settings"
        settings.mkdir()
        os.environ["MPLCONFIGDIR"] = str(settings)
        try:
            with redirect_stdout(output), redirect_stderr(output):
                import numpy
                import torch
                with definition_source_view(stage, records, config["n20"]["definition_omissions"]) as skipped, definition_guard(torch, numpy, stage) as observed:
                    helper = source_record(config, "public_verify_inputs")
                    helper_module = load_definition("verify_inputs", stage / helper["layout_path"])
                    helper_module.configure_layout(stage / config["n20"]["first_layout"], config["n20"]["model_sources"])
                    target = stage / config["n20"]["entries"][entry]["layout_path"]
                    sys.path.insert(0, str(target.parent))
                    load_definition("public_n20_" + entry.replace("-", "_"), target)
                    # Native verification imports its models inside main; check those definitions explicitly.
                    if entry == "native":
                        helper_module.scientific_imports()
                imported = sorted(reverse[p] for p in observed)
            return {"entry": entry, "definition_import": "PASS", "source_hashes": "PASS",
                    "definition_import_scope": "AST definition view; explicit import-time size and seed calls omitted",
                    "import_time_calls_omitted": sorted(skipped), "captured_source_bytes_changed": False,
                    "model_helper_mappings": 11, "imported_public_sources": imported,
                    "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "torch": str(torch.__version__)},
                    "guarded": ["scientific_main", "data_load", "model_construction", "forward", "backward", "CUDA_initialization"],
                    "cuda_initialized": False, "scientific_execution": False,
                    "import_messages_published": False, "full_workflow_validated": False}
        finally:
            for name, module in list(sys.modules.items()):
                filename = getattr(module, "__dict__", {}).get("__file__")
                if isinstance(filename, str) and Path(filename).resolve().is_relative_to(stage):
                    sys.modules.pop(name, None)
            sys.path[:] = old_path
            if old_env is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_env

def emit(result):
    print(json.dumps(result, indent=2))

def run_cli(function):
    try:
        function()
    except EntryError as error:
        emit({"status": "BLOCKED", "reason": error.code, "scientific_execution": False})
        raise SystemExit(2) from None
    except Exception as error:
        emit({"status": "BLOCKED", "reason": "ENTRY_CHECK_FAILED", "error_type": type(error).__name__,
              "scientific_execution": False})
        raise SystemExit(2) from None
