"""Public code-layout plumbing; does not implement the original input audit."""
from pathlib import Path, PurePosixPath
import hashlib
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE / "original_workspace"
REV = WORK / "LMA/revision_2026"
CORE = REV / "neural_reviewer_study_2026_09_07"
COMP = REV / "reviewer_completion_2026_09_08/components_width"
PAR = REV / "reviewer_completion_2026_09_08/synthetic_parity"
DATA = REV / "data/lp_pdbbind/tensors_reconstructed"
_configured = False

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()

def configure_layout(first_root, public_model_records):
    global HERE, WORK, REV, CORE, COMP, PAR, DATA, _configured
    if len(public_model_records) != 11:
        raise ValueError("PUBLIC_MODEL_MAPPING_INCOMPLETE")
    HERE = Path(first_root).resolve()
    WORK = HERE / "original_workspace"
    for record in public_model_records:
        logical = PurePosixPath(record["logical_path"])
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError("UNSAFE_PUBLIC_LAYOUT")
        path = WORK / logical
        if not path.resolve().is_relative_to(WORK.resolve()) or not path.is_file():
            raise ValueError("PUBLIC_LAYOUT_SOURCE_MISSING")
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("PUBLIC_LAYOUT_SOURCE_CHANGED")
    REV = WORK / "LMA/revision_2026"
    CORE = REV / "neural_reviewer_study_2026_09_07"
    COMP = REV / "reviewer_completion_2026_09_08/components_width"
    PAR = REV / "reviewer_completion_2026_09_08/synthetic_parity"
    DATA = REV / "data/lp_pdbbind/tensors_reconstructed"
    _configured = True

def scientific_imports():
    if not _configured:
        raise ValueError("EXPLICIT_PUBLIC_LAYOUT_REQUIRED")
    for path in (REV, CORE, COMP, PAR):
        sys.path.insert(0, str(path))
    import component_models
    import study
    import parity_models
    import parity_common
    return component_models, study, parity_models, parity_common

def main():
    raise SystemExit("Public helper supplies import definitions only; original data validation is not implemented.")

if __name__ == "__main__":
    main()

