"""Execute the declared finite checks in a new, separately bound working tree."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(base, relative):
    path = (base/relative).resolve()
    path.relative_to(base.resolve())
    return path


def worker(spec_path):
    import runpy
    spec = json.loads(Path(spec_path).read_text())
    tree = Path(spec["tree"]).resolve()
    script = safe(tree, spec["script"])
    blocked = os.path.normcase(spec["historical_root"]).rstrip("\\/")
    compiled = []

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            value = os.path.normcase(os.fsdecode(args[0]))
            if value == blocked or value.startswith(blocked+os.sep):
                raise RuntimeError("Original-workspace read rejected: "+value)
        if event == "compile" and len(args) >= 2 and isinstance(args[1], str):
            value = Path(args[1])
            if value.is_absolute() and value.is_relative_to(tree):
                compiled.append(str(value))
    sys.addaudithook(audit)
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
    Path(spec["worker_receipt"]).write_text(json.dumps(dict(script=str(script), compiled_sources=sorted(set(compiled)),
        original_workspace_open_rejected_by_audit_hook=True,
        instrumentation_scope="Observed Python file opens; not an operating-system sandbox."), indent=2)+"\n")


def compare(left, right, trail, counts):
    if isinstance(left, dict):
        # Generated source identities are rebound to actual working inputs below.
        ignored = {"created_utc", "completed_utc", "computed_utc", "sources"}
        lk, rk = set(left)-ignored, set(right)-ignored
        assert lk == rk, (trail, lk, rk)
        for key in sorted(lk):
            compare(left[key], right[key], trail+"."+key, counts)
    elif isinstance(left, list):
        assert len(left) == len(right), trail
        for index, (a,b) in enumerate(zip(left,right)):
            compare(a,b,f"{trail}[{index}]",counts)
    elif isinstance(left, float):
        assert isinstance(right, (int,float)) and math.isfinite(left) and math.isfinite(right), trail
        delta=abs(left-right)
        assert delta <= 1e-10+1e-10*abs(right), (trail,left,right)
        counts["float_comparisons"] += 1
        counts["maximum_absolute_difference"] = max(counts["maximum_absolute_difference"],delta)
    else:
        assert left == right, (trail,left,right)
        counts["exact_field_comparisons"] += 1


def main():
    manifest_path = HERE/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        assert digest(safe(HERE,row["path"])) == row["sha256"], row["path"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = HERE/"runs"/stamp
    tree = run/"LMA"
    tree.mkdir(parents=True,exist_ok=False)
    for row in manifest["files"]:
        if row["role"] == "input":
            target = safe(tree,row["workspace_relative_path"])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(safe(HERE,row["path"]),target)
    (tree/"revision_2026/theory_revision/output").mkdir(parents=True,exist_ok=True)
    environment = dict(python=sys.version,executable=sys.executable,platform=sys.platform,
                       dependencies={name:importlib.metadata.version(name) for name in ("numpy","sympy","matplotlib")})
    summaries=[]
    for suite in manifest["suites"]:
        spec = dict(tree=str(tree),script=suite["script"],historical_root=manifest["historical_root"],
                    worker_receipt=str(run/(suite["id"]+"_worker.json")))
        spec_path=run/(suite["id"]+"_execution.json")
        spec_path.write_text(json.dumps(spec,indent=2)+"\n")
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                 MPLBACKEND="Agg",MPLCONFIGDIR=str(run/"matplotlib_config"))
        with (run/(suite["id"]+".stdout.log")).open("w",encoding="utf-8") as stdout, (run/(suite["id"]+".stderr.log")).open("w",encoding="utf-8") as stderr:
            completed=subprocess.run([sys.executable,"-I","-B",str(Path(__file__).resolve()),"worker",str(spec_path)],
                                     cwd=run,env=env,stdout=stdout,stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        assert completed.returncode == 0, (suite["id"],"Read its preserved stdout/stderr; do not overwrite the failed run")
        generated=safe(tree,suite["output"])
        actual=json.loads(generated.read_text())
        reference=json.loads(safe(HERE,suite["reference"]).read_text())
        stats=dict(float_comparisons=0,exact_field_comparisons=0,maximum_absolute_difference=0.)
        compare(actual,reference,suite["id"],stats)
        for source in actual.get("sources",[]):
            path=Path(source["path"]).resolve()
            path.relative_to(tree)
            assert digest(path)==source["sha256"], "Generated source identity differs"
        summaries.append(dict(id=suite["id"],passed=True,output=str(generated),output_sha256=digest(generated),
                              reference_sha256=digest(safe(HERE,suite["reference"])),**stats))
        print(json.dumps(dict(suite=suite["id"],passed=True,**stats)),flush=True)
    for row in manifest["files"]:
        assert digest(safe(HERE,row["path"])) == row["sha256"]
        if row["role"] == "input":
            assert digest(safe(tree,row["workspace_relative_path"])) == row["sha256"], "An input changed during a check"
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),passed=True,manifest_sha256=digest(manifest_path),
                 environment=environment,suites=summaries,all_input_sources_unchanged=True,
                 finite_check_scope=manifest["scope"],global_theorem_proved_by_this_execution=False,
                 priority_established=False,new_fitting_or_neural_test_scoring=False)
    path=run/"completion.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    (HERE/"latest_run.json").write_text(json.dumps(dict(run=str(run),receipt=str(path),sha256=digest(path)),indent=2)+"\n")
    print(json.dumps(dict(complete=True,suites=len(summaries),receipt=str(path))),flush=True)


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="worker":
        worker(sys.argv[2])
    else:
        main()
