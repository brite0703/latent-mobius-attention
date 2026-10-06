"""Source-bound native fitting inputs; scientific test reads require selection."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import torch

import campaign_control as control
import campaign_lifecycle as life

HERE = Path(__file__).resolve().parent
CUBIC = HERE.parent/"first_cubic"
POCKET = HERE.parent/"receptor_context/pocket_reconstruction"


def imported(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules(domain):
    if domain == "cubic":
        return imported("residual_native_cubic_models",CUBIC/"neural_models.py"), imported("residual_native_cubic_generator",CUBIC/"generator.py")
    if domain != "ligand_contact":
        raise ValueError("Unknown native domain")
    sys.path.insert(0,str(POCKET))
    import matched_models
    import matched_execution
    return matched_models,matched_execution


def authorize_test(domain,seed,authorization):
    if not isinstance(authorization,dict) or set(authorization)!={"root","test_access","parent_bundle"}:
        raise ValueError("Scientific test data require a bound retained common-selection authorization")
    root = Path(authorization["root"]).resolve()
    context = control.verify_context(root)
    if context["scope"] != control.RETAINED or context["parent_bundle"] != authorization["parent_bundle"]:
        raise ValueError("A discarded or unrelated context cannot decode scientific test data")
    control.test_access(root)
    if life.artifact(root/"test_access.json") != authorization["test_access"]:
        raise ValueError("The native test reader has a different common-selection access record")
    bundle = control.checked(context["parent_bundle"])
    parents = [row for row in bundle["parents"] if (row["domain"],row["seed"])==(domain,seed)]
    if len(parents)!=1 or parents[0]["status"]!="available":
        raise ValueError("A scientific test read requires its own available frozen parent")
    return parents[0]


def read_inputs(domain,seed,split,*,authorization=None):
    if domain not in ("cubic","ligand_contact") or seed not in (range(100,110) if domain=="cubic" else range(42,47)) or split not in ("train","val","test"):
        raise ValueError("Unknown prescribed native input request")
    # This call must precede even the source-archive/manifest reader below.
    if split=="test":
        authorize_test(domain,seed,authorization)
    if torch.get_num_threads()!=1 or torch.cuda.is_initialized():
        raise ValueError("Native residual preparation requires one CPU thread and no CUDA runtime")
    if domain=="cubic":
        lock_path = CUBIC/"implementation_lock.json"
        locked = life.read(lock_path)
        path = CUBIC/"data"/f"seed{seed}.npz"
        entries = [r for r in locked["files"] if Path(r["path"]).resolve()==path.resolve()]
        if len(entries)!=1:
            raise ValueError("The original cubic lock must bind this partition archive")
        life.verify_artifact(entries[0])
        generator_path = CUBIC/"generator.py"
        generator_entry = next(r for r in locked["files"] if Path(r["path"]).resolve()==generator_path.resolve())
        life.verify_artifact(generator_entry)
        generator = imported("residual_native_cubic_data_generator",generator_path)
        with np.load(path) as archive:
            ids = archive[split+"_ids"].copy()
        np.testing.assert_array_equal(ids,generator.split(seed)[split])
        bits = ((ids[:,None] >> np.arange(12)) & 1).astype(np.uint8)
        numerators = generator.target_numerators(bits,"cubic")
        supports = (7,11,19,97,161,1792,2816,1092)
        signs = (1,-1,1,1,-1,1,-1,1)
        reference = [sum(sign*(1 if (int(identifier)&support).bit_count()%2 else -1)
                         for support,sign in zip(supports,signs)) for identifier in ids]
        np.testing.assert_array_equal(numerators,np.asarray(reference,dtype=np.int64))
        truth = torch.from_numpy(numerators/math.sqrt(8)).double()
        payload = dict(split=split,ids=[f"subset:{int(i)}" for i in ids],
                       inputs=dict(x=torch.eye(12,dtype=torch.float32).expand(len(ids),-1,-1).clone(),
                                   mask=torch.as_tensor(bits.astype(bool))),
                       truth=truth,target_mean=0.,target_sd=1.)
        sources = [life.artifact(lock_path),entries[0],generator_entry]
    else:
        _,reader = modules(domain)
        manifest_path = reader.CACHE/"manifest.json"
        manifest = life.read(manifest_path)
        manifest_identity = life.artifact(manifest_path)
        blob = reader.load_split(split,manifest)
        scale = manifest["target_scale"]
        payload = dict(split=split,ids=list(blob["ids"]),
            inputs=dict(x=blob["X"],mask=blob["mask"],adj=blob["adj"],contact=blob["contact"]),
            truth=blob["y"].double(),target_mean=float(scale["mean"]),target_sd=float(scale["population_sd"]))
        data_record = next(r for r in manifest["files"] if r["split"]==split)
        sources = [manifest_identity,{k:data_record[k] for k in ("path","sha256")}]
    expected = {("cubic","train"):2048,("cubic","val"):1024,("cubic","test"):1024,
                ("ligand_contact","train"):1096,("ligand_contact","val"):150,("ligand_contact","test"):366}[domain,split]
    if len(payload["ids"])!=expected or len(set(payload["ids"]))!=expected:
        raise ValueError("Native input cohort or unique record count changed")
    for bound in sources:
        life.verify_artifact(bound)
    payload["sources"] = sources
    return payload


def load_parent(row):
    chosen = control.original_parent_choice(row)
    if chosen is None:
        raise ValueError("Two failed original rates do not supply a native parent")
    domain,seed = row["domain"],row["seed"]
    study = CUBIC if domain=="cubic" else POCKET/"matched_study"
    lock_path = study/"implementation_lock.json"
    locked = life.read(lock_path)
    lock_sha = life.sha(lock_path)
    if control.checked(row["parent_selection"])["implementation_lock_sha256"]!=lock_sha:
        raise ValueError("Parent selection and implementation lock disagree")
    sources = locked["files" if domain=="cubic" else "sources"]
    python_sources = [s for s in sources if Path(s["path"]).suffix==".py"]
    if not python_sources:
        raise ValueError("The original parent must bind its Python implementation")
    for bound in python_sources:
        life.verify_artifact(bound)
    candidates = [control.checked(s) for s in row["parent_candidates"]]
    candidate = next(c for c in candidates if (c["spec"]["id"] if domain=="cubic" else c["id"])==chosen)
    if candidate["implementation_lock_sha256"]!=lock_sha:
        raise ValueError("The selected native candidate belongs to another implementation")
    checkpoint = torch.load(row["parent_checkpoint"]["path"],map_location="cpu",weights_only=True)
    if domain=="cubic":
        if set(checkpoint)!={"state_dict","spec","implementation_lock_sha256"} or checkpoint["spec"]!=candidate["spec"] or checkpoint["implementation_lock_sha256"]!=lock_sha:
            raise ValueError("Cubic checkpoint metadata differs from its selected candidate")
        state = checkpoint["state_dict"]
        scale = (0.,1.)
    else:
        state = checkpoint
        scale = (candidate["target_mean"],candidate["target_population_sd"])
        if scale!=(locked["target_scale"]["mean"],locked["target_scale"]["population_sd"]):
            raise ValueError("Matched candidate and original fitting scale disagree")
    with torch.random.fork_rng(devices=[]):
        module,_ = modules(domain)
        model = module.build(seed,"lma1") if domain=="cubic" else module.build_model(seed,"ligand_contact","lma1")
    model.load_state_dict(state,strict=True)
    model.eval()
    life.verify_artifact(row["parent_checkpoint"])
    return model,("head.layers.0" if domain=="cubic" else "pool.layers.0"),scale,[life.artifact(lock_path)]+python_sources+[row["parent_checkpoint"]]


def parent_and_inputs(row,split,*,authorization=None):
    # Authorization is checked before any scientific data read, including the
    # parent loader's original candidate and implementation records.
    if split=="test":
        bound_parent=authorize_test(row["domain"],row["seed"],authorization)
        if bound_parent!=row:
            raise ValueError("The test request substituted a different parent row")
    model,layer_path,scale,sources=load_parent(row)
    payload=read_inputs(row["domain"],row["seed"],split,authorization=authorization)
    lock_path=CUBIC/"implementation_lock.json" if row["domain"]=="cubic" else POCKET/"matched_study/implementation_lock.json"
    locked=life.read(lock_path)
    original={str(Path(s["path"]).resolve()):s["sha256"] for s in locked["files" if row["domain"]=="cubic" else "sources"]}
    for bound in payload["sources"]:
        path=str(Path(bound["path"]).resolve())
        if path==str(lock_path.resolve()):
            continue
        if original.get(path)!=bound["sha256"]:
            raise ValueError("The native input no longer matches the selected parent's original source lock")
    if scale!=(payload["target_mean"],payload["target_sd"]):
        raise ValueError("The native input target transformation differs from its selected parent")
    payload["sources"] += sources+[row["parent_selection"]]+row["parent_candidates"]
    return model,layer_path,payload
