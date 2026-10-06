"""Source-bound residual campaign control; no action occurs at import."""
from copy import deepcopy
import inspect
import math
from pathlib import Path

import torch

import campaign_lifecycle as life
import execution
import scoring
import selection

HERE = Path(__file__).resolve().parent
RETAINED = "retained_residual_campaign"
DISCARDED = "discarded_residual_campaign_audit"
CORE = ("models.py", "execution.py", "selection.py", "scoring.py", "costs.py",
        "campaign_lifecycle.py", "campaign_control.py", "campaign_control_specification.md",
        "campaign_lifecycle_origin.json", "campaign_lifecycle_origin.diff")
REPLICATES = tuple((domain, seed) for domain, seeds in selection.DOMAINS.items() for seed in seeds)
PREFLIGHT_ROLES = ("sequence_final_audit", "sequence_profiles", "matched_final_audit", "matched_profiles",
                   "parent_cache_audit", "campaign_control_audit", "native_cost_adapter_audit")


def checked(record):
    life.verify_artifact(record)
    return life.read(record["path"])


def fixture_root(root):
    root = Path(root).resolve()
    return root.is_relative_to(HERE) and bool(root.relative_to(HERE).parts) and root.relative_to(HERE).parts[0].startswith("discarded_campaign_")


def source_inventory(parent_bundle):
    files = [HERE/name for name in CORE]
    extras = parent_bundle["producer_sources"]
    if not extras:
        raise ValueError("A native cache producer must be frozen before fitting")
    for source in extras:
        life.verify_artifact(source)
        if Path(source["path"]).suffix != ".py":
            raise ValueError("Producer sources must identify actual Python files")
        files.append(Path(source["path"]).resolve())
    return [life.artifact(p) for p in sorted(set(files))]


def original_parent_choice(row):
    """Recheck both original rates using their actual candidate formats."""
    domain, seed = row["domain"], row["seed"]
    locked = checked(row["parent_selection"])
    root = Path(row["parent_selection"]["path"]).parent
    expected_ids = [f"{'cubic' if domain == 'cubic' else 'ligand_contact'}_lma1_seed{seed}_lr{i}" for i in range(2)]
    selected_rows = [r for r in locked["selections"] if r.get("head") == "lma1" and r.get("seed") == seed
                     and r.get("task" if domain == "cubic" else "setting") == domain]
    if len(selected_rows) != 1 or len(selected_rows[0]["candidate_ids"]) != 2 or set(selected_rows[0]["candidate_ids"]) != set(expected_ids):
        raise ValueError("The parent must be its original prescribed first-order choice")
    bound = {(root/s["path"]).resolve(): s["sha256"] for s in locked["candidate_records"]}
    records = []
    if len(row["parent_candidates"]) != 2:
        raise ValueError("Both original parent rates are required")
    for index, source in enumerate(row["parent_candidates"]):
        path = Path(source["path"]).resolve()
        if path != (root/"candidates"/(expected_ids[index]+".json")).resolve() or bound.get(path) != source["sha256"]:
            raise ValueError("Parent rate record differs from its original selection lock")
        value = checked(source)
        spec = value["spec"] if domain == "cubic" else value
        if any(spec.get(k) != v for k, v in dict(id=expected_ids[index], seed=seed, head="lma1", lr=selection.RATES[index], lr_index=index).items()):
            raise ValueError("Parent candidate identity or learning rate differs")
        if spec.get("task" if domain == "cubic" else "setting") != domain or value["status"] not in ("valid", "failed"):
            raise ValueError("Unfinished or substituted parent candidate")
        score = value.get("best_validation_loss" if domain == "cubic" else "best_validation_rmse")
        if value["status"] == "valid" and (not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0):
            raise ValueError("Invalid original parent validation metric")
        records.append((value, score, index))
    eligible = [r for r in records if r[0]["status"] == "valid"]
    chosen = min(eligible, key=lambda r: (r[1], r[2])) if eligible else None
    choice = selected_rows[0]
    if choice["selected_id"] != (expected_ids[chosen[2]] if chosen else None):
        raise ValueError("Parent selection does not minimize its original validation criterion")
    if chosen is None:
        return None
    value, score, _ = chosen
    if choice["validation_loss" if domain == "cubic" else "validation_rmse"] != score:
        raise ValueError("Original parent selected metric differs")
    expected_path = (root/"checkpoints"/(choice["selected_id"]+".pt")).resolve()
    if not isinstance(row.get("parent_checkpoint"), dict):
        raise ValueError("A selected original parent cannot be replaced by a missing checkpoint")
    if Path(row["parent_checkpoint"]["path"]).resolve() != expected_path:
        raise ValueError("The supplied parent checkpoint is not the selected original file")
    life.verify_artifact(row["parent_checkpoint"])
    if domain == "cubic":
        if (root/value["checkpoint"]).resolve() != expected_path or value["checkpoint_sha256"] != row["parent_checkpoint"]["sha256"] or choice["checkpoint_sha256"] != value["checkpoint_sha256"]:
            raise ValueError("Original cubic checkpoint binding differs")
    else:
        artifacts = {(root/s["path"]).resolve(): s["sha256"] for s in value["artifacts"]}
        if artifacts.get(expected_path) != row["parent_checkpoint"]["sha256"]:
            raise ValueError("Original matched checkpoint binding differs")
    return choice["selected_id"]


def validate_parents(bundle_path, scope, *, deep=True):
    bundle = life.read(bundle_path)
    if bundle["schema"] != 1 or bundle["scope"] != scope:
        raise ValueError("Parent bundle scope differs from the campaign")
    rows = bundle["parents"]
    keys = [(r["domain"], r["seed"]) for r in rows]
    if len(rows) != 15 or len(set(keys)) != 15 or set(keys) != set(REPLICATES):
        raise ValueError("Exactly all fifteen prescribed parent replicates are required")
    source_inventory(bundle)
    if not deep:
        return bundle
    molecular = {}
    for row in rows:
        chosen_id = original_parent_choice(row)
        if chosen_id is None:
            if row["status"] != "missing_parent" or row.get("parent_checkpoint") is not None or row.get("caches") or row.get("native_cache_audit") is not None or not row.get("reason", "").strip():
                raise ValueError("Both failed parent rates require an explicit missing-parent outcome")
            continue
        if row["status"] != "available" or row["selected_id"] != chosen_id or set(row["caches"]) != {"train", "val"}:
            raise ValueError("A selected parent must retain both of its verified caches")
        parent_sha = row["parent_checkpoint"]["sha256"]
        native = checked(row["native_cache_audit"])
        if native.get("passed") is not True or (native["domain"], native["seed"], native["parent_checkpoint_sha256"]) != (row["domain"], row["seed"], parent_sha):
            raise ValueError("Native fitting/validation audit does not bind this parent")
        for source in native["sources"]:
            life.verify_artifact(source)
        if not native["sources"]:
            raise ValueError("Native cache verification needs its source evidence")
        caches = {}
        for split in ("train", "val"):
            record = row["caches"][split]
            cache = life.load_cache(record, split, parent_sha)
            if cache.z.dtype != torch.float32 or cache.z.shape[-1] != (12 if row["domain"] == "cubic" else 8):
                raise ValueError("The campaign requires the native float32 bucket dimension")
            if native["cache_content_digests"][split] != cache.digest:
                raise ValueError("Native audit and supplied cache content differ")
            if row["domain"] == "cubic" and (cache.target_mean, cache.target_sd) != (0., 1.):
                raise ValueError("Cubic parents retain their original raw-target convention")
            if scope == RETAINED and len(cache.ids) != {("cubic", "train"):2048, ("cubic", "val"):1024, ("ligand_contact", "train"):1096, ("ligand_contact", "val"):150}[row["domain"], split]:
                raise ValueError("Retained fitting/validation count differs from the prescribed cohort")
            if row["domain"] == "ligand_contact":
                signature = (cache.ids, cache.truth.numpy().tobytes(), cache.target_mean, cache.target_sd)
                if split in molecular and signature != molecular[split]:
                    raise ValueError("Molecular parents do not use one common cohort, truth and scale")
                molecular[split] = signature
            caches[split] = cache
        if set(caches["train"].ids) & set(caches["val"].ids) or (caches["train"].target_mean, caches["train"].target_sd) != (caches["val"].target_mean, caches["val"].target_sd):
            raise ValueError("Parent fitting/validation identity or transformation is inconsistent")
    return bundle


def validate_activation(decision, preflight, bundle_path):
    if decision.get("decision") != "activate_conditional_residual" or not decision.get("reason", "").strip() or not decision.get("recorded_utc"):
        raise ValueError("Retained activation requires a dated substantive decision")
    if decision.get("outcome_informed") is not True or decision.get("tests_reused") is not True or decision.get("candidate_plan") != selection.candidate_plan():
        raise ValueError("Activation must retain the fixed allocation and disclose its evidential status")
    if preflight.get("passed") is not True or set(preflight["evidence"]) != set(PREFLIGHT_ROLES) or preflight["parent_bundle"] != life.artifact(bundle_path):
        raise ValueError("All preceding studies and residual prerequisite audits are required")
    for role, source in preflight["evidence"].items():
        receipt = checked(source)
        if receipt.get("complete" if role.endswith("profiles") else "passed") is not True:
            raise ValueError("An activation prerequisite is unfinished: "+role)
        for bound in receipt.get("sources", []) + receipt.get("artifacts", []):
            life.verify_artifact(bound)
    for source in preflight["sources"]:
        life.verify_artifact(source)
    required = {str((HERE/name).resolve()) for name in CORE}
    current = {str(Path(s["path"]).resolve()) for s in checked(preflight["evidence"]["campaign_control_audit"])["sources"]}
    if not required <= current:
        raise ValueError("The campaign audit does not cover the complete current controller")


def create_context(root, bundle_path, *, scope=DISCARDED, settings=None, decision_path=None, preflight_path=None):
    root = Path(root).resolve()
    if (root/"study_context.json").exists():
        raise FileExistsError("Preserve the existing campaign context")
    if scope == RETAINED:
        if root != HERE/"retained_study" or decision_path is None or preflight_path is None:
            raise ValueError("Retained activation needs its designated path and complete decision records")
        settings = dict(epoch_limit=100, batch_size=256, patience=8)
        validate_activation(life.read(decision_path), life.read(preflight_path), bundle_path)
    elif scope != DISCARDED or not fixture_root(root) or decision_path is not None or preflight_path is not None:
        raise ValueError("A discarded context cannot authorize retained fitting")
    settings = settings or dict(epoch_limit=2, batch_size=13, patience=8)
    if set(settings) != {"epoch_limit", "batch_size", "patience"} or any(type(v) is not int for v in settings.values()) or not (1 <= settings["epoch_limit"] <= 100 and 1 <= settings["batch_size"] <= 256 and 1 <= settings["patience"] <= 8):
        raise ValueError("Invalid fixed execution allowance")
    bundle = validate_parents(bundle_path, scope)
    context = dict(schema=1, scope=scope, created_utc=life.utc(), root=str(root), settings=settings,
                   candidate_plan=selection.candidate_plan(), sources=source_inventory(bundle),
                   parent_bundle=life.artifact(bundle_path),
                   activation_decision=life.artifact(decision_path) if decision_path else None,
                   preflight=life.artifact(preflight_path) if preflight_path else None,
                   retained_residual_study_activated=scope == RETAINED, test_scoring_allowed=False)
    life.atomic_json(root/"study_context.json", context, immutable=True)
    return verify_context(root)


def verify_context(root):
    root = Path(root).resolve()
    context = life.read(root/"study_context.json")
    scope = context["scope"]
    if context["schema"] != 1 or context["root"] != str(root) or scope not in (RETAINED, DISCARDED) or context["candidate_plan"] != selection.candidate_plan():
        raise ValueError("Campaign identity or fixed allocation differs")
    if context["test_scoring_allowed"] or context["retained_residual_study_activated"] != (scope == RETAINED):
        raise ValueError("A source context alone cannot authorize test evaluation")
    bundle = checked(context["parent_bundle"])
    if context["sources"] != source_inventory(bundle):
        raise ValueError("The complete campaign source inventory changed")
    origin = life.read(HERE/"campaign_lifecycle_origin.json")
    for role in ("source", "derived", "diff"):
        life.verify_artifact(origin[role])
    if scope == RETAINED:
        if root != HERE/"retained_study" or context["settings"] != dict(epoch_limit=100, batch_size=256, patience=8):
            raise ValueError("The retained schedule or output directory changed")
        checked(context["activation_decision"]); checked(context["preflight"])
    elif not fixture_root(root) or context["activation_decision"] is not None or context["preflight"] is not None:
        raise ValueError("Discarded and retained contexts cannot be interchanged")
    return context


def bind_all(root):
    context = verify_context(root)
    bundle = validate_parents(context["parent_bundle"]["path"], context["scope"])
    parents = {(r["domain"], r["seed"]):r for r in bundle["parents"]}
    for spec in context["candidate_plan"]:
        parent = parents[spec["domain"], spec["seed"]]
        folder = life.candidate_folder(root, spec)
        if parent["status"] == "missing_parent":
            life.missing_parent(root, spec["domain"], spec["seed"], parent["reason"])
        elif (folder/"binding.json").exists():
            binding, _, _ = life.read_binding(root, spec)
            expected_caches = {split:{key:record[key] for key in ("path", "sha256", "content_digest")}
                               for split, record in parent["caches"].items()}
            if binding["parent_checkpoint"] != parent["parent_checkpoint"] or binding["caches"] != expected_caches:
                raise ValueError("Candidate sources differ from the locked parent bundle")
        else:
            life.bind_candidate(root, spec, parent["caches"]["train"]["path"], parent["caches"]["val"]["path"], parent["parent_checkpoint"]["path"])
    return bundle


def fit_all(root, *, maximum_new_candidates=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    context = verify_context(root)
    if torch.cuda.is_initialized():
        raise ValueError("The CPU residual campaign must not initialize CUDA")
    if maximum_new_candidates is not None and (type(maximum_new_candidates) is not int or maximum_new_candidates < 1):
        raise ValueError("A bounded continuation requires a positive candidate count")
    with life.exclusive(Path(root)/"campaign_process"):
        bind_all(root)
        completed, new = [], 0
        for spec in context["candidate_plan"]:
            existing = (life.candidate_folder(root, spec)/"result.json").exists()
            if not existing and maximum_new_candidates is not None and new >= maximum_new_candidates:
                break
            completed.append(life.fit_candidate(root, spec))
            new += int(not existing)
            life.atomic_json(Path(root)/"progress.json", dict(updated_utc=life.utc(), observed_terminal=len(completed), total=90,
                valid=sum(r["status"] == "valid" for r in completed), failed_numerical=sum(r["status"] == "failed_numerical" for r in completed),
                missing_parent=sum(r["status"] == "missing_parent" for r in completed), last_id=spec["id"], scope=context["scope"]))
        return completed


def verify_selection(root):
    context = verify_context(root)
    bind_all(root)
    path = Path(root)/"selection_lock.json"
    if not path.exists():
        raise ValueError("Test access requires the completed common 45-choice selection")
    locked = life.read(path)
    # Recomputes all terminal candidates and validates an existing immutable lock.
    verified = life.lock_selection(root)
    if locked != verified or locked["scope"] != context["scope"]:
        raise ValueError("The common selection changed")
    return locked


def lock_selection(root):
    bind_all(root)
    life.lock_selection(root)
    return verify_selection(root)


def test_access(root):
    root = Path(root)
    locked = verify_selection(root)
    context = verify_context(root)
    body = dict(schema=1, scope=context["scope"], study_context=life.artifact(root/"study_context.json"),
                selection_lock=life.artifact(root/"selection_lock.json"), parent_bundle=context["parent_bundle"],
                selections=deepcopy(locked["choices"]), scientific_test_access=context["scope"] == RETAINED)
    path = root/"test_access.json"
    if path.exists():
        old = life.read(path)
        if {k:v for k,v in old.items() if k != "granted_utc"} != body:
            raise ValueError("An existing test-access binding changed")
        return old
    body["granted_utc"] = life.utc()
    life.atomic_json(path, body, immutable=True)
    return body


def validate_test_manifest(root, manifest_path):
    permit = test_access(root)
    context = verify_context(root)
    manifest = life.read(manifest_path)
    if manifest["schema"] != 1 or manifest["scope"] != context["scope"] or manifest["test_access"] != life.artifact(Path(root)/"test_access.json") or manifest["parent_bundle"] != context["parent_bundle"]:
        raise ValueError("Test caches must cite the actual prior common-selection access record")
    if manifest.get("native_cache_audit_passed") is not True or manifest["producer_sources"] != checked(context["parent_bundle"])["producer_sources"]:
        raise ValueError("Test cache production lacks its frozen native-equivalence evidence")
    for source in manifest["sources"]:
        life.verify_artifact(source)
    if not manifest["sources"]:
        raise ValueError("Native test-cache production must retain source evidence")
    parents = {(r["domain"],r["seed"]):r for r in checked(context["parent_bundle"])["parents"]}
    rows = manifest["parents"]
    by_key = {(r["domain"],r["seed"]):r for r in rows}
    if len(rows) != 15 or set(by_key) != set(REPLICATES):
        raise ValueError("Every prescribed parent requires a test-cache outcome")
    caches, common = {}, None
    for key in REPLICATES:
        parent, row = parents[key], by_key[key]
        if parent["status"] == "missing_parent":
            if row["status"] != "missing_parent" or row.get("cache") is not None or row["reason"] != parent["reason"]:
                raise ValueError("A missing parent cannot acquire a test substitute")
            continue
        if row["status"] != "available" or row["parent_checkpoint"] != parent["parent_checkpoint"] or row.get("native_prediction_and_bucket_equal") is not True:
            raise ValueError("Native test cache and selected parent disagree")
        cache = life.load_cache(row["cache"], "test", parent["parent_checkpoint"]["sha256"])
        train = life.load_cache(parent["caches"]["train"], "train", cache.parent_checkpoint_sha256)
        val = life.load_cache(parent["caches"]["val"], "val", cache.parent_checkpoint_sha256)
        if set(cache.ids) & (set(train.ids) | set(val.ids)) or (cache.target_mean, cache.target_sd, cache.z.dtype, cache.z.shape[1:]) != (train.target_mean, train.target_sd, train.z.dtype, train.z.shape[1:]):
            raise ValueError("Test records overlap fitting/validation or use a different native transformation")
        if context["scope"] == RETAINED and len(cache.ids) != (1024 if key[0] == "cubic" else 366):
            raise ValueError("Retained test count differs from the prescribed cohort")
        if key[0] == "ligand_contact":
            signature = (cache.ids, cache.truth.numpy().tobytes(), cache.target_mean, cache.target_sd)
            if common is not None and signature != common:
                raise ValueError("Molecular test caches must use one common cohort and truth")
            common = signature
        caches[key] = cache
    return manifest, parents, caches, permit


def prepare_test_caches(root, producer):
    # This gate precedes any invocation of the native test-data reader.
    test_access(root)
    context = verify_context(root)
    allowed = checked(context["parent_bundle"])["producer_sources"]
    try:
        source_path = inspect.getsourcefile(producer)
    except TypeError:
        source_path = None
    if source_path is None or life.artifact(source_path) not in allowed:
        raise ValueError("The test-cache producer was not frozen before residual fitting")
    path = producer(Path(root), life.artifact(Path(root)/"test_access.json"), context["parent_bundle"])
    validate_test_manifest(root, path)
    return path


def evaluate_one(root, domain, seed, procedure, parents, caches, choices):
    parent = parents[domain, seed]
    if parent["status"] == "missing_parent":
        return scoring.unavailable(domain, seed, procedure, "missing_parent", parent["reason"]), None
    cache = caches[domain, seed]
    if procedure == "baseline":
        return scoring.baseline(cache, domain, seed)
    choice = choices[domain, seed, procedure]
    if choice["selected_id"] is None:
        return scoring.unavailable(domain, seed, procedure, "no_valid_residual", "Both prescribed residual rates failed numerically", cache.parent_checkpoint_sha256), None
    spec = next(s for s in selection.candidate_plan() if s["id"] == choice["selected_id"])
    checkpoint = life.artifact(life.candidate_folder(root, spec)/"selected_checkpoint.pt")
    return scoring.residual(cache, spec, choice, checkpoint, cache.parent_checkpoint_sha256)


def score_all(root, manifest_path):
    root = Path(root)
    with life.exclusive(root/"evaluation_process"):
        _, parents, caches, permit = validate_test_manifest(root, manifest_path)
        choices = {(r["domain"],r["seed"],r["arm"]):r for r in permit["selections"]}
        dependencies = dict(test_access=life.artifact(root/"test_access.json"), test_manifest=life.artifact(manifest_path))
        outcomes, files = [], []
        for domain, seed in REPLICATES:
            for procedure in scoring.PROCEDURES:
                row, vectors = evaluate_one(root, domain, seed, procedure, parents, caches, choices)
                folder = root/"outcomes"/f"{domain}_{seed}_{procedure}"
                path = folder/"outcome.json"
                if path.exists():
                    old = life.read(path)
                    if old["dependencies"] != dependencies or old["outcome"] != row:
                        raise ValueError("A retained evaluation differs from the locked predictor and cache")
                    if vectors is None:
                        if old["vectors"] is not None:
                            raise ValueError("An unavailable outcome cannot carry prediction vectors")
                    else:
                        life.verify_artifact(old["vectors"])
                        if Path(old["vectors"]["path"]).resolve() != (folder/"predictions.pt").resolve() or not life.same(torch.load(old["vectors"]["path"], weights_only=True, map_location="cpu"), vectors):
                            raise ValueError("Saved test vectors do not reproduce from the selected checkpoint")
                else:
                    vector_record = None
                    if vectors is not None:
                        life.atomic_tensor(folder/"predictions.pt", vectors, immutable=True)
                        vector_record = life.artifact(folder/"predictions.pt")
                    life.atomic_json(path, dict(recorded_utc=life.utc(), scope=permit["scope"], dependencies=dependencies,
                                               outcome=row, vectors=vector_record), immutable=True)
                outcomes.append(row)
                files.append(life.artifact(path))
        summary = scoring.summarize(outcomes)
        verify_context(root)
        for source in dependencies.values():
            life.verify_artifact(source)
        for row in life.read(manifest_path)["parents"]:
            if row["status"] == "available":
                life.verify_artifact(row["cache"])
        record = dict(scope=permit["scope"], dependencies=dependencies, outcome_files=files, summary=summary,
                      complete=True, scientific_evaluation=permit["scientific_test_access"])
        destination = root/"evaluation.json"
        life.atomic_json(destination, record, immutable=True)
        return record
