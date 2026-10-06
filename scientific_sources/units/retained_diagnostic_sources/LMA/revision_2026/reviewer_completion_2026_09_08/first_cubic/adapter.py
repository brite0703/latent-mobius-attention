"""Fixed data/model interface and post-selection polynomial references."""
from pathlib import Path
import importlib.util
import itertools
import json
import math
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def local(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g = local("first_cubic_generator", "generator.py")
m = local("first_cubic_neural_models", "neural_models.py")
u = m.pm.import_file("first_cubic_utilities", HERE.parent/"synthetic_parity/parity_common.py")
TASKS, SEEDS, HEADS = ["first", "cubic"], list(range(100, 110)), m.HEADS
KIND, LRS = "regression", [.0003, .001]
EPOCHS, BATCH, PATIENCE, CLIP = 100, 256, 8, 1.
ORDER_SEED = 2026090807
SIZES = m.SIZES
EVALUATION_SPLITS = ["test", "population"]
PAIRS = [("lma2", "lma1"), ("lma3", "lma1"), ("lma3", "lma2"),
         ("lma2", "additive2"), ("lma3", "additive3"),
         ("deepsets_ln", "deepsets_plain"), ("deepsets_wide", "deepsets_plain")]


def data_path(seed):
    return HERE/"data"/f"seed{seed}.npz"


def load(spec, split, device="cpu"):
    assert split in ["train", "val", "test", "population"]
    with np.load(data_path(spec["seed"])) as z:
        ids = np.arange(4096) if split == "population" else z[split+"_ids"].copy()
        mask = torch.as_tensor(z["bits"][ids].copy(), dtype=torch.bool, device=device)
        truth = z[spec["task"]+"_target"][ids].copy()
    x = torch.eye(12, dtype=torch.float32, device=device).expand(len(ids), -1, -1)
    return dict(inputs=(x, mask), y=torch.as_tensor(truth, dtype=torch.float32, device=device),
                truth=truth, ids=ids)


def build(spec):
    return m.build(spec["seed"], spec["head"])


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), HERE/"generator.py", HERE/"neural_audit.py"]+m.sources()))


def lock_files():
    files = [HERE/name for name in ["protocol.md", "neural_implementation.md", "generator_audit.json", "data_manifest.json",
        "closed_form.py", "closed_form_prefit_audit.json", "closed_form_lock.json", "closed_form_selection.json"]]
    files += [data_path(seed) for seed in SEEDS]
    for name in ["closed_form_candidates", "closed_form_coefficients"]:
        files += sorted((HERE/name).glob("*"))
    return files


def prepare():
    assert json.loads((HERE/"generator_audit.json").read_text())["passed"]
    folder = HERE/"data"
    folder.mkdir(exist_ok=True)
    bits = g.population()
    targets = {task+"_target":g.target_numerators(bits, task)/math.sqrt(8) for task in TASKS}
    rows = []
    for seed in SEEDS:
        splits = g.split(seed)
        expected = dict(bits=bits, **targets, **{key+"_ids":value for key, value in splits.items()})
        path = data_path(seed)
        if not path.exists():
            np.savez_compressed(path, **expected)
        with np.load(path) as z:
            for key, value in expected.items():
                np.testing.assert_array_equal(z[key], value)
        assert len(set(np.concatenate(list(splits.values())))) == 4096
        rows.append(dict(seed=seed, sha256=u.sha(path), file=str(path.relative_to(HERE)),
            sizes={key:len(value) for key, value in splits.items()},
            empty_set_split=next(key for key, value in splits.items() if 0 in value)))
    u.write_json(HERE/"data_manifest.json", dict(passed=True, prepared_utc=u.now(), datasets=rows,
        encoding="Twelve typed one-hot element vectors, arbitrary-position presence mask; padding values are ignored",
        targets="Float64 numerator/sqrt(8) for validation/reporting; float32 cast only for training loss"))


def diagnostics(model, data):
    if not hasattr(model.head, "factor"):
        return None
    totals = dict(product_entries=0, finite_entries=0, exact_zero_products=0,
                  tanh_derivative_below_1e_6=0, tanh_exactly_abs_one=0, exact_zero_factors=0)
    maximum = 0.
    model.eval()
    with torch.no_grad():
        x, mask = data["inputs"]
        for start in range(0, len(mask), BATCH):
            mm = mask[start:start+BATCH]
            h = model.encoder(x[start:start+BATCH])
            f = model.head.factor(h).masked_fill(~mm.unsqueeze(-1), 1.)
            p = f.double().prod(1)
            bounded = p.tanh()
            finite = torch.isfinite(p)
            totals["product_entries"] += p.numel()
            totals["finite_entries"] += int(finite.sum())
            totals["exact_zero_products"] += int((p == 0).sum())
            totals["tanh_derivative_below_1e_6"] += int(((1-bounded.square()) < 1e-6).sum())
            totals["tanh_exactly_abs_one"] += int((bounded.abs() == 1).sum())
            totals["exact_zero_factors"] += int((f == 0).sum())
            if bool(finite.any()):
                maximum = max(maximum, float(p[finite].abs().max()))
    return dict(**totals, maximum_absolute_product=maximum)


def polynomial_features(bits, degree):
    # Independently reconstruct the declared Walsh design without calling the fitted-control module.
    signs = np.where(bits == 1, 1., -1.)
    supports = [s for k in range(1, degree+1) for s in itertools.combinations(range(12), k)]
    return np.column_stack([np.prod(signs[:, s], axis=1) for s in supports])


def evaluate_references():
    choices = json.loads((HERE/"closed_form_selection.json").read_text())
    assert choices["lock_sha256"] == u.sha(HERE/"closed_form_lock.json")
    assert len(json.loads((HERE/"selection_lock.json").read_text())["selections"]) == 200
    for item in choices["candidates"]:
        assert u.sha(HERE/item["path"]) == item["sha256"]
    designs = {degree:polynomial_features(g.population(), degree) for degree in (1,3)}
    out = []
    for choice in choices["selections"]:
        candidate = json.loads((HERE/"closed_form_candidates"/(choice["selected_id"]+".json")).read_text())
        path = HERE/candidate["coefficients"]
        assert u.sha(path) == choice["coefficients_sha256"]
        with np.load(path) as z:
            prediction = designs[choice["degree"]]@z["coefficient"]+z["intercept"]
        truth = g.target_numerators(g.population(), choice["task"])/math.sqrt(8)
        ids = g.split(choice["seed"])["test"]
        record = dict(**choice, head=f'walsh_degree{choice["degree"]}', selected_lambda=candidate["lambda_value"],
            parameters=candidate["parameters"], test=dict(mse=math.fsum((prediction[ids]-truth[ids])**2)/len(ids)),
            population=dict(mse=math.fsum((prediction-truth)**2)/len(truth)))
        dest = HERE/"test_predictions"/(choice["selected_id"]+"_population.npz")
        np.savez_compressed(dest, ids=np.arange(4096), test_ids=ids, truth=truth, prediction=prediction)
        record.update(prediction_file=str(dest.relative_to(HERE)), prediction_sha256=u.sha(dest))
        out.append(record)
    return out


def audit_references(records):
    assert len(records) == 40
    for row in records:
        path = HERE/row["prediction_file"]
        assert u.sha(path) == row["prediction_sha256"]
        group = [json.loads((HERE/"closed_form_candidates"/(cid+".json")).read_text()) for cid in row["candidate_ids"]]
        chosen = min(group, key=lambda r:(r["validation_mse"], r["lambda_value"]))
        assert chosen["id"] == row["selected_id"]
        with np.load(HERE/chosen["coefficients"]) as coefficient, np.load(path) as saved:
            restored = polynomial_features(g.population(), row["degree"])@coefficient["coefficient"]+coefficient["intercept"]
            np.testing.assert_allclose(restored, saved["prediction"], atol=1e-12, rtol=1e-12)
            for split, ids in [("test", saved["test_ids"]), ("population", np.arange(4096))]:
                mse = math.fsum((restored[ids]-saved["truth"][ids])**2)/len(ids)
                assert abs(mse-row[split]["mse"]) < 1e-12
