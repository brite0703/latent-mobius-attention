"""Fit declared polynomial controls; keep test scoring behind the neural lock."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
import generator as g

HERE = g.HERE
LAMBDAS = [0., .001, .1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def features(bits, degree):
    w = 2*np.asarray(bits, dtype=np.float64)-1
    supports = [support for d in range(1, degree+1) for support in itertools.combinations(range(12), d)]
    x = np.column_stack([np.prod(w[:, support], axis=1) for support in supports])
    return x, supports


def audit():
    records = []
    for degree in (1, 3):
        x, supports = features(g.population(), degree)
        np.testing.assert_array_equal(x.sum(0), np.zeros(x.shape[1]))
        np.testing.assert_array_equal(x.T@x, 4096*np.eye(x.shape[1]))
        assert x.shape[1] == (12 if degree == 1 else 298)
        for seed in range(100, 110):
            ids = g.split(seed)
            assert not set(ids["train"]) & set(ids["val"])
            assert not set(ids["train"]) & set(ids["test"])
        records.append(dict(degree=degree, slope_features=len(supports), intercept=1,
                            exact_population_orthogonality=True))
    write(HERE/"closed_form_prefit_audit.json", dict(passed=True, records=records,
        sources=[dict(path=str(p), sha256=sha(p)) for p in (Path(__file__), HERE/"generator.py")]))


def fit():
    assert not (HERE/"closed_form_selection.json").exists()
    audit()
    sources = [Path(__file__), HERE/"generator.py", HERE/"protocol.md", HERE/"generator_audit.json",
               HERE/"closed_form_prefit_audit.json"]
    if (HERE/"closed_form_lock.json").exists():
        lock = json.loads((HERE/"closed_form_lock.json").read_text(encoding="utf-8"))
        assert all(sha(item["path"]) == item["sha256"] for item in lock["sources"])
    else:
        lock = dict(locked_utc=datetime.now(timezone.utc).isoformat(), candidates=120, selections=40,
            degrees=[1,3], lambdas=LAMBDAS, seeds=list(range(100,110)),
            objective="mean training squared error plus lambda times squared slope norm; unpenalized intercept",
            solver="lambda0 scipy.linalg.lstsq(gelsd), otherwise symmetric positive-definite normal equations",
            selection="minimum validation MSE, lower lambda for an exact tie",
            test_status="No test scoring until all 200 declared neural choices are locked",
            sources=[dict(path=str(p), sha256=sha(p)) for p in sources])
        write(HERE/"closed_form_lock.json", lock)
    destination = HERE/"closed_form_candidates"
    destination.mkdir(exist_ok=True)
    models = HERE/"closed_form_coefficients"
    models.mkdir(exist_ok=True)
    rows, choices = [], []
    bits = g.population()
    for degree in (1,3):
        x, supports = features(bits, degree)
        for seed in range(100,110):
            ids = g.split(seed)
            train, val = x[ids["train"]], x[ids["val"]]
            mean_x = train.mean(0)
            centered = train-mean_x
            gram = centered.T@centered
            for task in ("first", "cubic"):
                target = g.target_numerators(bits, task)/math.sqrt(8)
                y = target[ids["train"]]
                mean_y = y.mean()
                rhs = centered.T@(y-mean_y)
                group = []
                for j, penalty in enumerate(LAMBDAS):
                    key = f"{task}_degree{degree}_seed{seed}_lambda{j}"
                    if penalty == 0:
                        coef, residual, rank, spectrum = linalg.lstsq(centered, y-mean_y, lapack_driver="gelsd")
                        assert rank == len(supports)
                    else:
                        coef = linalg.solve(gram+len(train)*penalty*np.eye(len(supports)), rhs, assume_a="pos")
                    intercept = mean_y-mean_x@coef
                    prediction = val@coef+intercept
                    mse = math.fsum((prediction-target[ids["val"]])**2)/len(val)
                    assert np.isfinite(coef).all() and math.isfinite(mse)
                    normal_residual = float(np.linalg.norm(centered.T@(centered@coef-(y-mean_y))/len(train)+penalty*coef))
                    assert normal_residual < 1e-10
                    path = models/(key+".npz")
                    np.savez_compressed(path, coefficient=coef, intercept=intercept, train_ids=ids["train"],
                                        validation_ids=ids["val"], validation_prediction=prediction,
                                        validation_truth=target[ids["val"]])
                    row = dict(id=key, degree=degree, task=task, seed=seed, lambda_value=penalty,
                        status="valid", validation_mse=mse, parameters=len(coef)+1,
                        training_objective=float(np.mean((train@coef+intercept-y)**2)+penalty*(coef@coef)),
                        normal_equation_residual=normal_residual, coefficients=str(path.relative_to(HERE)),
                        coefficients_sha256=sha(path), lock_sha256=sha(HERE/"closed_form_lock.json"))
                    write(destination/(key+".json"), row)
                    group.append(row)
                    rows.append(row)
                chosen = min(group, key=lambda r:(r["validation_mse"],r["lambda_value"]))
                choices.append(dict(degree=degree, task=task, seed=seed, selected_id=chosen["id"],
                    validation_mse=chosen["validation_mse"], candidate_ids=[r["id"] for r in group],
                    coefficients_sha256=chosen["coefficients_sha256"]))
    assert len(rows)==120 and len(choices)==40
    write(HERE/"closed_form_selection.json", dict(locked_utc=datetime.now(timezone.utc).isoformat(),
        selections=choices, candidate_count=len(rows), selection_count=len(choices),
        candidates=[dict(path=str((destination/(r["id"]+".json")).relative_to(HERE)),
                         sha256=sha(destination/(r["id"]+".json"))) for r in rows],
        lock_sha256=sha(HERE/"closed_form_lock.json"), test_scored=False,
        largest_normal_equation_residual=max(r["normal_equation_residual"] for r in rows)))
    print(json.dumps(dict(stage="polynomial_controls_selected", valid_candidates=len(rows), choices=len(choices),
                          test_scored=False, largest_normal_equation_residual=max(r["normal_equation_residual"] for r in rows))))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        fit()
