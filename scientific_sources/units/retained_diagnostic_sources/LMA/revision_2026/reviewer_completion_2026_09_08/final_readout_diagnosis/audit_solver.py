"""Manufactured solver checks before scientific feature extraction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import linalg, optimize
from affine_solver import solve

HERE = Path(__file__).resolve().parent


def run():
    rng = np.random.default_rng(2026090901)
    a = np.column_stack([rng.normal(size=(37, 5)), np.ones(37)])
    truth = a @ np.arange(1., 7.) + .2*rng.normal(size=37)
    cases = []
    for name, design, y, beta0 in [
        ('full_rank_active', a, truth, np.ones(6)),
        ('full_rank_inactive', a, truth, np.full(6, 20.)),
        ('zero_radius', a, truth, np.zeros(6)),
        ('rank_deficient_active', np.column_stack([a, a[:, 0]]), truth, np.ones(7)),
        ('rank_deficient_inactive', np.column_stack([a, a[:, 0]]), truth, np.full(7, 20.)),
        ('zero_design', np.zeros_like(a), truth, np.ones(6)),
    ]:
        result = solve(design, y, beta0)
        m = result['metrics']
        assert m['numerical_checks_passed'], m
        other, _, rank, _ = linalg.lstsq(design, y, cond=max(design.shape)*np.finfo(float).eps, lapack_driver='gelsy')
        qr_mse = np.mean((y-design@other)**2)
        assert abs(qr_mse-m['ols_mse']) < 1e-10
        constrained_delta = None
        if m['radius_active']:
            def fun(b):
                return np.mean((y-design@b)**2)
            def jac(b):
                return 2*design.T@(design@b-y)/len(y)
            radius = m['native_coefficient_norm']
            alternate = optimize.minimize(fun, beta0, jac=jac, method='SLSQP',
                constraints=[dict(type='ineq', fun=lambda b: radius**2-b@b, jac=lambda b: -2*b)],
                options=dict(ftol=1e-12, maxiter=1000))
            assert alternate.success, alternate.message
            constrained_delta = abs(fun(alternate.x)-m['constrained_mse'])
            assert constrained_delta < 1e-9
        cases.append(dict(name=name, qr_rank=int(rank), qr_mse_delta=abs(qr_mse-m['ols_mse']),
                          independent_constrained_mse_delta=constrained_delta, **m))
    files = [HERE/'affine_solver.py', Path(__file__).resolve(), HERE/'protocol.md']
    receipt = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        manufactured_cases=cases, scientific_features_extracted=False,
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (HERE/'solver_audit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=True, manufactured_cases=len(cases))))


if __name__ == '__main__':
    run()
