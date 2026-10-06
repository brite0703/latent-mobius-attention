"""Numerical fitting diagnostics in fixed, unscaled feature coordinates."""
import numpy as np


def mse(a, y, beta):
    return float(np.mean(np.square(y - a @ beta)))


def solve(a, y, beta0):
    a, y, beta0 = [np.asarray(v, dtype=np.float64) for v in (a, y, beta0)]
    if a.ndim != 2 or y.shape != (len(a),) or beta0.shape != (a.shape[1],):
        raise ValueError("Incompatible affine diagnostic arrays")
    if len(a) < a.shape[1] or not all(np.isfinite(v).all() for v in (a, y, beta0)):
        raise ValueError("Require finite overdetermined fitting arrays")
    n, p = a.shape
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    cutoff = max(n, p) * np.finfo(np.float64).eps * s[0]
    keep = s > cutoff
    uy = u.T @ y
    coordinates = np.zeros_like(s)
    coordinates[keep] = uy[keep] / s[keep]
    ols = vt.T @ coordinates
    radius = float(np.linalg.norm(beta0))
    zero_radius = radius == 0.
    iterations, multiplier = 0, 0.
    if zero_radius:
        constrained = np.zeros_like(beta0)
        multiplier = None  # The singleton feasible set needs no finite multiplier.
    elif np.linalg.norm(ols) <= radius:
        constrained = ols.copy()
    else:
        def at(lam):
            return vt.T @ (s * uy / (s * s + n * lam))
        low, high = 0., max(s[0] * s[0] / n, np.finfo(float).tiny)
        while np.linalg.norm(at(high)) > radius:
            high *= 2.
            if not np.isfinite(high):
                raise ArithmeticError("Unable to bracket a finite feasible multiplier")
        for iterations in range(1, 301):
            middle = low + (high - low) / 2.
            if np.linalg.norm(at(middle)) > radius:
                low = middle
            else:
                high = middle
            if high - low <= 1e-12 * high:
                break
        else:
            raise ArithmeticError("Multiplier bisection did not converge")
        multiplier, constrained = float(high), at(high)
    q0, qo, qc = (mse(a, y, v) for v in (beta0, ols, constrained))
    d = a.T @ y / n
    normal = a.T @ (a @ ols - y) / n
    stationarity = None if zero_radius else a.T @ (a @ constrained - y) / n + multiplier * constrained
    dual = None if zero_radius else float(y @ y / n - d @ constrained - multiplier * radius**2)
    scale = 1. + float(np.linalg.norm(d))
    numerical = dict(
        ols_normal_residual=float(np.linalg.norm(normal)),
        constrained_stationarity=None if zero_radius else float(np.linalg.norm(stationarity)),
        constrained_dual_value=dual,
        primal_minus_dual=None if zero_radius else qc-dual,
        feasibility_excess=float(np.linalg.norm(constrained))-radius,
        complementarity=None if zero_radius else float(multiplier*(constrained@constrained-radius**2)),
        ols_pythagorean_discrepancy=float(q0-qo-np.mean((a@(beta0-ols))**2)),
    )
    passed = (qo <= qc + 1e-9 and qc <= q0 + 1e-9
              and numerical['ols_normal_residual'] <= 1e-9 * scale
              and numerical['feasibility_excess'] <= 1e-12 * (1.+radius)
              and (zero_radius or (numerical['constrained_stationarity'] <= 1e-9 * scale
                                   and abs(numerical['primal_minus_dual']) <= 1e-9*(1.+qc))))
    return dict(beta_ols=ols, beta_constrained=constrained,
        metrics=dict(native_affine_mse=q0, ols_mse=qo, constrained_mse=qc,
            native_coefficient_norm=radius, ols_coefficient_norm=float(np.linalg.norm(ols)),
            constrained_coefficient_norm=float(np.linalg.norm(constrained)),
            radius_active=not zero_radius and multiplier > 0., zero_radius=zero_radius,
            multiplier=multiplier, bisection_iterations=iterations,
            singular_values=s.tolist(), discarded_singular_indices=np.flatnonzero(~keep).tolist(),
            svd_cutoff=float(cutoff), numerical_rank=int(keep.sum()),
            n=n, p=p, numerical_checks_passed=bool(passed), **numerical))
