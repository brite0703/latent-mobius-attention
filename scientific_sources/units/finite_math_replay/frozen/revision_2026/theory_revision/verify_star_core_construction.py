"""Independent symbolic and full-population audit of the eighth-review construction."""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

from verify_border_path import coefficient_map

ROOT = Path(__file__).parent


def core_construction(m, t, symbolic=False):
    if m < 4:
        raise ValueError('At least four leaves are required.')
    zero = sp.zeros if symbolic else lambda a, b: np.zeros((a, b))
    half = sp.Rational(1, 2) if symbolic else .5
    third = sp.Rational(1, 3) if symbolic else 1 / 3
    r = zero(m, m + 1)
    for j in range(3):
        r[j, 0] = third
        r[j, j + 1] = t
        r[m - 1, j + 1] = 1 - t
    for j in range(3, m):
        r[j, j + 1] = 1
    slots = list(itertools.combinations(range(m), 2))
    lookup = {pair: i for i, pair in enumerate(slots)}
    theta = zero(len(slots), m)

    def add(a, b, target, value):
        theta[lookup[tuple(sorted((a, b)))], target] += value

    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        add(i, j, i, 3 * half / t)
        add(i, k, i, 3 * half / t)
        add(j, k, i, -3 * half / t)
        add(i, m - 1, i, -1)
        add(j, m - 1, i, half)
        add(k, m - 1, i, half)
    for j in range(3):
        add(j, m - 1, m - 1, 1)
    for a, b in itertools.combinations(range(3), 2):
        add(a, b, m - 1, -3 * half * (1 - t) / t)
    adjustment = -3 * t * (1 - t) / (1 + 3 * (1 - t) ** 2)
    for j in range(3, m - 1):
        for a in range(3):
            add(a, j, j, 1)
        add(j, m - 1, j, adjustment)
    return r, theta


def exact_risk(m, t):
    return t*t/m * (15 + 7.5*t + 5.25*t*t + 3*(m-4)/(1+3*(1-t)**2))


def budget_scale(budget):
    if budget < 6:
        raise ValueError('The stated budget construction requires Lambda >= 6.')
    return np.sqrt(27 / (4 * (budget**2 - 1.5)))


def main():
    t = sp.symbols('t', positive=True)
    symbolic = []
    for m in range(4, 10):
        r, theta = core_construction(m, t, True)
        b, pairs, _ = coefficient_map(r, True)
        target = sp.eye(len(pairs))[:, [pairs.index((0, i)) for i in range(1, m+1)]]
        residual = (b*theta-target).applyfunc(sp.factor)
        assert residual[:m, :] == sp.zeros(m, m)
        assert sp.factor(r[:, 1:].det()) == t**3
        assert r.T*sp.ones(m, 1) == sp.ones(m+1, 1)
        individual = [sp.factor(sum(x*x for x in residual[:, i])) for i in range(m)]
        norms = [sp.factor(theta[:, i].dot(theta[:, i])) for i in range(m)]
        for i in range(3):
            assert sp.simplify(individual[i]-sp.Rational(3, 4)*t*t*(5+4*t+2*t*t)) == 0
            assert sp.simplify(norms[i]-sp.Rational(27, 4)/t**2-sp.Rational(3, 2)) == 0
        assert sp.simplify(individual[-1]-sp.Rational(3, 4)*t*t*(5-2*t+t*t)) == 0
        assert sp.simplify(norms[-1]-3-sp.Rational(27, 4)*(1-t)**2/t**2) == 0
        for i in range(3, m-1):
            assert sp.simplify(individual[i]-3*t*t/(1+3*(1-t)**2)) == 0
        average = sp.factor(sum(individual)/m)
        leading = sp.limit(average/t**2, t, 0)*sp.Rational(27, 4)
        assert sp.simplify(leading-sp.Rational(81, 16)-sp.Rational(81, m)) == 0
        symbolic.append(dict(leaves=m, exact_risk=str(average), leading_upper_constant=str(leading)))
    population = []
    for m in [4, 5, 6, 8, 12]:
        x = np.array(list(itertools.product((-1., 1.), repeat=m+1)))
        truth = x[:, [0]]*x[:, 1:]
        for budget in [6., 16., 64., 256., 1024.]:
            scale = budget_scale(budget)
            r0, theta = core_construction(m, scale)
            for positive in [False, True]:
                r = (1-scale**3)*r0+scale**3/m if positive else r0
                b, pairs, slots = coefficient_map(r)
                target = np.eye(len(pairs))[:, :m]
                risk = float(np.mean(np.sum((b@theta-target)**2, axis=0)))
                z = x@r.T
                phi = np.stack([z[:, j]*z[:, k] for j, k in slots], axis=1)
                phi -= phi.mean(0)
                full = float(np.mean((phi@theta-truth)**2))
                assert abs(risk-full) < 1e-11
                assert np.linalg.norm(theta, axis=0).max() <= budget*(1+1e-12)
                if not positive:
                    assert abs(risk-exact_risk(m, scale)) < 1e-11
                population.append(dict(leaves=m, budget=budget, positive=positive,
                                       coefficient_mse=risk, cube_mse=full, discrepancy=abs(risk-full),
                                       scaled_mse=budget**2*risk))
    report = dict(symbolic=symbolic, population=population, population_cases=len(population),
                  max_discrepancy=max(row['discrepancy'] for row in population),
                  status='passed; improved feasible upper constant, not a sharp constant')
    (ROOT/'output'/'star_core_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['symbolic', 'population']}))


if __name__ == '__main__':
    main()
