"""Exact algebra and full-population checks for the path border construction."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

ROOT = Path(__file__).parent


def router(n, t, symbolic=False):
    half = sp.Rational(1, 2) if symbolic else .5
    rows = [[0, 1, 0, 0, 0], [0, 0, half, half, half],
            [half, 0, half, t, 0], [half, 0, 0, half-t, half]]
    if symbolic:
        return sp.Matrix(rows)
    r = np.zeros((n-1, n))
    r[:4, :5] = rows
    for vertex in range(5, n):
        r[vertex-1, vertex] = 1.
    return r


def coefficient_map(r, symbolic=False):
    m, n = r.shape
    pairs = list(itertools.combinations(range(n), 2))
    slots = list(itertools.combinations(range(m), 2))
    rows = [[r[j, a]*r[l, b]+r[j, b]*r[l, a] for j, l in slots] for a, b in pairs]
    return (sp.Matrix(rows) if symbolic else np.array(rows)), pairs, slots


def decoders(n, t, symbolic=False):
    slots = list(itertools.combinations(range(n-1), 2))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    if n > 5:
        edges += [(0, 5)] + [(v, v+1) for v in range(5, n-1)]
    theta = sp.zeros(len(slots), len(edges)) if symbolic else np.zeros((len(slots), len(edges)))
    entries = [((0, 1), 0, -1), ((0, 2), 0, 1), ((0, 3), 0, 1),
               ((0, 1), 1, 1), ((0, 2), 1, 1), ((0, 3), 1, -1),
               ((1, 2), 2, 1/t), ((2, 3), 2, -1/t),
               ((1, 3), 3, 2/(1-2*t)), ((2, 3), 3, -2/(1-2*t))]
    if n > 5:
        entries += [((1, 4), 4, -1), ((2, 4), 4, 1), ((3, 4), 4, 1)]
        for index, (a, b) in enumerate(edges[5:], 5):
            entries.append(((a-1, b-1), index, 1))
    for slot, edge_index, value in entries:
        theta[slots.index(slot), edge_index] = value
    return theta, edges


def symbolic_check():
    t = sp.symbols('t', positive=True)
    r = router(5, t, True)
    b, pairs, slots = coefficient_map(r, True)
    theta, edges = decoders(5, t, True)
    target = sp.zeros(len(pairs), len(edges))
    for j, edge in enumerate(edges):
        target[pairs.index(edge), j] = 1
    residual = (b*theta-target).applyfunc(sp.factor)
    expected = sp.zeros(*residual.shape)
    expected[pairs.index((1, 3)), 1] = 2*t
    assert residual == expected
    assert r.T*sp.ones(4, 1) == sp.ones(5, 1)
    assert b.rank() == 6
    first_indices = [pairs.index(pair) for pair in [(0,1),(1,2),(1,3),(1,4)]]
    first = b.extract(first_indices, [0,1,2])
    normal = sp.Matrix([0, -2*t, 1, 2*t-1])
    assert first.T*normal == sp.zeros(3, 1)
    assert first.rank() == 3
    exact_risk = sp.factor((normal[1]**2/(normal.dot(normal)))/4)
    assert sp.simplify(exact_risk-t**2/(2*(4*t**2-2*t+1))) == 0
    return dict(status='passed', symbolic_rank=6, symbolic_residual='2t on pair (2,4) for target (2,3), zero otherwise',
                exact_family_average_risk=str(exact_risk))


def main():
    algebra = symbolic_check()
    records = []
    for n in [5, 8, 12]:
        cube = np.array(list(itertools.product((-1., 1.), repeat=n)))
        for t in [.2, .1, .03, .01, .003, .001, .0003]:
            base = router(n, t)
            theta, edges = decoders(n, t)
            target_values = np.stack([cube[:,a]*cube[:,b] for a,b in edges], axis=1)
            for positive in (False, True):
                r = (1-t**3)*base+t**3/(n-1) if positive else base
                mat, pairs, slots = coefficient_map(r)
                target = np.eye(len(pairs))[:, [pairs.index(edge) for edge in edges]]
                residual = mat@theta-target
                coefficient_mse = float(np.sum(residual**2)/len(edges))
                z = cube@r.T
                phi = np.stack([z[:,j]*z[:,l] for j,l in slots], axis=1)
                phi -= phi.mean(0)
                cube_mse = float(np.mean(np.sum((phi@theta-target_values)**2, axis=1))/len(edges))
                discrepancy = abs(coefficient_mse-cube_mse)
                assert discrepancy < 1e-9
                assert np.max(abs(r.sum(0)-1)) < 1e-12
                if positive:
                    assert r.min() > 0
                    upper = (2*t/np.sqrt(n-1)+4*np.sqrt(2*len(pairs)*len(slots))*t*t)**2
                    assert coefficient_mse <= upper+1e-12
                else:
                    upper = 4*t*t/(n-1)
                    assert abs(coefficient_mse-upper) < 1e-12
                budget = np.sqrt(2)/t
                max_norm = float(np.max(np.linalg.norm(theta, axis=0)))
                assert max_norm <= budget+1e-9
                row = dict(N=n,M=n-1,t=t,strictly_positive=positive,
                           declared_coefficient_budget=budget,maximum_coefficient_norm=max_norm,
                           feasible_average_mse=coefficient_mse,full_cube_average_mse=cube_mse,
                           discrepancy=discrepancy,proved_upper_bound=float(upper))
                if n == 5 and not positive:
                    optimum_coefficients = np.linalg.lstsq(mat, target, rcond=1e-12)[0]
                    oracle = float(np.sum((mat@optimum_coefficients-target)**2)/4)
                    formula = t*t/(2*(4*t*t-2*t+1))
                    assert abs(oracle-formula) < 1e-12
                    assert np.max(abs(optimum_coefficients[:,2]-theta[:,2])) < 1e-7
                    row.update(exact_fixed_router_optimum=formula,independent_svd_optimum=oracle)
                records.append(row)
    output = ROOT/'output'
    output.mkdir(exist_ok=True)
    report = dict(interpretation='Exact scalar construction, not trained LMA; bounds are feasible upper bounds on optimized routing error.',
                  symbolic_verification=algebra, population_checks=len(records),
                  max_full_cube_discrepancy=max(row['discrepancy'] for row in records), records=records)
    (output/'border_path_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(7,4.4), layout='constrained')
    for positive, label, style in [(False,'Closed simplex: explicit decoder','o-'),
                                   (True,'Strictly positive: same decoder','s--')]:
        data = [row for row in records if row['N']==5 and row['strictly_positive']==positive]
        ax.loglog([row['declared_coefficient_budget'] for row in data],
                  [row['feasible_average_mse'] for row in data],style,label=label)
    data = [row for row in records if row['N']==5 and not row['strictly_positive']]
    ax.loglog([row['declared_coefficient_budget'] for row in data],
              [row['exact_fixed_router_optimum'] for row in data],'^-',label='Closed simplex: optimal readout for this router')
    ax.set(xlabel=r'Coefficient budget $\Lambda=\sqrt{2}/t$',ylabel='Average population MSE across four edges',
           title='Five-vertex path, four scalar buckets')
    ax.grid(alpha=.2,which='both')
    ax.legend(frameon=False,fontsize=9)
    fig.savefig(output/'border_path_construction.png',dpi=200)
    fig.savefig(output/'border_path_construction.svg')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
