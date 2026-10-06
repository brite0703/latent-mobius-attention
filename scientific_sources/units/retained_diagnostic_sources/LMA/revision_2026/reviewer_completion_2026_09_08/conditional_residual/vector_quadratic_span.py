"""Exact algebra on saved routing/value dictionaries; no data or fit access."""
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def transpose(a):
    return [list(x) for x in zip(*a)]


def multiply(a, b):
    bt = transpose(b)
    return [[sum((u*v for u, v in zip(row, col)), F(0)) for col in bt] for row in a]


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def inverse_or_rank(a):
    n = len(a)
    rows = [list(a[i])+identity(n)[i] for i in range(n)]
    rank = 0
    for column in range(n):
        pivot = next((i for i in range(rank, n) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [x/pivot_value for x in rows[rank]]
        for i in range(n):
            if i != rank:
                factor = rows[i][column]
                rows[i] = [x-factor*y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank, [row[n:] for row in rows] if rank == n else None


def fractions(a):
    return [[F(float(x)) for x in row] for row in a]


def encoded(a):
    return [[[str(x.numerator), str(x.denominator)] for x in row] for row in a]


def save_new(path, content):
    text = json.dumps(content, indent=2, allow_nan=False)+'\n'
    if path.exists():
        raise FileExistsError('Preserve prior diagnostic: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    source_report = HERE/'cubic_linear_access.json'
    prior = json.loads(source_report.read_text(encoding='utf-8'))
    assert prior['completed']
    bound = {Path(x['path']).resolve(): x for x in prior['artifacts']}
    sources = [artifact(Path(__file__)), artifact(HERE/'vector_quadratic_span.md'), artifact(source_report)]
    results, artifacts = [], []
    for seed in range(100, 110):
        path = HERE/f'linear_access_certificates/seed{seed}.npz'
        assert path.resolve() in bound and sha(path) == bound[path.resolve()]['sha256']
        sources.append(artifact(path))
        with np.load(path, allow_pickle=False) as z:
            p_float, v_float = z['routing'].copy(), z['values'].copy()
        assert p_float.shape == (12, 8) and v_float.shape == (12, 12)
        assert np.isfinite(p_float).all() and np.isfinite(v_float).all()
        p, v = fractions(p_float), fractions(v_float)
        n, m = len(p), len(p[0])
        s = [sum(row, F(0)) for row in p]
        c = [[s[i]*s[h]-sum((p[i][j]*p[h][j] for j in range(m)), F(0))
              for h in range(n)] for i in range(n)]
        cross = [[sum((p[i][j]*p[h][l]+p[i][l]*p[h][j]
                       for j in range(m) for l in range(j+1, m)), F(0))
                  for h in range(n)] for i in range(n)]
        assert c == cross
        positive_p = all(x > 0 for row in p for x in row)
        positive_c = all(x > 0 for row in c for x in row)
        rank, inverse = inverse_or_rank(v)
        row = dict(seed=seed, n=n, d=n, buckets=m, exact_value_rank=rank,
                   routing_strictly_positive=positive_p, C_strictly_positive=positive_c,
                   minimum_routing_entry=float(min(x for line in p for x in line)),
                   minimum_C_entry=float(min(x for line in c for x in line)),
                   maximum_stored_row_sum_difference_from_one=float(max(abs(x-1) for x in s)),
                   exact_C_entry_checks=n*n, numerical_value_condition_number=float(np.linalg.cond(v_float.astype(np.float64))),
                   relaxed_quadratic_realization_condition_met=rank == n and positive_c)
        if row['relaxed_quadratic_realization_condition_met']:
            assert multiply(v, inverse) == identity(n)
            assert multiply(inverse, v) == identity(n)
            # Fixed dense matrix, unrelated to any labels or fitted outcomes.
            h = [[F((-1)**(i+j)*(1+(i+j) % 5)) for j in range(n)] for i in range(n)]
            assert h == transpose(h)
            w = multiply(multiply(inverse, h), transpose(inverse))
            assert w == transpose(w)
            recovered_h = multiply(multiply(v, w), transpose(v))
            assert recovered_h == h
            q = [[c[i][j]*h[i][j]/2 for j in range(n)] for i in range(n)]
            recovered_q = [[cross[i][j]*recovered_h[i][j]/2 for j in range(n)] for i in range(n)]
            assert recovered_q == q
            certificate_path = HERE/f'vector_quadratic_span_certificates/seed{seed}.json'
            save_new(certificate_path, dict(seed=seed, source=artifact(path), inverse_rationals=encoded(inverse),
                                           exact_left_and_right_inverse=True, dense_shared_quadratic_coefficient_reconstruction=True))
            artifacts.append(artifact(certificate_path))
            row.update(exact_inverse_checked_both_sides=True, exact_dense_quadratic_coefficient_reconstruction=True)
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(completed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Exact arithmetic interpretation of preserved finite dictionaries; relaxed linear-after-product decoder only',
                  parents=len(results), full_row_rank_parents=sum(x['exact_value_rank'] == 12 for x in results),
                  sufficient_condition_parents=sum(x['relaxed_quadratic_realization_condition_met'] for x in results),
                  exact_C_entry_checks=sum(x['exact_C_entry_checks'] for x in results), results=results,
                  new_theorem_claim=False, native_layernorm_gelu_realization_proved=False,
                  new_predictive_result=False, new_test_predictions=False, residual_fits=0,
                  target_or_split_arrays_loaded=False, sources=sources, artifacts=artifacts,
                  limitations=['No coefficient budget or stability guarantee', 'Not a realization proof for the actual LayerNorm/GELU residual',
                               'No molecular conclusion', 'No quadratic decoder can directly represent the pure cubic population target'])
    save_new(HERE/'vector_quadratic_span.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','sources','artifacts')}), flush=True)


if __name__ == '__main__':
    main()
