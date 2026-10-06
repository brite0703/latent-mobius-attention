"""Exact coefficient and noise identities; no fitting, test scoring or sampling."""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
COMP = HERE.parent.parent
REV = COMP.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core(m, t):
    r = [[F(0) for _ in range(m + 1)] for _ in range(m)]
    for j in range(3):
        r[j][0] = F(1, 3)
        r[j][j + 1] = t
        r[m - 1][j + 1] = 1 - t
    r[m - 1][m] = 1
    for j in range(3, m - 1):
        r[j][j + 1] = 1
    theta = [defaultdict(F) for _ in range(m)]

    def add(i, j, k, coefficient):
        theta[i][tuple(sorted((j, k)))] += coefficient

    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        for a, b, coefficient in (
            (i, j, F(3, 2) / t), (i, k, F(3, 2) / t),
            (j, k, -F(3, 2) / t), (i, m - 1, -F(1)),
            (j, m - 1, F(1, 2)), (k, m - 1, F(1, 2)),
        ):
            add(i, a, b, coefficient)
    for j in range(3):
        add(m - 1, j, m - 1, F(1))
    for j, k in ((0, 1), (0, 2), (1, 2)):
        add(m - 1, j, k, -F(3, 2) * (1 - t) / t)
    for i in range(3, m - 1):
        for j in range(3):
            add(i, j, i, F(1))
        add(i, i, m - 1, -3 * t * (1 - t) / (1 + 3 * (1 - t) ** 2))
    matrices = []
    for row in theta:
        a = [[F(0) for _ in range(m)] for _ in range(m)]
        for (j, k), value in row.items():
            a[j][k] = a[k][j] = value / 2
        matrices.append(a)
    return r, matrices


def effective(r, a):
    m, n = len(r), len(r[0])
    return [[sum((r[j][p] * a[j][k] * r[k][q]
                  for j in range(m) for k in range(m)), F(0))
             for q in range(n)] for p in range(n)]


def error_polynomial(matrix, output_index, tau, noisy_center):
    n = len(matrix)
    polynomial = defaultdict(F)
    zero = (0,) * (2 * n)
    for p in range(n):
        for q in range(n):
            for noisy_p, noisy_q in product((False, True), repeat=2):
                powers = [0] * (2 * n)
                powers[p + n * noisy_p] += 1
                powers[q + n * noisy_q] += 1
                polynomial[tuple(powers)] += matrix[p][q] * tau ** (noisy_p + noisy_q)
    trace = sum((matrix[p][p] for p in range(n)), F(0))
    polynomial[zero] -= trace * (1 + tau ** 2 if noisy_center else 1)
    target = [0] * (2 * n)
    target[0] = target[output_index + 1] = 1
    polynomial[tuple(target)] -= 1
    return {powers: coefficient for powers, coefficient in polynomial.items() if coefficient}


def joint_moment(powers, n):
    if any(power % 2 for power in powers):
        return 0
    result = 1
    for power in powers[n:]:
        for factor in range(1, power, 2):
            result *= factor
    return result


def second_moment(polynomial, n):
    terms = list(polynomial.items())
    result = F(0)
    for i, (p, c) in enumerate(terms):
        for j in range(i, len(terms)):
            q, d = terms[j]
            moment = joint_moment(tuple(x + y for x, y in zip(p, q)), n)
            if moment:
                result += c * d * moment * (1 if i == j else 2)
    return result


def check_noise(matrix, output_index, tau):
    n = len(matrix)
    clean = second_moment(error_polynomial(matrix, output_index, F(0), True), n)
    noisy = second_moment(error_polynomial(matrix, output_index, tau, True), n)
    norm = sum((entry ** 2 for row in matrix for entry in row), F(0))
    trace = sum((matrix[j][j] for j in range(n)), F(0))
    expected = clean + (4 * tau ** 2 + 2 * tau ** 4) * norm
    assert noisy == expected
    clean_center_risk = second_moment(error_polynomial(matrix, output_index, tau, False), n)
    assert clean_center_risk == noisy + tau ** 4 * trace ** 2
    return clean, noisy


def main():
    diagonal_cases = []
    for m in (4, 5, 6, 7, 8, 9):
        for t in (F(1, 5), F(1, 10), F(1, 20)):
            r, a = core(m, t)
            assert all(sum((r[j][p] for j in range(m)), F(0)) == 1 for p in range(m + 1))
            q = [effective(r, ai)[0][0] for ai in a]
            expected_q = [F(1, 6) / t] * 3 + [F(0)] * (m - 4) + [-(1 - t) / (2 * t)]
            assert q == expected_q
            expected_sum = (1 + 3 * (1 - t) ** 2) / (12 * t ** 2)
            assert sum((entry ** 2 for entry in q), F(0)) == expected_sum
            diagonal_cases.append(dict(m=m, t=str(t), q=[str(x) for x in q], squared_norm=str(expected_sum)))

    noise_rows = []
    for m in (4, 5):
        for t in (F(1, 5), F(1, 10)):
            r, a = core(m, t)
            for i, ai in enumerate(a):
                matrix = effective(r, ai)
                for tau in (F(1, 100), F(1, 10)):
                    clean, noisy = check_noise(matrix, i, tau)
                    noise_rows.append(dict(family='core', m=m, t=str(t), i=i,
                                           tau=str(tau), clean=str(clean), noisy=str(noisy)))

    for m in (4, 5):
        for i in range(m):
            row = [F(1, m)] + [F(int(j == i)) for j in range(m)]
            matrix = [[F(m, 2) * a * b for b in row] for a in row]
            for tau in (F(1, 100), F(1, 10)):
                clean, noisy = check_noise(matrix, i, tau)
                expected = (tau ** 2 + tau ** 4 / 2) * (m + F(1, m)) ** 2
                assert clean == 0 and noisy == expected
                noise_rows.append(dict(family='square', m=m, i=i, tau=str(tau), noisy=str(noisy)))

    # A signed, nonnormalised router checks that the moment identity does not
    # depend on the probability-router structure of the constructive witness.
    r = [[F(2), F(-1), F(3, 2), F(0)],
         [F(-3, 4), F(2), F(1), F(-2)],
         [F(1, 3), F(0), F(-1), F(5, 2)]]
    for i in range(3):
        a = [[F(0) if j == k else F((i + 1) * (j + k + 1), 7)
              for k in range(3)] for j in range(3)]
        for tau in (F(1, 100), F(1, 10)):
            clean, noisy = check_noise(effective(r, a), i, tau)
            noise_rows.append(dict(family='signed_router', m=3, i=i, tau=str(tau),
                                   clean=str(clean), noisy=str(noisy)))

    old_run = REV/'results/architectural_relevance.json'
    old_check = REV/'results/architectural_relevance_verification.json'
    old = json.loads(old_run.read_text(encoding='utf-8-sig'))
    check = json.loads(old_check.read_text(encoding='utf-8-sig'))
    assert sha(old_run) == '270770f5375c0ccf210c9cc183e5e2182b44a17b3d134b539cf2b1defe4e559f'
    assert sha(old_check) == 'c31eaa7ebe622e0d2da8d319e100dba8174cda101f17b48dd8eca3fc1658dbb9'
    assert check['source_run_sha256'] == sha(old_run)
    sources = [Path(__file__), REV/'manuscript/input_bucket_noise.tex',
               COMP/'manuscript_revision/support/budget_proof.tex',
               COMP/'manuscript_revision/support/asymptotic_constant.tex', old_run, old_check]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  exact_rational_core_cases=len(diagonal_cases),
                  exact_noise_identity_cases=len(noise_rows),
                  exact_clean_center_bias_cases=len(noise_rows),
                  diagonal_cases=diagonal_cases, noise_rows=noise_rows,
                  historical_results_hashes_unchanged=True,
                  historical_experiments_rerun=False, new_fitting_or_test_scoring=False,
                  scope='Finite exact identities only; not a proof of the global bound, neural benefit or priority.',
                  sources=[dict(path=str(p),sha256=sha(p)) for p in sources])
    output = HERE/'exact_identity_verification.json'
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('passed', 'exact_rational_core_cases',
                        'exact_noise_identity_cases', 'exact_clean_center_bias_cases',
                        'historical_results_hashes_unchanged')}))


if __name__ == '__main__':
    main()
