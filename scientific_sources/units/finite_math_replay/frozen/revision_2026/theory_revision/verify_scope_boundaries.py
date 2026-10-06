"""Exact rational scope counterexamples; no neural fitting or test access."""
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NOTE = HERE.parent/'reviewer_completion_2026_09_08/theory_originality/structural_scope_boundaries_20260909.md'


def frac(x):
    return dict(numerator=str(x.numerator), denominator=str(x.denominator), approximate=float(x))


def zeros(n, p):
    return [[F(0) for _ in range(p)] for _ in range(n)]


def scalar_router(m, target, a=None):
    r = zeros(m, m+1)
    if a is None:
        for t in range(m+1):
            r[0 if t==0 else 1 if t==target+1 else 2][t] = F(1)
        theta = {(0,1):F(1)}
    else:
        for j in range(m):
            for t in range(m+1):
                r[j][t] = (1-a)/m if t in (0,target+1) else F(1,m)
        r[0][0] += a
        r[1][target+1] += a
        theta = {(0,1):1/a**2, (0,3):-1/a**2, (1,2):-1/a**2, (2,3):1/a**2}
    return r, theta


def scalar_polynomial(r, theta):
    n = len(r[0])
    q = zeros(n,n)
    for (j,l),weight in theta.items():
        for s in range(n):
            for t in range(n):
                q[s][t] += weight*(r[j][s]*r[l][t]+r[l][s]*r[j][t])/2
    return q


def check_router(r, positive):
    for t in range(len(r[0])):
        assert sum(row[t] for row in r) == 1
    assert all(x>0 if positive else x>=0 for row in r for x in row)


def scalar_cases():
    records = []
    for m in range(3,11):
        for target in range(m):
            for a in ([None] + ([F(1,4),F(1,2),F(2,3)] if m>=4 else [])):
                r,theta = scalar_router(m,target,a)
                check_router(r,a is not None)
                q = scalar_polynomial(r,theta)
                expected = zeros(m+1,m+1)
                expected[0][target+1] = expected[target+1][0] = F(1,2)
                assert q == expected
                norm2 = sum(v*v for v in theta.values())
                assert norm2 == (1 if a is None else 4/a**4)
                records.append(dict(m=m,target=target,a=None if a is None else frac(a),
                    strictly_positive=a is not None,minimum_routing_entry=frac(min(x for row in r for x in row)),
                    coefficient_squared_norm=frac(norm2), all_polynomial_coefficients_exact=True))
    return records


def vector_data(m,a):
    r = [[F(1,m)]+[a*int(j==l)+(1-a)/m for l in range(m)] for j in range(m)]
    inverse = [[F(int(i==l),1)/a-(1-a)/(a*m) for l in range(m)] for i in range(m)]
    check_router(r,True)
    for i in range(m):
        for l in range(m):
            assert sum(inverse[i][j]*r[j][l+1] for j in range(m)) == int(i==l)
            assert sum(r[i][j+1]*inverse[j][l] for j in range(m)) == int(i==l)
    return r,inverse


def vector_cases():
    records = []
    for m in range(2,11):
        for a in (F(1,4),F(1,2),F(2,3)):
            r,inverse = vector_data(m,a)
            pairs = list(itertools.permutations(range(m),2))
            for target in range(m):
                eta = {pair:F(m,m-1)*inverse[target][pair[1]] for pair in pairs}
                coefficients = [sum(eta[(j,l)]*r[j][0]*r[l][s+1] for j,l in pairs) for s in range(m)]
                assert coefficients == [F(int(s==target)) for s in range(m)]
                norm2 = sum(w*w for w in eta.values())
                assert norm2 == F(m,m-1)+m/a**2
                records.append(dict(m=m,target=target,a=frac(a),product_coordinates=len(pairs),
                    coefficient_squared_norm=frac(norm2),minimum_routing_entry=frac(min(x for row in r for x in row)),
                    all_polynomial_coefficients_exact=True))
    return records


def full_cube_checks():
    records = []
    for m in (4,6,8):
        x = np.asarray(list(itertools.product((-1.,1.),repeat=m+1)))
        a = F(1,2)
        vr,inv = vector_data(m,a)
        vr = np.array(vr,dtype=float)
        # Compute both value channels from their actual typed token contributions.
        zc = x[:,[0]]@vr[:,[0]].T
        zy = x[:,1:]@vr[:,1:].T
        for target in range(m):
            sr,theta = scalar_router(m,target,a)
            z = x@np.array(sr,dtype=float).T
            scalar = sum(float(w)*z[:,j]*z[:,l] for (j,l),w in theta.items())
            vector = sum(float(F(m,m-1)*inv[target][l])*zc[:,j]*zy[:,l]
                         for j,l in itertools.permutations(range(m),2))
            truth = x[:,0]*x[:,target+1]
            maximum = max(float(np.max(np.abs(scalar-truth))),float(np.max(np.abs(vector-truth))))
            assert maximum < 1e-12
            records.append(dict(m=m,target=target,rows=len(x),maximum_scalar_vector_error=maximum,
                scalar_mean_error=abs(float(np.mean(scalar))),vector_mean_error=abs(float(np.mean(vector)))))
    return records


def main():
    scalar,vector,cubes = scalar_cases(),vector_cases(),full_cube_checks()
    paths = [Path(__file__).resolve(),NOTE]
    result = dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        target_specific_scalar_cases=scalar,shared_two_channel_cases=vector,
        complete_boolean_cube_checks=cubes,
        exact_rational_polynomial_cases=len(scalar)+len(vector),
        scalar_vector_cube_predictions=2*sum(r['rows'] for r in cubes),
        maximum_float64_cube_error=max(r['maximum_scalar_vector_error'] for r in cubes),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        neural_containment_claim=False,new_originality_claim=False,scientific_data_access=False,
        interpretation='Exact boundaries of the shared scalar model, not a theorem for native neural LMA')
    path=HERE/'output/scope_boundaries_20260909.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','target_specific_scalar_cases','shared_two_channel_cases','complete_boolean_cube_checks')}))


if __name__=='__main__':
    main()
