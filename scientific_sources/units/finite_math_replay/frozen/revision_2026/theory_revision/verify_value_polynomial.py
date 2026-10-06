"""Exact finite checks of the new value-polynomial formula, not quantifier elimination."""
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import random
import sympy as sp

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng = random.Random(49092026)
    cases = []
    evaluated = 0
    for m in range(2, 7):
        pairs = list(combinations(range(m), 2))
        input_pairs = list(combinations(range(m+1), 2))
        for trial in range(8):
            columns = []
            for _ in range(m+1):
                raw = [rng.randrange(0, 6) for _ in range(m)]
                if not sum(raw):
                    raw[0] = 1
                columns.append([F(x, sum(raw)) for x in raw])
            router = [list(row) for row in zip(*columns)]
            theta = [[F(rng.randrange(-6, 7), rng.randrange(1, 6)) for _ in pairs]
                     for _ in range(m)]
            coefficient_risk = F(0)
            for i in range(m):
                for r, s in input_pairs:
                    value = sum(theta[i][k] * (router[j][r]*router[l][s]
                                + router[j][s]*router[l][r])
                                for k, (j, l) in enumerate(pairs))
                    coefficient_risk += (value-int(r==0 and s==i+1))**2
            coefficient_risk /= m
            centers = [sum(router[j][r]*router[l][r] for r in range(m+1))
                       for j, l in pairs]
            full_risk = F(0)
            for x in product((-1, 1), repeat=m+1):
                z = [sum(router[j][r]*x[r] for r in range(m+1)) for j in range(m)]
                features = [z[j]*z[l]-centers[k] for k, (j, l) in enumerate(pairs)]
                for i in range(m):
                    prediction = sum(a*b for a, b in zip(theta[i], features))
                    full_risk += (prediction-x[0]*x[i+1])**2
                    evaluated += 1
            full_risk /= m*2**(m+1)
            assert coefficient_risk == full_risk
            cases.append(dict(m=m,trial=trial,risk_numerator=str(full_risk.numerator),
                              risk_denominator=str(full_risk.denominator),exact=True))

    # Symbolic limits use exactly the established bounding functions.
    lam, dimension = sp.symbols('Lambda m', positive=True)
    lower = 1/(dimension*(1+sp.sqrt(1+2*(dimension-1)*lam**2))**2)
    # A positive auxiliary dimension offset makes m>=4 explicit to SymPy.
    offset = sp.symbols('offset', nonnegative=True)
    lower_limit = sp.simplify(sp.limit(lam**2*lower.subs(dimension, offset+4),lam,sp.oo))
    assert sp.simplify(lower_limit-1/(2*(offset+4)*(offset+3))) == 0
    t = sp.sqrt(27/(4*(lam**2-sp.Rational(3,2))))
    upper = t**2/dimension*(15+sp.Rational(15,2)*t+sp.Rational(21,4)*t**2
                              +3*(dimension-4)/(1+3*(1-t)**2))
    upper_limit = sp.simplify(sp.limit(lam**2*upper,lam,sp.oo))
    assert sp.simplify(upper_limit-(sp.Rational(81,16)+81/dimension)) == 0
    for p in (1,2):
        a, scalar_t = sp.symbols('a t')
        assert sp.Poly((1-a*scalar_t)**2+scalar_t**(2*p),a,scalar_t).total_degree()==4

    sources = [HERE/'asymptotic_value_constant_20260909.md', Path(__file__).resolve(),
               HERE/'global_star_budget.tex']
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        exact_rational_objective_cases=len(cases),full_cube_scalar_predictions=evaluated,
        cases=cases,lower_scaled_limit='1/[2*m*(m-1)]',upper_scaled_limit='81/16+81/m',
        bound_limits_verified_symbolically=True,toy_objectives_both_degree_four=True,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        qualification='Checks only the finite polynomial representation and displayed bound limits. '
        'The all-dimension statement uses the written proof and standard quantifier elimination. '
        'No global minimizer, numerical c_m, neural fit, new test prediction or originality is certified.')
    output = HERE/'output/value_polynomial_20260909.json'
    assert not output.exists(), 'Preserve the first receipt rather than rerunning unchanged checks.'
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['passed','exact_rational_objective_cases',
                                           'full_cube_scalar_predictions','bound_limits_verified_symbolically']}))


if __name__=='__main__':
    main()
