"""Check the global critical-width star bound and its explicit construction.

Universal claims rely on the accompanying proof. Finite symbolic and population
checks verify identities and catch implementation errors; they do not establish
priority, a sharp leading constant, or a theorem about nonlinear vector LMA.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

from verify_border_path import coefficient_map

ROOT = Path(__file__).parent


def star_construction(m, t, symbolic=False):
    if m<4:
        raise ValueError('The construction requires at least four buckets.')
    zero = sp.zeros if symbolic else lambda r,c:np.zeros((r,c))
    third = sp.Rational(1,3) if symbolic else 1/3
    three_quarters = sp.Rational(3,4) if symbolic else .75
    three_halves = sp.Rational(3,2) if symbolic else 1.5
    r = zero(m,m+1)
    for j in range(3):
        r[j,0]=third
    for j in range(m-1):
        r[j,j+1]=t
        r[m-1,j+1]=1-t
    r[m-1,m]=1
    bases=[]
    for i in range(m-1):
        a=zero(m,m)
        if i<3:
            for j,k in itertools.combinations(range(3),2):
                a[j,k]=a[k,j]=three_quarters if i in (j,k) else -three_quarters
        else:
            a[0,i]=a[i,0]=three_halves
        bases.append(a)
    a_star=zero(m,m)
    a_star[0,m-1]=a_star[m-1,0]=three_halves
    total=sum(bases,zero(m,m))
    matrices=[a/t for a in bases]+[a_star-(1-t)*total/t]
    slots=list(itertools.combinations(range(m),2))
    theta=zero(len(slots),m)
    for e,a in enumerate(matrices):
        for j,(u,v) in enumerate(slots):
            theta[j,e]=2*a[u,v]
    return r,theta,matrices


def symbolic_checks():
    t=sp.symbols('t',positive=True)
    records=[]
    for m in range(4,10):
        r,theta,matrices=star_construction(m,t,True)
        b,pairs,slots=coefficient_map(r,True)
        edges=[(0,i) for i in range(1,m+1)]
        target=sp.eye(len(pairs))[:,[pairs.index(e) for e in edges]]
        residual=(b*theta-target).applyfunc(sp.factor)
        assert residual.extract([pairs.index(e) for e in edges],range(m))==sp.zeros(m,m)
        assert r.T*sp.ones(m,1)==sp.ones(m+1,1)
        assert sp.factor(r[:,1:].det())==t**(m-1)
        risk=sp.factor(sum(v*v for v in residual)/m)
        expected=9*t**2-sp.Rational(27,2*m)*t**3+sp.Rational(27,4*m)*t**4
        assert sp.simplify(risk-expected)==0
        norms=[sp.factor(theta[:,j].dot(theta[:,j])) for j in range(m)]
        for i in range(m-1):
            wanted=sp.Rational(27,4)/t**2 if i<3 else 9/t**2
            assert sp.simplify(norms[i]-wanted)==0
        wanted_last=9+sp.Rational(9*(4*m-13),4)*(1-t)**2/t**2
        assert sp.simplify(norms[-1]-wanted_last)==0
        records.append(dict(buckets=m,inputs=m+1,average_mse=str(risk),
                            coefficient_norms_squared=[str(v) for v in norms],
                            leaf_router_determinant=str(t**(m-1))))
    # Exact identity for a nontrivial positive leaf router, including signed u.
    l=sp.Matrix([[2,1,4,3],[3,5,2,1],[1,3,5,2],[4,1,1,4]])/10
    c=sp.Matrix([7,1,1,1])/10
    u=l.inv()*c
    g=l.inv()*l.inv().T
    w=sp.Matrix([g[i,i]/u[i] for i in range(4)])
    d=sp.Matrix([2*g[i,j]-w[i]*u[j]-w[j]*u[i] for i,j in itertools.combinations(range(4),2)])
    coefficients=sp.symbols('a0:6')
    a=sp.zeros(4)
    for value,(i,j) in zip(coefficients,itertools.combinations(range(4),2)):
        a[i,j]=a[j,i]=value/2
    q=l.T*a*l
    cross=2*l.T*a*c
    off=sp.Matrix([2*q[i,j] for i,j in itertools.combinations(range(4),2)])
    assert sp.expand(w.dot(cross)+d.dot(off))==0
    return dict(status='passed',construction_cases=records,
                rational_trace_certificate='identically zero',rational_u=[str(x) for x in u])


def certificate(l,c):
    inverse=np.linalg.inv(l)
    u=inverse@c
    g=inverse@inverse.T
    w=np.diag(g)/u
    pairs=list(itertools.combinations(range(l.shape[0]),2))
    d=np.array([2*g[i,j]-w[i]*u[j]-w[j]*u[i] for i,j in pairs])
    scale=np.linalg.norm(w)
    return u,w/scale,d/scale


def main():
    algebra=symbolic_checks()
    population=[]
    for m in [4,5,6,8,12]:
        cube=np.array(list(itertools.product((-1.,1.),repeat=m+1)))
        truth=cube[:,[0]]*cube[:,1:]
        for t in [.5,.2,.05,.01,.002]:
            base,theta,_=star_construction(m,t)
            for positive in [False,True]:
                r=(1-t**3)*base+t**3/m if positive else base
                b,pairs,slots=coefficient_map(r)
                target=np.eye(len(pairs))[:,[pairs.index((0,i)) for i in range(1,m+1)]]
                coefficient_mse=float(np.sum((b@theta-target)**2)/m)
                z=cube@r.T
                features=np.stack([z[:,j]*z[:,k] for j,k in slots],axis=1)
                features-=features.mean(0)
                cube_mse=float(np.mean((features@theta-truth)**2))
                k=1.5*np.sqrt(4*m-13)
                budget=k/t
                lower=1/(m*(1+np.sqrt(1+2*(m-1)*budget**2))**2)
                assert np.max(np.linalg.norm(theta,axis=0))<=budget*(1+1e-12)
                assert abs(coefficient_mse-cube_mse)<1e-10
                assert coefficient_mse>=lower-1e-12
                if not positive:
                    exact=9*t*t-27*t**3/(2*m)+27*t**4/(4*m)
                    assert abs(coefficient_mse-exact)<1e-12
                    assert coefficient_mse<=9*t*t+1e-12
                else:
                    assert r.min()>0
                population.append(dict(buckets=m,inputs=m+1,t=t,strictly_positive=positive,
                                       budget=budget,maximum_coefficient_norm=float(np.linalg.norm(theta,axis=0).max()),
                                       coefficient_mse=coefficient_mse,cube_mse=cube_mse,
                                       discrepancy=abs(coefficient_mse-cube_mse),global_lower_bound=lower))
    # Direct checks of the PSD certificate for diverse signed inverse coordinates.
    rng=np.random.default_rng(2026090708)
    certificate_checks=[]
    for m in [3,4,5,8]:
        for seed in range(128):
            l=rng.dirichlet(np.ones(m),size=m).T
            c=rng.dirichlet(np.ones(m))
            if np.linalg.cond(l)>1e8:
                continue
            u,w,d=certificate(l,c)
            upper=8*(1-1/m)*np.linalg.norm(u)**2
            assert np.linalg.norm(d)**2<=upper*(1+1e-10)
            r=np.column_stack((c,l))
            b,pairs,slots=coefficient_map(r)
            erows=[pairs.index((0,i)) for i in range(1,m+1)]
            nrows=[i for i in range(len(pairs)) if i not in erows]
            identity=w@b[erows]+d@b[nrows]
            error=float(np.linalg.norm(identity))
            assert error<1e-7
            certificate_checks.append(dict(buckets=m,normalized_identity_error=error,
                                           any_negative_inverse_coordinate=bool(np.any(u<0))))
    saved=json.loads((ROOT/'output'/'bounded_star_probe.json').read_text())
    saved_checks=[]
    for block in saved['records']:
        budget=block['budget'];m=4
        lower=1/(m*(1+np.sqrt(1+2*(m-1)*budget**2))**2)
        for row in block['runs']:
            assert row['average_mse']>=lower-1e-10
            saved_checks.append(row['average_mse']-lower)
    report=dict(interpretation='Global optimized rate is proved in the companion note; constants are unmatched; scalar centered quadratic scope.',
                symbolic_checks=algebra,population_cases=len(population),population=population,
                max_population_discrepancy=max(r['discrepancy'] for r in population),
                dual_certificate_cases=len(certificate_checks),
                max_normalized_certificate_error=max(r['normalized_identity_error'] for r in certificate_checks),
                signed_inverse_cases=sum(r['any_negative_inverse_coordinate'] for r in certificate_checks),
                saved_bounded_fit_checks=len(saved_checks))
    (ROOT/'output'/'global_star_budget_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['population','symbolic_checks']}),flush=True)


if __name__=='__main__':
    main()
