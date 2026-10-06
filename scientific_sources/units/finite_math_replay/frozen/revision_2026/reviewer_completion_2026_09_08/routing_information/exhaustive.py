"""All finite-domain hard routers and exact rational decoder risks."""
from fractions import Fraction
from itertools import product, permutations
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
X=np.asarray(list(product((-1,1),repeat=6)),dtype=np.int64)
T=X[:,0]*X[:,1]+X[:,2]*X[:,3]+X[:,4]*X[:,5]


def risk(z):
    groups={}
    for row,t in zip(z,T):
        groups.setdefault(tuple(int(a) for a in row),[]).append(int(t))
    total=Fraction(0)
    mean_square=Fraction(0)
    conflict=0
    pairs=0
    for values in groups.values():
        n=len(values)
        s=sum(values)
        total += sum(v*v for v in values)-Fraction(s*s,n)
        mean_square += Fraction(s*s,n)
        conflict += n if len(set(values))>1 else 0
        pairs += n*(n-1)
    result=total/(len(X)*3)
    independent=Fraction(sum(int(t)**2 for t in T),len(X)*3)-mean_square/(len(X)*3)
    assert result==independent and 0<=result<=1
    return dict(risk_exact=str(result),risk=float(result),distinct_representations=len(groups),
        maximum_fibre=max(map(len,groups.values())),ambiguous_target_mass_exact=str(Fraction(conflict,len(X))),
        ambiguous_target_mass=conflict/len(X),distinct_input_collision_probability_exact=str(Fraction(pairs,len(X)*(len(X)-1))),
        distinct_input_collision_probability=pairs/(len(X)*(len(X)-1)))


def summarize(rows):
    values=[Fraction(r["risk_exact"]) for r in rows]
    return dict(routers=len(rows),minimum_risk_exact=str(min(values)),maximum_risk_exact=str(max(values)),
        mean_risk_exact=str(sum(values)/len(values)),zero_risk_routers=sum(v==0 for v in values),
        distinct_risks={str(v):values.count(v) for v in sorted(set(values))})


def main():
    assert T.sum()==0 and int(T@T)==len(X)*3
    assert risk(X)["risk_exact"]=="0"
    assert risk(np.zeros((len(X),3),dtype=np.int64))["risk_exact"]=="1"
    rows=[]
    for router in product(range(3),repeat=6):
        matrix=np.zeros((3,6),dtype=np.int64)
        matrix[router,np.arange(6)]=1
        counts=matrix.sum(1).tolist()
        penalty=3*sum((Fraction(n,6)-Fraction(1,3))**2 for n in counts)
        z=X@matrix.T
        information=risk(z)
        for perm in permutations(range(3)):
            assert risk(z[:,perm])==information
        rows.append(dict(router="".join(map(str,router)),counts="/".join(map(str,counts)),
            balance_penalty_exact=str(penalty),balance_penalty=float(penalty),**information))
    assert len(rows)==729
    balanced=[r for r in rows if r["balance_penalty_exact"]=="0"]
    uniform=dict(router="uniform_soft",counts="2/2/2 expected allocation mass",balance_penalty_exact="0",
                 balance_penalty=0.,**risk(np.tile(X.sum(1,keepdims=True),(1,3))))
    fixed=[next(r for r in rows if r["router"]==key) for key in ("001122","010212")]
    output=dict(computed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((HERE/"protocol.md").read_bytes()).hexdigest(),
        passed=True,input_size=64,hard_routers=729,enumerated_balanced_routers=len(balanced),
        target="(X1*X2+X3*X4+X5*X6)/sqrt(3)",input_distribution="Uniform {-1,+1}^6",
        all_hard=summarize(rows),balanced_hard=summarize(balanced),prespecified_examples=fixed+[uniform],
        checks=dict(identity_risk_zero=True,constant_risk_one=True,two_rational_risk_formulas_agree=True,
                    all_six_bucket_permutations_agree=True),
        qualification="Exact population diagnostic for fixed scalar linear compression; no neural model fitting or molecular risk identification")
    (HERE/"audit.json").write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")
    with (HERE/"all_routers.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows+[uniform])
    table="\n".join(f'| {r["router"]} | {r["balance_penalty_exact"]} | {r["risk_exact"]} | {r["distinct_representations"]} | {r["ambiguous_target_mass_exact"]} |' for r in fixed+[uniform])
    text=f'''# Exact routing information diagnostic

All 729 hard assignments of six coordinates to three labelled buckets and the uniform soft assignment were evaluated over all 64 inputs. Target variance is one. Every risk below is an exact rational population value, not a test estimate.

| Prespecified router | Load penalty | Minimum decoder MSE | Distinct representations | Mass with target ambiguity |
|---|---:|---:|---:|---:|
{table}

There are {len(balanced)} balanced hard routers. Their minimum, maximum and mean risks are {output["balanced_hard"]["minimum_risk_exact"]}, {output["balanced_hard"]["maximum_risk_exact"]} and {output["balanced_hard"]["mean_risk_exact"]}; {output["balanced_hard"]["zero_risk_routers"]} attain zero risk. The complete uniform distribution over all 729 hard routers has mean risk {output["all_hard"]["mean_risk_exact"]}. The minima are oracle diagnostics over a fully enumerated population, not learning results.

The load penalty therefore cannot identify target information, even on this small population. Target-pair specialization can be useful: a bucket containing Xi and Xj makes Xi*Xj=((Xi+Xj)^2-2)/2 recoverable. It should not automatically be called semantic collapse. Conversely, perfectly balanced uniform soft routing repeats the same statistic in every bucket and has the reported positive target risk. Neither observation proves that the neural balancing penalty helps or harms training; that requires the separate fitted ablation.

The classical conditional-expectation risk identity was evaluated by two rational formulas. Identity and constant representations give risks zero and one, and all six bucket-label permutations preserve the information metrics. The full 730-row table records both representation collision probability and target-ambiguous mass; these are distinct quantities. No approximate floating-point collision threshold was used.

These results concern input-independent scalar compression. They are not molecular Bayes risks, guarantees for vector/input-dependent LMA routing, or a new theorem. The model's residual and query paths are outside this compression diagnostic.
'''
    (HERE/"report.md").write_text(text,encoding="utf-8")
    print(json.dumps(output),flush=True)


if __name__=="__main__":
    main()
