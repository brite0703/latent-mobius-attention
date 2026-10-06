"""A completely specified majority-of-three tree and exact count-information audit."""
from pathlib import Path
import itertools
import json
import math
from fractions import Fraction
import numpy as np

HERE = Path(__file__).resolve().parent


def targets(bits):
    values = np.asarray(bits, dtype=np.uint8)
    n = values.shape[1]
    assert n >= 3
    while values.shape[1] > 1:
        assert values.shape[1] % 3 == 0
        values = (values.reshape(len(values), -1, 3).sum(2) >= 2).astype(np.uint8)
    return values[:, 0].astype(np.int64)


def generate(depth, seed):
    n = 3**depth
    rng = np.random.default_rng(2026090805+1000*depth+seed)
    bits = rng.integers(0, 2, size=(6000, n), dtype=np.uint8)
    return dict(x=bits, y=targets(bits), count=bits.sum(1, dtype=np.int64))


def convolve(a, b):
    output = [0]*(len(a)+len(b)-1)
    for i, aa in enumerate(a):
        for j, bb in enumerate(b):
            output[i+j] += int(aa)*int(bb)
    return output


def count_table(depth):
    table = [[1, 0], [0, 1]]
    for _ in range(depth):
        child = [[row[label] for row in table] for label in (0, 1)]
        n = 3*(len(table)-1)
        updated = [[0, 0] for _ in range(n+1)]
        for labels in itertools.product((0, 1), repeat=3):
            multiplicities = convolve(convolve(child[labels[0]], child[labels[1]]), child[labels[2]])
            out = int(sum(labels) >= 2)
            for c, value in enumerate(multiplicities):
                updated[c][out] += value
        table = updated
    return table


def audit():
    rows = []
    for depth in (2, 3, 4):
        n = 3**depth
        table = count_table(depth)
        assert all(sum(row) == math.comb(n, c) for c, row in enumerate(table))
        assert sum(row[0] for row in table) == sum(row[1] for row in table) == 2**(n-1)
        risk = Fraction(sum(min(row) for row in table), 2**n)
        assert risk > 0
        rows.append(dict(depth=depth, n=n, count_by_label_exact=table,
                         optimal_count_only_classification_error_exact=str(risk),
                         optimal_count_only_classification_error=float(risk)))
    bits = np.asarray(list(itertools.product((0, 1), repeat=9)), dtype=np.uint8)
    truth = targets(bits)
    table = [[0, 0] for _ in range(10)]
    for x, y in zip(bits, truth):
        table[int(x.sum())][int(y)] += 1
        # Independent recursive sign polynomial at depth two.
        signs = 2*x.astype(np.int64)-1
        child = [(a+b+c-a*b*c)//2 for a, b, c in signs.reshape(3, 3)]
        a, b, cc = child
        root = (a+b+cc-a*b*cc)//2
        assert int(root > 0) == y
    assert table == count_table(2)
    witnesses = None
    for count in range(10):
        zero = np.where((bits.sum(1) == count) & (truth == 0))[0]
        one = np.where((bits.sum(1) == count) & (truth == 1))[0]
        if len(zero) and len(one):
            witnesses = dict(count=count, zero_label_sequence=bits[zero[0]].tolist(),
                             one_label_sequence=bits[one[0]].tolist())
            break
    assert witnesses
    output = dict(passed=True, depths=rows, depth2_complete_enumeration=512,
        same_count_opposite_label_witness=witnesses,
        qualification="Exact count-only classification diagnostic; not neural results, not a theorem about position-aware encoders")
    (HERE/"generator_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=True, count_only_errors=[(r["depth"], r["optimal_count_only_classification_error"]) for r in rows],
                          witness=witnesses)))


if __name__ == "__main__":
    audit()
