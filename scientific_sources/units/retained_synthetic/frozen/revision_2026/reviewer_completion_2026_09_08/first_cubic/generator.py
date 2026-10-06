"""Specified finite subset population and independent Walsh coefficient audit."""
from pathlib import Path
import itertools
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
SUPPORTS = [(0,1,2),(0,1,3),(0,1,4),(0,5,6),(0,5,7),(8,9,10),(8,9,11),(2,6,10)]
SIGNS = [1,-1,1,1,-1,1,-1,1]


def population():
    return ((np.arange(4096)[:, None] >> np.arange(12)) & 1).astype(np.uint8)


def target_numerators(bits, task):
    w = 2*np.asarray(bits, dtype=np.int64)-1
    supports = [(i,) for i in range(8)] if task == "first" else SUPPORTS
    assert task in ("first", "cubic")
    return sum(sign*np.prod(w[:, support], axis=1) for sign, support in zip(SIGNS, supports))


def split(seed):
    ids = np.random.default_rng(2026090806+seed).permutation(4096)
    return dict(train=ids[:2048], val=ids[2048:3072], test=ids[3072:])


def walsh_transform(values):
    # Boolean-cube Walsh-Hadamard transform; array order uses least-significant bit first.
    out = np.asarray(values, dtype=np.int64).copy()
    width = 1
    while width < len(out):
        for start in range(0, len(out), 2*width):
            a = out[start:start+width].copy()
            b = out[start+width:start+2*width].copy()
            out[start:start+width], out[start+width:start+2*width] = a+b, a-b
        width *= 2
    return out


def audit():
    bits = population()
    records = []
    for task in ("first", "cubic"):
        numerator = target_numerators(bits, task)
        assert int(numerator.sum()) == 0
        assert int(numerator @ numerator) == 4096*8
        transformed = walsh_transform(numerator)
        active = np.where(transformed != 0)[0]
        supports = [(i,) for i in range(8)] if task == "first" else SUPPORTS
        expected = {sum(1 << i for i in support): sign*((-1)**len(support))*4096
                    for sign, support in zip(SIGNS, supports)}
        assert {int(i): int(transformed[i]) for i in active} == expected
        records.append(dict(task=task, supports=supports, integer_signs=SIGNS,
            denominator="sqrt(8)", population_mean_exact=0, population_variance_exact=1,
            complete_walsh_nonzero_count=len(active), maximum_walsh_degree=max(int(i).bit_count() for i in active),
            empty_set_target_numerator=int(numerator[0]), full_set_target_numerator=int(numerator[-1])))
    partitions = []
    for seed in range(100, 110):
        ids = split(seed)
        assert sorted(np.concatenate(list(ids.values())).tolist()) == list(range(4096))
        assert all(not (set(ids[a]) & set(ids[b])) for a, b in itertools.combinations(ids, 2))
        partitions.append(dict(seed=seed, sizes={key: len(value) for key, value in ids.items()},
            empty_set_split=next(key for key, value in ids.items() if 0 in value),
            full_set_split=next(key for key, value in ids.items() if 4095 in value)))
    result = dict(passed=True, exact_population=4096, targets=records, partitions=partitions,
        qualification="Exact generator/Fourier audit only; no neural training or empirical advantage is established")
    (HERE/"generator_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=True, target_degrees=[r["maximum_walsh_degree"] for r in records], partitions=len(partitions))))


if __name__ == "__main__":
    audit()
