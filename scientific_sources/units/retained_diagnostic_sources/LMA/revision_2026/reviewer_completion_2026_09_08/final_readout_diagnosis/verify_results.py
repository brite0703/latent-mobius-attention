"""Independent QR and scalar-summation checks on all saved fitting outcomes."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy import linalg

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'output/run_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def verify(bound):
    assert artifact(bound['path'])['sha256'] == bound['sha256'], bound['path']


def average_square(values):
    return math.fsum(float(x)*float(x) for x in values)/len(values)


def run():
    result = read(OUTPUT/'results.json')
    verify(result['source_lock'])
    for bound in read(result['source_lock']['path'])['sources'] + result['artifacts']:
        verify(bound)
    rows = result['models']
    assert {(r['head'],r['seed']) for r in rows} == {(h,s) for h in ('lma1','lma2','additive2','cp_pool') for s in range(100,110)}
    assert len(rows) == 40 and all(r['status'] == 'evaluated' for r in rows)
    checked = []
    for row in rows:
        verify(row['checkpoint'])
        with np.load(row['fitting_arrays']['path']) as archive:
            values = {k:archive[k].copy() for k in archive.files}
        ids, a, y = values['ids'], values['a'], values['truth']
        b0, bo, bc = values['beta0'], values['beta_ols'], values['beta_constrained']
        assert a.shape == (2048,25) and np.array_equal(a[:,-1], np.ones(2048))
        np.testing.assert_array_equal(a[:,:24], a[:,:24].astype(np.float32).astype(np.float64))
        original_ids = np.random.default_rng(2026090806+row['seed']).permutation(4096)[:2048]
        np.testing.assert_array_equal(ids, original_ids)
        # Independent target formula: multiply the signed membership variables.
        signs = 2*((ids[:,None] >> np.arange(12)) & 1)-1
        supports = ((0,1,2),(0,1,3),(0,1,4),(0,5,6),(0,5,7),(8,9,10),(8,9,11),(2,6,10))
        weights = (1,-1,1,1,-1,1,-1,1)
        truth = np.array([sum(w*math.prod(int(bits[j]) for j in support) for w,support in zip(weights,supports))/math.sqrt(8)
                          for bits in signs])
        np.testing.assert_array_equal(y, truth)
        # Pivoted QR provides an independent factorization, without the production solver.
        qr, _, rank, _ = linalg.lstsq(a, y, cond=max(a.shape)*np.finfo(float).eps, lapack_driver='gelsy')
        qr_error = average_square(y-a@qr)
        assert rank == row['numerical_rank']
        assert abs(qr_error-row['ols_mse']) <= 1e-10
        metric_deltas = []
        for name, prediction in [('native_float32_mse',values['native_prediction']),
                                 ('native_affine_mse',a@b0),('ols_mse',a@bo),('constrained_mse',a@bc)]:
            delta = abs(average_square(y-prediction)-row[name])
            assert delta < 1e-12, (row['selected_id'],name,delta)
            metric_deltas.append(delta)
        # Scalar summation checks the actual equations rather than rerunning SVD.
        n = len(a)
        ols_residual, bounded_residual = a@bo-y, a@bc-y
        normal = np.array([math.fsum(float(x)*float(e) for x,e in zip(a[:,j],ols_residual))/n for j in range(25)])
        lam = row['multiplier']
        assert lam is not None and lam >= 0
        stationarity = np.array([math.fsum(float(x)*float(e) for x,e in zip(a[:,j],bounded_residual))/n + lam*float(bc[j]) for j in range(25)])
        radius2 = math.fsum(float(x)**2 for x in b0)
        bounded2 = math.fsum(float(x)**2 for x in bc)
        assert bounded2 <= radius2 + 1e-11*(1.+radius2)
        assert abs(lam*(bounded2-radius2)) < 1e-9
        assert np.linalg.norm(normal) < 1e-9 and np.linalg.norm(stationarity) < 1e-9
        orthogonal_gap = average_square(a@(b0-bo))
        pythagorean_error = abs(row['native_affine_mse']-row['ols_mse']-orthogonal_gap)
        assert pythagorean_error < 1e-9
        direct_native_delta = max(abs(math.fsum(float(x)*float(c) for x,c in zip(a[i],b0))-float(values['native_prediction'][i]))
                                  for i in (0,1,127,255,256,1023,2047))
        assert direct_native_delta < 1e-5
        checked.append(dict(selected_id=row['selected_id'], fitting_rows=n, qr_rank=int(rank),
            qr_mse_delta=abs(qr_error-row['ols_mse']), maximum_independent_metric_delta=max(metric_deltas),
            scalar_normal_residual=float(np.linalg.norm(normal)), scalar_stationarity=float(np.linalg.norm(stationarity)),
            pythagorean_discrepancy=pythagorean_error, scalar_native_maximum=direct_native_delta))
    # Recompute summaries and contrasts from all saved rows, preserving every seed.
    for head, summary in result['summaries'].items():
        group = [r for r in rows if r['head'] == head]
        for metric, reported in summary['errors'].items():
            assert abs(math.fsum(r[metric] for r in group)/10-reported['mean']) < 1e-14
            assert float(np.median([r[metric] for r in group])) == reported['median']
    lookup = {(r['head'],r['seed']):r for r in rows}
    for contrast in result['paired_contrasts']:
        expected = [lookup[(contrast['left'],s)][contrast['measure']]-lookup[(contrast['right'],s)][contrast['measure']] for s in range(100,110)]
        assert [r['difference'] for r in contrast['seeds']] == expected
        assert abs(math.fsum(expected)/10-contrast['mean_difference']) < 1e-14
        assert sum(x<0 for x in expected) == contrast['lower_count']
    receipt = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        checked_models=len(checked), fitting_record_model_pairs=sum(r['fitting_rows'] for r in checked),
        independent_scalar_native_predictions=7*len(checked), checks=checked,
        sources=[artifact(Path(__file__).resolve()), artifact(OUTPUT/'results.json'), artifact(OUTPUT/'all_models.csv')],
        scientific_validation_or_test_predictions=False, neural_parameter_updates=False,
        formal_rank_or_optimality_certificate=False)
    target = HERE/'independent_verification.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=True, checked_models=40, fitting_record_model_pairs=81920,
        maximum_qr_mse_delta=max(r['qr_mse_delta'] for r in checked),
        maximum_scalar_stationarity=max(r['scalar_stationarity'] for r in checked))))


if __name__ == '__main__':
    run()
