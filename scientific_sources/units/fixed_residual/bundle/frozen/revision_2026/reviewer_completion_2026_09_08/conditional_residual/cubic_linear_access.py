"""Exact dyadic certificates and input recovery for the fixed cubic caches."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'first_cubic'
CACHE = HERE/'caches/cubic_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dyadic(array):
    """Represent every finite binary float as integers over one exact denominator."""
    ratios = [float(value).as_integer_ratio() for value in np.asarray(array).flat]
    denominator = max(d for _, d in ratios)
    values = np.array([n*(denominator//d) for n, d in ratios], dtype=object)
    assert all(denominator % d == 0 for _, d in ratios)
    return values.reshape(np.shape(array)), denominator


def fraction_record(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximate=float(value))


def certificate(dictionary, inverse):
    k, kd = dyadic(dictionary)
    q, qd = dyadic(inverse)
    denominator = kd*qd
    error = np.eye(12, dtype=object)*denominator-q@k
    row_sum = max(sum(abs(int(x)) for x in row) for row in error)
    bound = Fraction(row_sum, denominator)
    return bound, q, qd


def exact_cached_recovery(blob, q, qd):
    ids = [int(value.split(':')[1]) for value in blob['ids']]
    bits = ((np.asarray(ids)[:, None] >> np.arange(12)) & 1).astype(np.uint8)
    z = blob['z'].numpy().reshape(len(ids), 96)
    largest = Fraction(0)
    wrong_bits, wrong_records = 0, 0
    for start in range(0, len(ids), 256):
        integers, denominator = dyadic(z[start:start+256])
        recovered = integers@q.T
        denominator *= qd
        for row, truth in zip(recovered, bits[start:start+256]):
            missed = 0
            for value, bit in zip(row, truth):
                value = int(value)
                missed += int((2*value >= denominator) != bool(bit))
                largest = max(largest, Fraction(abs(value-int(bit)*denominator), denominator))
            wrong_bits += missed
            wrong_records += int(missed > 0)
    return dict(records=len(ids), exact_integer_rounding_wrong_bits=wrong_bits,
                exact_integer_rounding_wrong_records=wrong_records,
                maximum_coordinate_error=fraction_record(largest)), bits


def main():
    result_path = HERE/'cubic_linear_access.json'
    if result_path.exists():
        raise FileExistsError('Preserve the completed information-access diagnostic')
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = CACHE/'manifest.json'
    manifest = read(manifest_path)
    if not manifest['passed'] or not read(HERE/'cubic_cache_audit.json')['passed']:
        raise ValueError('Frozen caches must pass their prior checks')
    for row in manifest['sources']+manifest['files']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('A cache source or artifact changed: '+row['path'])
    spec = importlib.util.spec_from_file_location('linear_access_parent_models', PARENT/'neural_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = {manifest_path, HERE/'cubic_cache_audit.json', Path(__file__), HERE/'cubic_linear_access_protocol.md'}
    rows, artifacts = [], []
    for parent in manifest['checks']:
        seed = parent['seed']
        candidate_path = PARENT/'candidates'/(parent['selected_id']+'.json')
        candidate = read(candidate_path)
        checkpoint_path = (PARENT/candidate['checkpoint']).resolve()
        if not checkpoint_path.is_relative_to((PARENT/'checkpoints').resolve()) or sha(checkpoint_path) != parent['parent_checkpoint_sha256']:
            raise ValueError('Parent checkpoint binding changed')
        sources.update([candidate_path, checkpoint_path])
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model = module.build(seed, 'lma1').eval().requires_grad_(False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        layer = model.head.layers[0]
        with torch.no_grad():
            x = torch.eye(12).expand(256, -1, -1)
            h = model.encoder(x)
            routing = F.softmax(layer.W_H(layer.W_k(h)), -1)
            values = layer.W_v(h)
        if not torch.equal(routing, routing[:1].expand_as(routing)) or not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError('The fixed token features differ between identical records')
        p, v = routing[0].numpy().copy(), values[0].numpy().copy()
        dictionary = np.array([[float(p[i, m])*float(v[i, a]) for i in range(12)]
                               for m in range(8) for a in range(12)], dtype=np.float64)
        for m in range(8):
            for a in range(12):
                for i in range(12):
                    assert Fraction(float(dictionary[m*12+a, i])) == Fraction(float(p[i, m]))*Fraction(float(v[i, a]))
        u, singular, vh = np.linalg.svd(dictionary, full_matrices=False)
        inverse = np.linalg.pinv(dictionary)
        if not np.isfinite(inverse).all():
            raise ValueError('The proposed inverse is not finite; preserve and diagnose')
        bound, q, qd = certificate(dictionary, inverse)
        split_checks = []
        for split in ('train', 'val'):
            path = CACHE/f'seed{seed}_{split}.pt'
            blob = torch.load(path, weights_only=True, map_location='cpu')
            if blob['parent_checkpoint_sha256'] != parent['parent_checkpoint_sha256']:
                raise ValueError('Cached representation belongs to another predictor')
            recovery, bits = exact_cached_recovery(blob, q, qd)
            numerical_linear = bits.astype(np.float64)@dictionary.T
            cached_z = blob['z'].numpy().reshape(len(bits), 96).astype(np.float64)
            recovery.update(split=split,
                native_vs_exact_dictionary_max_delta=float(np.max(np.abs(cached_z-numerical_linear))))
            split_checks.append(recovery)
        for name, value in model.state_dict().items():
            if not torch.equal(value, checkpoint['state_dict'][name]):
                raise ValueError('The diagnostic changed its parent parameters/buffers')
        directory = HERE/'linear_access_certificates'
        directory.mkdir(exist_ok=True)
        witness_path = directory/f'seed{seed}.npz'
        if witness_path.exists():
            with np.load(witness_path) as old:
                for key, value in dict(routing=p, values=v, dictionary=dictionary, inverse=inverse).items():
                    np.testing.assert_array_equal(old[key], value)
        else:
            np.savez_compressed(witness_path, routing=p, values=v, dictionary=dictionary, inverse=inverse)
        artifacts.append(dict(path=str(witness_path),sha256=sha(witness_path)))
        rows.append(dict(seed=seed, parent_checkpoint_sha256=parent['parent_checkpoint_sha256'],
            dictionary_shape=[96,12], singular_values=singular.tolist(),
            numerical_condition_number=float(singular[0]/singular[-1]) if singular[-1] else None,
            exact_neumann_residual_bound=fraction_record(bound),
            dictionary_full_column_rank_certified=bool(bound < 1),
            all_exact_dictionary_boolean_inputs_round_correctly=bool(bound < Fraction(1,2)),
            native_cache_recovery=split_checks, parent_state_exact=True,
            certificate_artifact=str(witness_path)))
        print(json.dumps(dict(seed=seed, dictionary_rank_certified=bool(bound < 1),
            native_cached_bit_errors=sum(r['exact_integer_rounding_wrong_bits'] for r in split_checks))), flush=True)
    assert not torch.cuda.is_initialized()
    result = dict(completed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        parents=10, checks=rows, exact_rank_certificates_passed=sum(r['dictionary_full_column_rank_certified'] for r in rows),
        native_record_parent_pairs=sum(c['records'] for r in rows for c in r['native_cache_recovery']),
        native_cached_wrong_bits=sum(c['exact_integer_rounding_wrong_bits'] for r in rows for c in r['native_cache_recovery']),
        native_cached_wrong_records=sum(c['exact_integer_rounding_wrong_records'] for r in rows for c in r['native_cache_recovery']),
        cuda_initialized=False, optimizer_updates=0, target_values_used=False, new_test_predictions=False,
        original_parent_scores_reselected=False, sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(sources)],
        artifacts=artifacts,
        exact_arithmetic_scope='Stored finite float32 routing/value dictionary and finite float64 inverse, represented as dyadic rationals. Native cached tensor bit recovery is also evaluated with exact integer arithmetic.',
        limitations='Classical source-level diagnostic, not a novel theorem, neural training guarantee, unrounded transcendental-network certificate, molecular result, or test-set evaluation.')
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['completed','parents','exact_rank_certificates_passed',
        'native_record_parent_pairs','native_cached_wrong_bits','native_cached_wrong_records','new_test_predictions']}))


if __name__ == '__main__':
    main()
