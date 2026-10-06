"""Independent stored-cache checks using integer parity and declared row IDs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import execution
from models import ARMS, PairResidual

HERE = Path(__file__).resolve().parent
CACHE = HERE/'caches/cubic_v1'
PARENT = HERE.parent/'first_cubic'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integer_target(identifier):
    # Independent parity evaluation of the eight specified odd-degree terms.
    supports = (7, 11, 19, 97, 161, 1792, 2816, 1092)
    signs = (1, -1, 1, 1, -1, 1, -1, 1)
    numerator = sum(sign*(2*((identifier & support).bit_count() % 2)-1)
                    for support, sign in zip(supports, signs))
    return numerator/math.sqrt(8)


def main():
    output = HERE/'cubic_cache_audit.json'
    if output.exists():
        raise FileExistsError('Preserve the completed independent cache audit')
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = CACHE/'manifest.json'
    manifest = read(manifest_path)
    assert manifest['passed'] and manifest['parent_predictors'] == 10 and len(manifest['files']) == 20
    assert manifest['residual_optimizer_updates'] == 0 and not manifest['new_test_predictions']
    assert not manifest['ligand_contact_parents_ready'] and not manifest['residual_study_activated']
    for source in manifest['sources']:
        assert sha(source['path']) == source['sha256'], source['path']
    by_seed = {row['seed']: row for row in manifest['checks']}
    assert set(by_seed) == set(range(100, 110))
    file_map = {Path(row['path']).name: row for row in manifest['files']}
    rows, original_files = [], set()
    for seed in range(100, 110):
        order = np.random.default_rng(2026090806+seed).permutation(4096)
        seen_ids = set()
        for split, expected in [('train', order[:2048]), ('val', order[2048:3072])]:
            filename = f'seed{seed}_{split}.pt'
            entry = file_map[filename]
            path = CACHE/filename
            assert Path(entry['path']).resolve() == path.resolve() and sha(path) == entry['sha256']
            blob = torch.load(path, weights_only=True, map_location='cpu')
            assert blob['split'] == split
            assert blob['target_transform'] == 'identity_original_target_units'
            assert not blob['normalization_fitted_to_rows'] and (blob['target_mean'], blob['target_sd']) == (0., 1.)
            ids = [int(value.removeprefix('subset:')) for value in blob['ids']]
            assert blob['ids'] == [f'subset:{value}' for value in ids]
            np.testing.assert_array_equal(ids, expected)
            assert not (set(ids) & seen_ids)
            seen_ids.update(ids)
            truth = torch.tensor([integer_target(value) for value in ids], dtype=torch.float64)
            assert torch.equal(truth, blob['truth'])
            assert blob['f0'].dtype == blob['z'].dtype == torch.float32 and blob['truth'].dtype == torch.float64
            assert blob['z'].shape == (len(ids), 8, 12) and blob['f0'].shape == (len(ids),)
            assert all(torch.isfinite(blob[name]).all() for name in ('f0', 'z', 'truth'))
            assert blob['parent_checkpoint_sha256'] == by_seed[seed]['parent_checkpoint_sha256']
            runtime_cache = execution.make_cache(split, blob['ids'], blob['f0'], blob['z'], truth,
                                                 0., 1., blob['parent_checkpoint_sha256'])
            assert runtime_cache.digest == blob['content_digest'] == entry['content_digest']
            assert torch.equal(runtime_cache.y_std, truth.float())
            # Every residual arm must preserve these actual parent predictions
            # at initialization, including the correct original target units.
            for arm in ARMS:
                model = PairResidual(arm, 12, seed=seed).eval()
                working, original = execution.predict(model, runtime_cache)
                assert torch.equal(working, blob['f0'])
                assert torch.equal(original, blob['f0'].double())
            native_mse = float(torch.mean((blob['f0'].double()-truth)**2))
            row = dict(seed=seed, split=split, rows=len(ids),
                independent_integer_target_exact=True, source_row_order_exact=True,
                all_three_zero_residual_predictions_exact=True,
                cache_units_identity=True, content_digest_verified=True)
            if split == 'val':
                check = next(s for s in by_seed[seed]['splits'] if s['split'] == 'val')
                assert abs(native_mse-check['cpu_validation_mse']) < 2e-13
                candidate_path = PARENT/'candidates'/(by_seed[seed]['selected_id']+'.json')
                candidate = read(candidate_path)
                archive_path = (PARENT/candidate['validation_prediction']).resolve()
                assert archive_path.is_relative_to((PARENT/'validation_predictions').resolve())
                assert sha(archive_path) == candidate['validation_prediction_sha256']
                original_files.update([candidate_path, archive_path])
                with np.load(archive_path) as archived:
                    np.testing.assert_array_equal(archived['ids'], expected)
                    np.testing.assert_array_equal(archived['truth'], truth.numpy())
                    prior = archived['prediction'].copy()
                differences = np.abs(prior-blob['f0'].double().numpy())
                assert np.all(differences <= 3e-5+3e-5*np.abs(prior))
                assert float(differences.max()) == check['archived_gpu_max_prediction_delta']
                row.update(cpu_validation_mse=native_mse,
                           archived_gpu_max_difference=float(differences.max()),
                           archived_gpu_mse_difference=check['archived_gpu_validation_mse_delta'])
            assert runtime_cache.digest == runtime_cache.fingerprint()
            rows.append(row)
        assert len(seen_ids) == 3072
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), HERE/'cache_protocol.md', HERE/'models.py', HERE/'execution.py',
               HERE/'selection.py', manifest_path] + sorted(original_files)
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        parents=10, split_files=20, parent_row_pairs=sum(r['rows'] for r in rows), checks=rows,
        maximum_archived_gpu_prediction_delta=max(r['archived_gpu_max_difference'] for r in rows if r['split']=='val'),
        maximum_absolute_archived_gpu_validation_mse_delta=max(abs(r['archived_gpu_mse_difference']) for r in rows if r['split']=='val'),
        source_manifest_sha256=sha(manifest_path), python_version=sys.version,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        cpu_threads=1, cuda_initialized=False, new_test_evaluation=False, residual_optimizer_updates=0,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        artifacts=[dict(path=r['path'],sha256=r['sha256']) for r in manifest['files']],
        qualification='10 inherited synthetic parents, not 10 independent populations. Fitting/validation cache and target-unit checks only; molecular parents and residual efficacy remain unestablished.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, parents=10, split_files=20, parent_row_pairs=result['parent_row_pairs'],
        max_cpu_gpu_prediction_delta=result['maximum_archived_gpu_prediction_delta'],
        max_validation_mse_delta=result['maximum_absolute_archived_gpu_validation_mse_delta'],
        residual_updates=0, new_test_evaluation=False)))


if __name__ == '__main__':
    main()
