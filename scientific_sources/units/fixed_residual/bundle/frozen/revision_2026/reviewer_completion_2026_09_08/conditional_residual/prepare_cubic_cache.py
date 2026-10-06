"""Verify selected cubic parents and cache fitting/validation rows only."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from models import FrozenLmaFeatures
import execution

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'first_cubic'
DESTINATION = HERE/'caches/cubic_v1'
ABS_TOL, REL_TOL = 3e-5, 3e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_path(relative):
    path = (PARENT/relative).resolve()
    if not path.is_relative_to(PARENT.resolve()):
        raise ValueError('Parent artifact escapes its declared study')
    return path


def cache_blob(cache):
    return dict(split=cache.split, ids=list(cache.ids), f0=cache.f0, z=cache.z,
        truth=cache.truth, target_mean=cache.target_mean, target_sd=cache.target_sd,
        parent_checkpoint_sha256=cache.parent_checkpoint_sha256, content_digest=cache.digest,
        target_transform='identity_original_target_units', normalization_fitted_to_rows=False)


def reload_cache(path):
    blob = torch.load(path, weights_only=True, map_location='cpu')
    if blob['target_transform'] != 'identity_original_target_units' or blob['normalization_fitted_to_rows']:
        raise ValueError('The cubic parent does not use fitted target standardization')
    if (blob['target_mean'], blob['target_sd']) != (0., 1.):
        raise ValueError('The inherited cubic target transformation must be identity')
    cache = execution.make_cache(blob['split'], blob['ids'], blob['f0'], blob['z'], blob['truth'],
        blob['target_mean'], blob['target_sd'], blob['parent_checkpoint_sha256'])
    if cache.digest != blob['content_digest']:
        raise ValueError('Serialized cache content differs from its digest')
    return cache


def save_cache(cache, path):
    if path.exists():
        restored = reload_cache(path)
        if restored.digest != cache.digest:
            raise ValueError('An existing frozen cache differs; preserve and diagnose it')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(cache_blob(cache), temporary)
        restored = reload_cache(temporary)
        if restored.digest != cache.digest:
            raise ValueError('Cache round-trip changed its content')
        # A new destination only; no existing result is replaced.
        temporary.rename(path)
    return dict(path=str(path), sha256=sha(path), content_digest=cache.digest,
                rows=len(cache.ids), split=cache.split)


def bind_parents():
    sources = {}

    def verify(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Parent source or artifact changed: '+str(path))
        sources[str(path)] = digest
        return path

    lock_path = verify(PARENT/'implementation_lock.json')
    lock = read(lock_path)
    for row in lock['files']:
        verify(row['path'], row['sha256'])
    selection_path = verify(PARENT/'selection_lock.json')
    selected = read(selection_path)
    if selected['implementation_lock_sha256'] != sha(lock_path):
        raise ValueError('Parent selection and implementation locks disagree')
    candidate_digests = {}
    for row in selected['candidate_records']:
        path = checked_path(row['path'])
        verify(path, row['sha256'])
        candidate_digests[path.name] = row['sha256']
    if not read(verify(PARENT/'final_audit.json'))['passed']:
        raise ValueError('Parent campaign audit is incomplete')
    selections = sorted((r for r in selected['selections'] if r['task'] == 'cubic' and r['head'] == 'lma1'),
                        key=lambda r: r['seed'])
    if [r['seed'] for r in selections] != list(range(100, 110)):
        raise ValueError('All ten specified cubic parents are required')
    parents = []
    for choice in selections:
        if len(choice['candidate_ids']) != 2:
            raise ValueError('Expected exactly two declared parent rates')
        candidates = []
        for identifier in choice['candidate_ids']:
            path = checked_path('candidates/'+identifier+'.json')
            if path.name not in candidate_digests:
                raise ValueError('A candidate is not bound by the parent selection record')
            candidate = read(path)
            spec = candidate['spec']
            if spec not in lock['candidates'] or (spec['task'], spec['head'], spec['seed']) != ('cubic', 'lma1', choice['seed']):
                raise ValueError('Unexpected parent specification')
            if candidate['implementation_lock_sha256'] != sha(lock_path):
                raise ValueError('Parent candidate belongs to a different implementation')
            candidates.append(candidate)
        eligible = [c for c in candidates if c['status'] == 'valid' and math.isfinite(c['best_validation_loss'])]
        if not eligible:
            raise ValueError('The declared baseline has no valid parent candidate')
        best = min(eligible, key=lambda c: (c['best_validation_loss'], c['spec']['lr_index']))
        if best['spec']['id'] != choice['selected_id'] or best['best_validation_loss'] != choice['validation_loss']:
            raise ValueError('Stored baseline selection does not follow the parent validation rule')
        checkpoint = verify(checked_path(best['checkpoint']), best['checkpoint_sha256'])
        if sha(checkpoint) != choice['checkpoint_sha256']:
            raise ValueError('Selected checkpoint hashes disagree')
        validation = verify(checked_path(best['validation_prediction']), best['validation_prediction_sha256'])
        parents.append(dict(choice=choice, candidate=best, checkpoint=checkpoint, validation=validation))
    return parents, sources


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = DESTINATION/'manifest.json'
    if manifest_path.exists():
        existing = read(manifest_path)
        for source in existing['sources']:
            if sha(source['path']) != source['sha256']:
                raise ValueError('Existing cache source changed: '+source['path'])
        for row in existing['files']:
            if sha(row['path']) != row['sha256'] or reload_cache(row['path']).digest != row['content_digest']:
                raise ValueError('Existing cache changed')
        print(json.dumps(dict(preserved_existing_cache=True, files=len(existing['files']))))
        return
    parents, sources = bind_parents()
    module = imported('conditional_cubic_parent_models', PARENT/'neural_models.py')
    generator = imported('conditional_cubic_parent_generator', PARENT/'generator.py')
    files, checks = [], []
    for parent in parents:
        choice, candidate = parent['choice'], parent['candidate']
        seed = choice['seed']
        checkpoint = torch.load(parent['checkpoint'], weights_only=True, map_location='cpu')
        if set(checkpoint) != {'state_dict', 'spec', 'implementation_lock_sha256'}:
            raise ValueError('Unexpected cubic checkpoint unit/provenance schema')
        if checkpoint['spec'] != candidate['spec'] or checkpoint['implementation_lock_sha256'] != candidate['implementation_lock_sha256']:
            raise ValueError('Checkpoint metadata does not match its candidate')
        baseline = module.build(seed, 'lma1').eval()
        baseline.load_state_dict(checkpoint['state_dict'], strict=True)
        saved = execution.tensor_copy(baseline.state_dict())
        view = FrozenLmaFeatures(baseline, 'head.layers.0').eval()
        layer = baseline.head.layers[0]
        partitions = generator.split(seed)
        data_path = PARENT/'data'/f'seed{seed}.npz'
        if str(data_path.resolve()) not in sources:
            raise ValueError('The parent lock does not bind its partition source')
        split_checks = []
        # Only the fitting/validation ID members of the source archive are read.
        with np.load(data_path) as source_npz:
            split_ids = {split: source_npz[split+'_ids'].copy() for split in ('train', 'val')}
        for split in ('train', 'val'):
            ids = split_ids[split]
            np.testing.assert_array_equal(ids, partitions[split])
            bits = ((ids[:, None] >> np.arange(12)) & 1).astype(np.uint8)
            truth = generator.target_numerators(bits, 'cubic')/math.sqrt(8)
            mask = torch.as_tensor(bits.astype(bool))
            predictions, buckets = [], []
            reference_error, native_error = 0., 0.
            for start in range(0, len(ids), 256):
                mm = mask[start:start+256]
                xx = torch.eye(12, dtype=torch.float32).expand(len(mm), -1, -1)
                f0, z = view(xx, mm)
                with torch.no_grad():
                    native = baseline(xx, mm)
                    h = baseline.encoder(xx)
                    routing = F.softmax(layer.W_H(layer.W_k(h)), dim=-1)
                    reference = routing.transpose(1, 2) @ (layer.W_v(h)*mm.unsqueeze(-1))
                native_error = max(native_error, float((f0-native).abs().max()))
                reference_error = max(reference_error, float((z-reference).abs().max()))
                if not torch.equal(f0, native) or not torch.equal(z, reference):
                    raise ValueError('Same-device native prediction or bucket reconstruction differs')
                predictions.append(f0); buckets.append(z)
            f0, z = torch.cat(predictions), torch.cat(buckets)
            cache = execution.make_cache(split, [f'subset:{int(i)}' for i in ids], f0, z,
                torch.from_numpy(truth), 0., 1., choice['checkpoint_sha256'])
            if not torch.equal(cache.y_std, torch.from_numpy(truth).float()):
                raise ValueError('Synthetic fitting target units changed')
            row = dict(seed=seed, split=split, rows=len(ids), native_prediction_max_delta=native_error,
                       independent_bucket_max_delta=reference_error, target_transform='identity',
                       fitting_target_cast_matches=True)
            if split == 'val':
                with np.load(parent['validation']) as archive:
                    np.testing.assert_array_equal(archive['ids'], ids)
                    np.testing.assert_array_equal(archive['truth'], truth)
                    prior = archive['prediction'].copy()
                old_mse = float(np.mean((prior-truth)**2))
                if abs(old_mse-candidate['best_validation_loss']) > 1e-12:
                    raise ValueError('Archived parent validation score differs')
                rebuilt = f0.double().numpy()
                difference = np.abs(rebuilt-prior)
                if not np.all(difference <= ABS_TOL+REL_TOL*np.abs(prior)):
                    raise ValueError('CPU/GPU reconstruction exceeds the prespecified tolerance')
                row.update(archived_gpu_max_prediction_delta=float(difference.max()),
                    archived_gpu_validation_mse=old_mse,
                    cpu_validation_mse=float(np.mean((rebuilt-truth)**2)),
                    archived_gpu_validation_mse_delta=float(np.mean((rebuilt-truth)**2))-old_mse,
                    parent_rate_reselection_performed=False)
            path = DESTINATION/f'seed{seed}_{split}.pt'
            files.append(save_cache(cache, path))
            split_checks.append(row)
        for name, value in baseline.state_dict().items():
            if not torch.equal(value, saved[name]) or not torch.equal(value, checkpoint['state_dict'][name]):
                raise ValueError('The frozen parent state changed')
        checks.append(dict(seed=seed, selected_id=choice['selected_id'],
            parent_checkpoint_sha256=choice['checkpoint_sha256'], parent_state_exact=True,
            splits=split_checks, parent_validation_selection_reconciled=True))
        print(json.dumps(dict(cached_parent_seed=seed, splits=['train','val'])), flush=True)
    assert not torch.cuda.is_initialized()
    for path in [Path(__file__), HERE/'cache_protocol.md', HERE/'models.py', HERE/'execution.py', HERE/'selection.py']:
        sources[str(path.resolve())] = sha(path)
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError('A bound source changed during preparation: '+path)
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        domain='cubic', parent_predictors=10, files=files, checks=checks,
        source_data_access='Only train_ids and val_ids arrays decoded; corresponding inputs/targets regenerated. Whole parent files hashed as bytes.',
        target_transform=dict(mean=0., scale=1., meaning='Identity inherited from the unstandardized cubic parent, not fitted moments'),
        archived_gpu_tolerance=dict(absolute=ABS_TOL, relative=REL_TOL), cpu_threads=1,
        torch_version=str(torch.__version__), cuda_initialized=False,
        heldout_targets_decoded=False, new_test_predictions=False, residual_optimizer_updates=0,
        all_residual_parents_ready=False, ligand_contact_parents_ready=False, residual_study_activated=False,
        sources=[dict(path=path,sha256=digest) for path,digest in sorted(sources.items())],
        scope='Frozen fitting/validation preparation only. No residual efficacy result, no receptor parent substitution, and no test-cache evaluation.')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.json.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.rename(manifest_path)
    print(json.dumps(dict(passed=True, parents=10, split_files=len(files),
        records=sum(r['rows'] for r in files),
        maximum_cpu_gpu_prediction_difference=max(s['archived_gpu_max_prediction_delta'] for r in checks for s in r['splits'] if s['split']=='val'),
        no_cuda=True, new_test_predictions=False)))


if __name__ == '__main__':
    main()
