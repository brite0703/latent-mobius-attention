"""Read-only checkpoint diagnosis using original cubic fitting records only."""
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from affine_solver import solve

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent/'first_cubic'
HEADS = ('lma1', 'lma2', 'additive2', 'cp_pool')
SEEDS = tuple(range(100, 110))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verify(bound):
    if sha(bound['path']) != bound['sha256']:
        raise ValueError('A bound source or artifact has changed: '+bound['path'])


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('This diagnostic does not replace an existing artifact: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_models():
    implementation_path, selection_path = STUDY/'implementation_lock.json', STUDY/'selection_lock.json'
    implementation, selection = read(implementation_path), read(selection_path)
    lock_sha = sha(implementation_path)
    if selection['implementation_lock_sha256'] != lock_sha:
        raise ValueError('The original selection has a different implementation')
    sources = [artifact(implementation_path), artifact(selection_path)]
    for bound in implementation['files']:
        if Path(bound['path']).suffix == '.py':
            verify(bound)
            sources.append(bound)
    candidate_bindings = {(STUDY/b['path']).resolve(): b['sha256'] for b in selection['candidate_records']}
    parents = []
    for head in HEADS:
        for seed in SEEDS:
            matches = [r for r in selection['selections'] if (r['task'], r['head'], r['seed']) == ('cubic', head, seed)]
            if len(matches) != 1:
                raise ValueError('Missing or duplicated original selection')
            choice = matches[0]
            ids = [f'cubic_{head}_seed{seed}_lr{i}' for i in range(2)]
            if len(choice['candidate_ids']) != 2 or set(choice['candidate_ids']) != set(ids):
                raise ValueError('The original two-rate comparison has changed')
            candidates = []
            for i, identifier in enumerate(ids):
                path = STUDY/'candidates'/f'{identifier}.json'
                bound = artifact(path)
                if candidate_bindings.get(path.resolve()) != bound['sha256']:
                    raise ValueError('Candidate record differs from original selection')
                row = read(path)
                expected = dict(id=identifier, task='cubic', head=head, seed=seed, lr=(.0003, .001)[i], lr_index=i)
                if row['spec'] != expected or row['implementation_lock_sha256'] != lock_sha or row['status'] not in ('valid', 'failed'):
                    raise ValueError('Incomplete or substituted original candidate')
                if row['status'] == 'valid' and not (math.isfinite(row['best_validation_loss']) and row['best_validation_loss'] >= 0):
                    raise ValueError('Invalid original selection metric')
                candidates.append(row)
                sources.append(bound)
            valid = [r for r in candidates if r['status'] == 'valid']
            best = min(valid, key=lambda r: (r['best_validation_loss'], r['spec']['lr_index'])) if valid else None
            if choice['selected_id'] != (best['spec']['id'] if best else None):
                raise ValueError('The frozen choice is not the original validation minimum')
            if best is None:
                parents.append(dict(head=head, seed=seed, status='missing_original_selection'))
                continue
            if choice['validation_loss'] != best['best_validation_loss']:
                raise ValueError('The original selected metric changed')
            path = (STUDY/best['checkpoint']).resolve()
            if path != (STUDY/'checkpoints'/f"{best['spec']['id']}.pt").resolve():
                raise ValueError('Unexpected original checkpoint path')
            bound = artifact(path)
            if bound['sha256'] != best['checkpoint_sha256'] or bound['sha256'] != choice['checkpoint_sha256']:
                raise ValueError('Original checkpoint hash mismatch')
            sources.append(bound)
            parents.append(dict(head=head, seed=seed, status='available', candidate=best, checkpoint=bound))
    for seed in SEEDS:
        path = STUDY/'data'/f'seed{seed}.npz'
        matches = [r for r in implementation['files'] if Path(r['path']).resolve() == path.resolve()]
        if len(matches) != 1:
            raise ValueError('Original fitting archive is not bound')
        verify(matches[0])
        sources.append(matches[0])
    return parents, sources, lock_sha


def fitting_input(seed, generator):
    # Open only the fitting ID member, never the target/full-population members.
    with np.load(STUDY/'data'/f'seed{seed}.npz') as archive:
        ids = archive['train_ids'].copy()
    np.testing.assert_array_equal(ids, generator.split(seed)['train'])
    if ids.shape != (2048,) or len(set(ids.tolist())) != 2048:
        raise ValueError('The fitting cohort changed')
    bits = ((ids[:, None] >> np.arange(12)) & 1).astype(np.uint8)
    supports, signs = (7,11,19,97,161,1792,2816,1092), (1,-1,1,1,-1,1,-1,1)
    numerator = np.array([sum(sign*(1 if (int(i)&support).bit_count()%2 else -1)
                             for support, sign in zip(supports, signs)) for i in ids])
    np.testing.assert_array_equal(numerator, generator.target_numerators(bits, 'cubic'))
    return ids, torch.eye(12).expand(len(ids), -1, -1), torch.as_tensor(bits.astype(bool)), numerator/math.sqrt(8)


def extract(model, final_path, x, mask):
    final = model.get_submodule(final_path)
    if not isinstance(final, torch.nn.Linear) or final.in_features != 24 or final.out_features != 1 or final.bias is None:
        raise ValueError('Unexpected native final affine layer')
    state_before = {k:v.clone() for k,v in model.state_dict().items()}
    flags_before = [(m.training, len(m._forward_pre_hooks), len(m._forward_hooks)) for m in model.modules()]
    rng_before = torch.get_rng_state().clone()
    captured, features, predictions = [], [], []
    handle = final.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    maximum_exact_reconstruction = 0.
    try:
        with torch.inference_mode():
            for start in range(0, len(x), 256):
                captured.clear()
                pred = model(x[start:start+256], mask[start:start+256])
                if len(captured) != 1 or captured[0].shape != (len(pred), 24):
                    raise ValueError('Final feature hook did not capture the actual scalar path')
                reference = F.linear(captured[0], final.weight, final.bias).squeeze(-1)
                maximum_exact_reconstruction = max(maximum_exact_reconstruction, float((reference-pred).abs().max()))
                if not torch.equal(reference, pred):
                    raise ValueError('Native affine reconstruction is not exact')
                features.append(captured[0].numpy().copy())
                predictions.append(pred.numpy().copy())
    finally:
        handle.remove()
    if not all(torch.equal(v, model.state_dict()[k]) for k,v in state_before.items()):
        raise ValueError('Native inference changed model state')
    if flags_before != [(m.training, len(m._forward_pre_hooks), len(m._forward_hooks)) for m in model.modules()]:
        raise ValueError('Native inference changed modes or left hooks')
    if not torch.equal(rng_before, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError('Native inference changed RNG or initialized CUDA')
    h = np.concatenate(features).astype(np.float64)
    a = np.column_stack([h, np.ones(len(h))])
    beta0 = np.r_[final.weight.detach().numpy().reshape(-1), final.bias.detach().numpy()].astype(np.float64)
    return a, beta0, np.concatenate(predictions), maximum_exact_reconstruction


def aggregate(rows):
    measures = ('native_float32_mse', 'native_affine_mse', 'ols_mse', 'constrained_mse')
    summary, paired = {}, []
    for head in HEADS:
        present = [r for r in rows if r['head'] == head and r['status'] == 'evaluated']
        summary[head] = dict(evaluated=len(present), expected=10)
        if len(present) == 10:
            summary[head]['errors'] = {key:dict(mean=float(np.mean([r[key] for r in present])),
                median=float(np.median([r[key] for r in present]))) for key in measures}
    lookup = {(r['head'], r['seed']):r for r in rows if r['status'] == 'evaluated'}
    for right in ('lma1', 'additive2', 'cp_pool'):
        for key in measures:
            differences = [dict(seed=s, difference=lookup[('lma2',s)][key]-lookup[(right,s)][key])
                           for s in SEEDS if ('lma2',s) in lookup and (right,s) in lookup]
            paired.append(dict(left='lma2', right=right, measure=key, seeds=differences,
                mean_difference=float(np.mean([r['difference'] for r in differences])) if len(differences)==10 else None,
                lower_count=sum(r['difference'] < 0 for r in differences)))
    return summary, paired


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('This diagnostic must remain on CPU')
    audit = read(HERE/'solver_audit.json')
    if not audit['passed'] or audit['scientific_features_extracted']:
        raise ValueError('Manufactured solver verification is incomplete')
    for bound in audit['sources']:
        verify(bound)
    parents, sources, lock_sha = selected_models()
    sources += [artifact(HERE/name) for name in ('protocol.md', 'web_review_47_disposition.md', 'affine_solver.py', 'audit_solver.py', 'solver_audit.json', 'run_diagnostic.py')]
    unique = {str(Path(r['path']).resolve()):r for r in sources}
    sources = list(unique.values())
    output = HERE/'output/run_v1'
    write(output/'source_lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        scope='retrospective_fitting_only', expected_models=40, records_per_model=2048,
        scientific_validation_or_test_predictions=False, neural_parameter_updates=False))
    rows, artifacts = [], []
    try:
        with torch.random.fork_rng(devices=[]):
            models = imported('affine_diagnostic_native_models', STUDY/'neural_models.py')
            generator = imported('affine_diagnostic_native_generator', STUDY/'generator.py')
            for parent in parents:
                head, seed = parent['head'], parent['seed']
                if parent['status'] != 'available':
                    rows.append(dict(head=head, seed=seed, status=parent['status']))
                    continue
                ids, x, mask, y = fitting_input(seed, generator)
                blob = torch.load(parent['checkpoint']['path'], map_location='cpu', weights_only=True)
                if set(blob) != {'state_dict','spec','implementation_lock_sha256'} or blob['spec'] != parent['candidate']['spec'] or blob['implementation_lock_sha256'] != lock_sha:
                    raise ValueError('Original checkpoint metadata mismatch')
                model = models.build(seed, head).cpu().eval()
                model.load_state_dict(blob['state_dict'], strict=True)
                path = 'head.readout.2' if head == 'cp_pool' else 'head.head.2'
                a, beta0, native, native_delta = extract(model, path, x, mask)
                result = solve(a, y, beta0)
                m = result['metrics']
                affine_delta = float(np.max(np.abs(native.astype(float)-a@beta0)))
                if affine_delta > 1e-5:
                    raise ArithmeticError('Float64 readout reconstruction exceeds the declared numerical tolerance')
                status = 'evaluated' if m['numerical_checks_passed'] else 'unresolved_numerical_check'
                identifier = parent['candidate']['spec']['id']
                cache_path = output/f'{identifier}.npz'
                np.savez_compressed(cache_path, ids=ids, a=a, truth=y, native_prediction=native,
                    beta0=beta0, beta_ols=result['beta_ols'], beta_constrained=result['beta_constrained'])
                row = dict(head=head, seed=seed, selected_id=identifier, status=status,
                    final_layer=path, checkpoint=parent['checkpoint'], fitting_arrays=artifact(cache_path),
                    native_float32_mse=float(np.mean((native.astype(float)-y)**2)),
                    native_exact_reconstruction_maximum=native_delta,
                    float64_affine_reconstruction_maximum=affine_delta, **m)
                row_path = output/f'{identifier}.json'
                write(row_path, row)
                rows.append(row)
                artifacts += [artifact(cache_path), artifact(row_path)]
                print(json.dumps(dict(completed=len(rows), total=40, head=head, seed=seed, status=status)), flush=True)
        for bound in sources:
            verify(bound)
        summary, paired = aggregate(rows)
        result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), scope='retrospective_fitting_only',
            models=rows, summaries=summary, paired_contrasts=paired, source_lock=artifact(output/'source_lock.json'),
            artifacts=artifacts, cpu_threads=torch.get_num_threads(), cuda_initialized=torch.cuda.is_initialized(),
            scientific_validation_or_test_predictions=False, neural_parameter_updates=False,
            formal_theorem_or_generalization_claim=False)
        write(output/'results.json', result)
        fields = ['head','seed','selected_id','status','native_float32_mse','native_affine_mse','ols_mse','constrained_mse',
                  'native_coefficient_norm','ols_coefficient_norm','constrained_coefficient_norm','numerical_rank','multiplier']
        with (output/'all_models.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(dict(completed=len(rows), summaries=summary, cuda_initialized=False)), flush=True)
    except Exception as exc:
        write(output/'failure.json', dict(failed_utc=datetime.now(timezone.utc).isoformat(),
            completed_models=len(rows), error_type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    run()
