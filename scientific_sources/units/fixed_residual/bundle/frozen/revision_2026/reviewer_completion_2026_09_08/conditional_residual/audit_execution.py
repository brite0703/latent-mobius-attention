"""Independent manufactured CPU execution and selection checks; no dataset I/O."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import torch
import execution as engine
import selection

HERE = Path(__file__).resolve().parent


def equal_tree(a, b, exact=True):
    if isinstance(a, torch.Tensor):
        if exact:
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=5e-10, atol=3e-12)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key], exact)
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y, exact)
    elif isinstance(a, float) and not exact:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=3e-12), (a, b)
    else:
        assert a == b, (a, b)


def rejected(call, kind=ValueError):
    try:
        call()
    except kind:
        return
    raise AssertionError("Invalid operation was not rejected")


def fixture(domain, n=19, nval=7, *, perfect_val=False, dtype=torch.float64):
    d = 12 if domain == 'cubic' else 8
    generator = torch.Generator().manual_seed(2026090845+d)
    caches = []
    for split, size in [('train', n), ('val', nval)]:
        f0 = torch.randn(size, generator=generator, dtype=dtype)*.1
        z = torch.randn(size, 8, d, generator=generator, dtype=dtype)*.15
        original = f0.double()*1.7+2.3
        if split == 'train' or not perfect_val:
            original += .03+.05*z[:, 0, 0].double()+.02*z[:, 1, 1].double().square()
        ids = [f'{split}_{i}' for i in range(size)]
        caches.append(engine.make_cache(split, ids, f0, z, original, 2.3, 1.7, 'a'*64))
    return caches


def spec_for(domain, arm, rate_index=1):
    return next(row for row in selection.candidate_plan() if row['domain'] == domain and row['arm'] == arm
                and row['rate_index'] == rate_index and row['seed'] == (100 if domain == 'cubic' else 42))


def independent_epoch(model, optimizer, cache, order, *, wrong_tail=False):
    # Independently specify the loss in the baseline's target units and form
    # each actual batch. This does not call the engine's update helper.
    sizes, total, largest_norm = [], 0., 0.
    for start in range(0, len(order), 256):
        chosen = order[start:start+256]
        optimizer.zero_grad(set_to_none=True)
        predicted = cache.f0[chosen]+model(cache.z[chosen])
        target = ((cache.truth[chosen]-2.3)/1.7).to(predicted.dtype)
        denominator = 256 if wrong_tail else len(chosen)
        loss = ((predicted-target)**2).sum()/denominator
        total += float(loss.detach())*len(chosen)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        largest_norm = max(largest_norm, float(norm))
        optimizer.step()
        sizes.append(len(chosen))
    return sizes, total/len(order), largest_norm


def actual_batch_and_continuation():
    rows = []
    for domain, n in [('ligand_contact', 1096), ('cubic', 2048)]:
        train, val = fixture(domain, n, 23)
        original_train, original_val = train.fingerprint(), val.fingerprint()
        for arm in ('product', 'additive', 'pair_mlp'):
            spec = spec_for(domain, arm)
            runtime = engine.make_runtime(spec, train, val, epoch_limit=2)
            initial = engine.export_runtime(runtime)
            reference = deepcopy(runtime['model'])
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.0001)
            reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference_optimizer, T_max=2)
            order_generator = torch.Generator().manual_seed(71000+spec['seed'])
            order = torch.randperm(n, generator=order_generator)
            sizes, manual_loss, largest_norm = independent_epoch(reference, reference_optimizer, train, order)
            reference_scheduler.step()
            actual = engine.train_epoch(runtime, train, val)
            expected_sizes = [256]*4+[72] if domain == 'ligand_contact' else [256]*8
            assert sizes == actual['batch_sizes'] == expected_sizes
            assert abs(actual['training_mse_std']-manual_loss) < 1e-12
            assert actual['order_sha256'] == hashlib.sha256(order.numpy().tobytes()).hexdigest()
            equal_tree(runtime['model'].state_dict(), reference.state_dict(), exact=False)
            equal_tree(runtime['optimizer'].state_dict(), reference_optimizer.state_dict(), exact=False)
            equal_tree(runtime['scheduler'].state_dict(), reference_scheduler.state_dict())
            delta = max(float((a-b).abs().max()) for a, b in zip(runtime['model'].parameters(), reference.parameters()))
            wrong_delta = None
            if domain == 'ligand_contact':
                # The check must distinguish the common divide-by-256 error.
                wrong = deepcopy(reference)
                wrong.load_state_dict(initial['model_state'])
                wrong_optimizer = torch.optim.AdamW(wrong.parameters(), lr=.001, weight_decay=.0001)
                independent_epoch(wrong, wrong_optimizer, train, order, wrong_tail=True)
                wrong_delta = max(float((a-b).abs().max()) for a, b in zip(runtime['model'].parameters(), wrong.parameters()))
                assert wrong_delta > 1e-6, wrong_delta
            stream = io.BytesIO()
            torch.save(engine.export_runtime(runtime), stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=True, map_location='cpu')
            restored = engine.restore_runtime(saved, train, val)
            equal_tree(engine.export_runtime(runtime), engine.export_runtime(restored))
            engine.train_epoch(runtime, train, val)
            engine.train_epoch(restored, train, val)
            equal_tree(engine.export_runtime(runtime), engine.export_runtime(restored))
            predicted_a = engine.predict(runtime['model'], val)
            predicted_b = engine.predict(restored['model'], val)
            equal_tree(predicted_a, predicted_b)
            assert train.fingerprint() == original_train and val.fingerprint() == original_val
            record = engine.completed_record(runtime)
            assert record['completed_epochs'] == 2 and record['optimizer_updates'] == 2*len(sizes)
            rejected(lambda: engine.train_epoch(runtime, train, val))
            rows.append(dict(domain=domain, arm=arm, fitting_records=n, exact_batch_sizes=sizes,
                independent_update_max_parameter_delta=delta, wrong_tail_divisor_parameter_delta=wrong_delta,
                independent_gradient_norm_maximum=largest_norm,
                exact_serialized_full_epoch_continuation=True, exact_cache_preservation=True,
                optimizer_updates=record['optimizer_updates']))
    return rows


def selection_and_epoch_zero():
    rows = []
    for arm in ('product', 'additive', 'pair_mlp'):
        train, val = fixture('ligand_contact', n=11, nval=7, perfect_val=True)
        runtime = engine.make_runtime(spec_for('ligand_contact', arm), train, val,
                                      epoch_limit=7, batch_size=4, patience=2)
        initial_state = engine.tensor_copy(runtime['model'].state_dict())
        assert runtime['best_score'] == 0.
        while runtime['termination'] is None:
            engine.train_epoch(runtime, train, val)
        assert runtime['completed_epochs'] == 5 and runtime['termination'] == 'early_stopped'
        assert [r['epoch'] for r in runtime['history']] == [0, 1, 5]
        assert runtime['best_epoch'] == 0 and runtime['best_score'] == 0.
        assert torch.equal(runtime['best_prediction_std'], val.f0)
        equal_tree(runtime['best_state'], initial_state)
        assert any(not torch.equal(initial_state[k], runtime['model'].state_dict()[k]) for k in initial_state)
        record = engine.completed_record(runtime)
        assert record['best_epoch'] == 0 and record['status'] == 'valid'
        # The best state is a clone, not an alias of the trainable final state.
        with torch.no_grad():
            runtime['model'].output.bias.add_(19.)
        equal_tree(runtime['best_state'], initial_state)
        rows.append(dict(arm=arm, selected_epoch=0, exact_baseline_prediction=True,
                         early_stop_epoch=5, best_checkpoint_is_independent_clone=True))
    plan = selection.candidate_plan()
    assert len(plan) == 90 and len({r['id'] for r in plan}) == 90
    records = [dict(spec=deepcopy(spec), status='valid', parent_checkpoint_sha256='a'*64,
        best_validation_mse=1., epoch_zero_validation_mse=1., best_epoch=0, completed_epochs=5) for spec in plan]
    choices = selection.select_all(records)
    assert len(choices) == 45 and all(c['selected_id'].endswith('lr0') and c['outcome'] == 'selected_epoch_zero' for c in choices)
    # Arbitrarily favorable or adverse test fields have no effect on selection.
    changed = deepcopy(records)
    for row in changed:
        row['test_mse'] = -1e12 if row['spec']['rate_index'] == 1 else 1e12
        row['test_rmse'] = float('nan')
    assert selection.select_all(changed) == choices
    failed = deepcopy(records)
    target = [r for r in failed if r['spec']['domain'] == 'cubic' and r['spec']['seed'] == 100 and r['spec']['arm'] == 'product']
    for r in target:
        r.update(status='failed_numerical', best_validation_mse=-1e12)
    chosen = next(c for c in selection.select_all(failed) if c['domain'] == 'cubic' and c['seed'] == 100 and c['arm'] == 'product')
    assert chosen['selected_id'] is None and chosen['outcome'] == 'no_valid_residual'
    target[0]['status'] = 'valid'; target[0]['best_validation_mse'] = .9
    chosen = next(c for c in selection.select_all(failed) if c['domain'] == 'cubic' and c['seed'] == 100 and c['arm'] == 'product')
    assert chosen['selected_id'] == target[0]['spec']['id']
    missing = deepcopy(records)
    for row in missing:
        if row['spec']['domain'] == 'cubic' and row['spec']['seed'] == 100:
            row['status'] = 'missing_parent'
            row.pop('parent_checkpoint_sha256')
    assert sum(c['outcome'] == 'missing_parent' for c in selection.select_all(missing)) == 3
    partial_parent = deepcopy(missing)
    first = next(r for r in partial_parent if r['status'] == 'missing_parent')
    first.update(status='valid', parent_checkpoint_sha256='a'*64)
    rejected(lambda: selection.select_all(partial_parent))
    rejected(lambda: selection.select_all(records[:-1]))
    rejected(lambda: selection.select_all(records+[records[0]]))
    tampered = deepcopy(records); tampered[0]['spec']['rate'] = .1
    rejected(lambda: selection.select_all(tampered))
    tampered = deepcopy(records); tampered[0]['parent_checkpoint_sha256'] = 'b'*64
    rejected(lambda: selection.select_all(tampered))
    tampered = deepcopy(records); tampered[0]['status'] = 'running'
    rejected(lambda: selection.select_all(tampered))
    return dict(epoch_zero_cases=rows, candidate_count=90, choices=45,
        rate_ties_prefer_zero=True, test_fields_do_not_change_choices=True,
        failed_trajectories_never_win=True, both_failed_stays_missing=True,
        missing_parent_applies_to_all_six_candidates=True,
        incomplete_duplicate_changed_spec_parent_and_running_records_rejected=True)


def cache_and_failure_guards():
    train, val = fixture('cubic')
    before = train.fingerprint()
    copied = engine.make_cache('train', train.ids, train.f0, train.z, train.truth,
                               train.target_mean, train.target_sd, train.parent_checkpoint_sha256)
    assert copied.f0.data_ptr() != train.f0.data_ptr() and copied.z.data_ptr() != train.z.data_ptr()
    copied.z[0, 0, 0] += .3
    rejected(copied.verify)
    assert train.fingerprint() == before
    selected = spec_for('cubic', 'product')
    runtime = engine.make_runtime(selected, train, val, epoch_limit=2)
    rejected(lambda: engine.completed_record(runtime))
    test_fixture = engine.make_cache('test', val.ids, val.f0, val.z, val.truth,
                                     val.target_mean, val.target_sd, val.parent_checkpoint_sha256)
    rejected(lambda: engine.make_runtime(selected, train, test_fixture))
    rejected(lambda: engine.validation(runtime['model'], test_fixture))
    overlap = engine.make_cache('val', train.ids[:len(val.ids)], val.f0, val.z, val.truth,
                                val.target_mean, val.target_sd, val.parent_checkpoint_sha256)
    rejected(lambda: engine.make_runtime(selected, train, overlap))
    saved = engine.export_runtime(runtime)
    corrupt_source = deepcopy(saved); corrupt_source['sources']['models.py'] = '0'*64
    rejected(lambda: engine.restore_runtime(corrupt_source, train, val))
    corrupt_cache = deepcopy(saved); corrupt_cache['cache_bindings']['val'] = '0'*64
    rejected(lambda: engine.restore_runtime(corrupt_cache, train, val))
    with torch.no_grad():
        runtime['model'].output.bias.fill_(float('inf'))
    rejected(lambda: engine.train_epoch(runtime, train, val), engine.NumericalFailure)
    rejected(lambda: engine.completed_record(runtime))
    # Every arm/rate inherits the same shuffle-generator state per parent seed.
    generator_states = {}
    for spec in selection.candidate_plan():
        a, b = fixture(spec['domain'], n=3, nval=2, dtype=torch.float32)
        trial = engine.make_runtime(spec, a, b, epoch_limit=1)
        key = spec['domain'], spec['seed']
        value = trial['generator'].get_state()
        if key in generator_states:
            assert torch.equal(generator_states[key], value)
        else:
            generator_states[key] = value.clone()
    return dict(cache_storage_is_independent=True, cache_mutation_rejected=True,
                test_cache_rejected_for_fitting_and_selection=True, overlapping_ids_rejected=True,
                source_or_cache_changed_resume_rejected=True, numerical_failure_not_valid_completion=True,
                paired_initial_shuffle_states=90, parent_seed_groups=len(generator_states))


def main():
    destination = HERE/'execution_cpu_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve the completed preparation receipt')
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    update_rows = actual_batch_and_continuation()
    selection_checks = selection_and_epoch_zero()
    guards = cache_and_failure_guards()
    assert not torch.cuda.is_initialized()
    files = ['models.py', 'selection.py', 'execution.py', 'audit_execution.py',
             'protocol_draft.md', 'protocol_clarifications.md']
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version, torch_version=str(torch.__version__), cpu_threads=1,
        cuda_initialized=False, real_data_or_retained_parent_loaded=False,
        actual_size_update_and_continuation_cases=update_rows, selection_checks=selection_checks,
        cache_and_failure_guards=guards,
        sources=[dict(path=str(HERE/f), sha256=hashlib.sha256((HERE/f).read_bytes()).hexdigest()) for f in files],
        scope='Manufactured CPU execution and pure selection checks. Actual parent/data provenance, real caching, '
              'file-backed candidate lifecycle, test metrics, profiling, activation and a retained-run lock remain outstanding.')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, update_cases=len(update_rows), candidates=90, choices=45,
        maximum_parameter_difference=max(r['independent_update_max_parameter_delta'] for r in update_rows),
        minimum_wrong_divisor_difference=min(r['wrong_tail_divisor_parameter_delta'] for r in update_rows
                                             if r['wrong_tail_divisor_parameter_delta'] is not None),
        no_cuda=True, no_real_data=True)))


if __name__ == '__main__':
    main()
