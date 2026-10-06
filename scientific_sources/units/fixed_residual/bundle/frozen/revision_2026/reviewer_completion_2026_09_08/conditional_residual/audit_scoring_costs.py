"""Independent arithmetic, coverage and profiling-state checks on fixtures."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct

import mpmath as mp
import numpy as np
import scipy
import torch

import costs
import execution
from models import ARMS, PairResidual
import scoring
import selection

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def require_failure(call, expected=ValueError):
    try:
        call()
    except expected as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("The intentionally invalid scoring/profiling input was accepted")


def fixture_cache(domain, seed, split="test", count=17, *, offset=0., f0_override=None):
    d = 12 if domain == "cubic" else 8
    mean, sd = (0., 1.) if domain == "cubic" else (6.5, 1.2)
    index = seed-(100 if domain == "cubic" else 42)
    generator = torch.Generator().manual_seed(99000+seed+count)
    z = torch.randn(count, 8, d, generator=generator)
    f0 = torch.tensor([.1*((i % 4)-2)+.002*index+offset for i in range(count)], dtype=torch.float32)
    if f0_override is not None:
        f0.fill_(f0_override)
    truth = torch.tensor([mean+sd*(.4+.07*((i % 5)-2)+(.003*index if domain == "cubic" else 0.)) for i in range(count)], dtype=torch.float64)
    # Molecular seeds deliberately share the same records and targets.
    id_scope = f"cubic:{seed}" if domain == "cubic" else "molecular_common"
    ids = [f"manufactured:{id_scope}:{split}:{i}" for i in range(count)]
    parent = hashlib.sha256(f"manufactured parent {domain} {seed}".encode()).hexdigest()
    return execution.make_cache(split, ids, f0, z, truth, mean, sd, parent)


def current_spec(domain, seed, arm):
    return next(s for s in selection.candidate_plan() if (s["domain"], s["seed"], s["arm"], s["rate_index"]) == (domain, seed, arm, 1))


def choice_for(spec, epoch=1):
    ids = [s["id"] for s in sorted((r for r in selection.candidate_plan() if (r["domain"], r["seed"], r["arm"]) == (
        spec["domain"], spec["seed"], spec["arm"])), key=lambda r: r["rate_index"])]
    return dict(domain=spec["domain"], seed=spec["seed"], arm=spec["arm"], candidate_ids=ids,
                candidate_statuses=["valid", "valid"], selected_id=spec["id"], selected_epoch=epoch,
                outcome="selected_epoch_zero" if epoch == 0 else "selected_trained")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def scalar_reference(cache, bias):
    predicted = [(f32(float(value)+bias)*cache.target_sd)+cache.target_mean for value in cache.f0]
    differences = [p-float(y) for p, y in zip(predicted, cache.truth)]
    mse = math.fsum(value*value for value in differences)/len(differences)
    return dict(mse=mse, rmse=math.sqrt(mse), mae=math.fsum(abs(value) for value in differences)/len(differences)), predicted


def main():
    receipt_path = HERE / "scoring_costs_cpu_audit.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve the completed scoring/cost audit")
    directory = HERE / "discarded_scoring_cost_audit_20260909"
    if directory.exists():
        raise FileExistsError("Preserve an earlier audit attempt")
    directory.mkdir()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    outcomes, caches, checkpoint_records = [], {}, {}
    largest_metric_difference, largest_prediction_difference = 0., 0.
    for domain, seeds in selection.DOMAINS.items():
        for seed in seeds:
            cache = fixture_cache(domain, seed)
            caches[domain, seed] = cache
            row, vectors = scoring.baseline(cache, domain, seed)
            expected, predictions = scalar_reference(cache, 0.)
            assert row["status"] == "evaluated"
            for key, value in expected.items():
                largest_metric_difference = max(largest_metric_difference, abs(value-row["metrics"][key]))
                assert math.isclose(value, row["metrics"][key], rel_tol=1e-13, abs_tol=1e-13)
            assert np.array_equal(vectors["prediction_original_units"].numpy(), np.asarray(predictions))
            outcomes.append(row)
            for arm in ARMS:
                spec = current_spec(domain, seed, arm)
                epoch = 0 if domain == "cubic" and seed == 100 and arm == "product" else 1
                bias = 0. if epoch == 0 else {"product": .2, "additive": .05, "pair_mlp": .1}[arm]
                model = PairResidual(arm, cache.z.shape[-1], seed=seed)
                with torch.no_grad():
                    model.output.bias.fill_(bias)
                bias = float(model.output.bias.item())
                checkpoint_path = directory / (spec["id"]+".pt")
                torch.save(execution.tensor_copy(model.state_dict()), checkpoint_path)
                checkpoint = source(checkpoint_path)
                checkpoint_records[domain, seed, arm] = checkpoint
                original_hash = checkpoint["sha256"]
                choice = choice_for(spec, epoch)
                row, vectors = scoring.residual(cache, spec, choice, checkpoint, cache.parent_checkpoint_sha256, batch_size=7)
                assert row["status"] == "evaluated" and sha(checkpoint_path) == original_hash
                expected, predictions = scalar_reference(cache, bias)
                for key, value in expected.items():
                    largest_metric_difference = max(largest_metric_difference, abs(value-row["metrics"][key]))
                    assert math.isclose(value, row["metrics"][key], rel_tol=1e-13, abs_tol=1e-13)
                delta = float(np.max(np.abs(vectors["prediction_original_units"].numpy()-np.asarray(predictions))))
                largest_prediction_difference = max(largest_prediction_difference, delta)
                assert delta == 0.
                if epoch == 0:
                    assert torch.equal(vectors["prediction_parent_units"], cache.f0)
                    assert row["selection_outcome"] == "selected_epoch_zero"
                cache.verify()
                outcomes.append(row)
    assert len(outcomes) == 60
    aggregate = scoring.summarize(outcomes)
    assert len(aggregate["procedure_summaries"]) == 8 and len(aggregate["primary_contrasts"]) == 6
    shuffled = deepcopy(outcomes)
    random.Random(2026090901).shuffle(shuffled)
    assert scoring.summarize(shuffled) == aggregate

    mp.mp.dps = 60
    quantiles = {}
    for count in (5, 10):
        df = mp.mpf(count-1)
        quantile = mp.findroot(lambda value: 1-mp.betainc(df/2, mp.mpf('.5'), 0, df/(df+value*value), regularized=True)/2-mp.mpf('.975'), (mp.mpf(2), mp.mpf(4)))
        quantiles[count] = float(quantile)
    max_interval_difference = 0.
    by_key = {(r["domain"], r["seed"], r["procedure"]):r for r in outcomes}
    for contrast in aggregate["primary_contrasts"]:
        domain = contrast["domain"]
        comparator = contrast["contrast"].removeprefix("product_minus_")
        metric = "mse" if domain == "cubic" else "rmse"
        values = [by_key[domain, seed, "product"]["metrics"][metric]-by_key[domain, seed, comparator]["metrics"][metric] for seed in selection.DOMAINS[domain]]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = quantiles[len(values)]*sd/math.sqrt(len(values))
        expected_interval = [mean-half, mean+half]
        assert math.isclose(mean, contrast["mean"], rel_tol=1e-13, abs_tol=1e-13)
        assert math.isclose(sd, contrast["sample_sd"], rel_tol=1e-13, abs_tol=1e-13)
        difference = max(abs(a-b) for a,b in zip(expected_interval,contrast["nominal_t_reference_interval95"]))
        max_interval_difference = max(max_interval_difference, difference)
        assert difference < 1e-11
        assert contrast["lower"] == sum(value < 0 for value in values)
        assert contrast["tied"] == sum(value == 0 for value in values)
        assert contrast["higher"] == sum(value > 0 for value in values)

    incomplete = deepcopy(outcomes)
    index = next(i for i,r in enumerate(incomplete) if (r["domain"],r["seed"],r["procedure"]) == ("cubic",101,"additive"))
    incomplete[index] = scoring.unavailable("cubic",101,"additive","no_valid_residual","Manufactured two-failed-rate case",caches["cubic",101].parent_checkpoint_sha256)
    incomplete_report = scoring.summarize(incomplete)
    for contrast in incomplete_report["primary_contrasts"]:
        if contrast["domain"] == "cubic" and contrast["contrast"] == "product_minus_additive":
            assert not contrast["complete"] and contrast["available_replicates"] == 9
            assert contrast["mean"] is contrast["sample_sd"] is contrast["nominal_t_reference_interval95"] is None
        else:
            assert contrast["complete"]
    absent = deepcopy(outcomes)
    for index,row in enumerate(absent):
        if row["domain"] == "cubic" and row["seed"] == 101:
            absent[index] = scoring.unavailable("cubic",101,row["procedure"],"missing_parent","Manufactured unavailable parent")
    absent_report = scoring.summarize(absent)
    assert all(not r["complete"] for r in absent_report["primary_contrasts"] if r["domain"] == "cubic")
    assert scoring.complete_summary([0.]*5,5)["nominal_t_reference_interval95"] == [0.,0.]

    failures = []
    for label, callback in (
        ("missing outcome",lambda:scoring.summarize(outcomes[:-1])),
        ("duplicate outcome",lambda:scoring.summarize(outcomes+[outcomes[0]])),
        ("nonfinite target",lambda:scoring.metrics([float('nan')],[0.])),
        ("unaligned vector",lambda:scoring.metrics([0.,1.],[0.])),
    ):
        failures.append(dict(case=label, error=require_failure(callback)))
    for label,prediction in (("nonfinite prediction",float('inf')),("overflowed squared error",1e308)):
        failures.append(dict(case=label,error=require_failure(lambda:scoring.metrics([0.],[prediction]),scoring.EvaluationFailure)))
    mismatched = deepcopy(outcomes)
    for row in mismatched:
        if row["domain"] == "ligand_contact" and row["seed"] == 43:
            row["record_ids_sha256"] = 'f'*64
    failures.append(dict(case="changed cohort across molecular seeds",error=require_failure(lambda:scoring.summarize(mismatched))))
    mismatched = deepcopy(outcomes)
    next(r for r in mismatched if r["domain"] == "cubic" and r["procedure"] == "additive")["cache_digest"] = 'f'*64
    failures.append(dict(case="unmatched cache within paired comparison",error=require_failure(lambda:scoring.summarize(mismatched))))
    cache = caches["cubic",101]
    spec = current_spec("cubic",101,"product")
    checkpoint = checkpoint_records["cubic",101,"product"]
    choice = choice_for(spec)
    failures.append(dict(case="different parent",error=require_failure(lambda:scoring.residual(cache,spec,choice,checkpoint,'0'*64))))
    failures.append(dict(case="false epoch-zero label",error=require_failure(lambda:scoring.residual(cache,spec,choice_for(spec,0),checkpoint,cache.parent_checkpoint_sha256))))
    wrong_choice = deepcopy(choice)
    wrong_choice["selected_id"] = wrong_choice["candidate_ids"][0]
    failures.append(dict(case="unselected candidate rate",error=require_failure(lambda:scoring.residual(cache,spec,wrong_choice,checkpoint,cache.parent_checkpoint_sha256))))
    changed_cache = deepcopy(cache)
    changed_cache.f0[0] += 1
    failures.append(dict(case="changed cached prediction",error=require_failure(lambda:scoring.baseline(changed_cache,"cubic",101))))
    altered_pairs = torch.load(checkpoint["path"],weights_only=True,map_location='cpu')
    altered_pairs['pairs'][0] = torch.tensor([1,2])
    altered_path = directory/'altered_pair_enumeration.pt'
    torch.save(altered_pairs,altered_path)
    failures.append(dict(case="changed pair enumeration with a refreshed file hash",error=require_failure(lambda:scoring.residual(cache,spec,choice,source(altered_path),cache.parent_checkpoint_sha256))))
    bad = torch.load(checkpoint["path"],weights_only=True,map_location='cpu')
    bad['output.bias'][0] = float('nan')
    bad_path = directory/'invalid_parameter.pt'
    torch.save(bad,bad_path)
    failures.append(dict(case="invalid selected parameter",error=require_failure(lambda:scoring.residual(cache,spec,choice,source(bad_path),cache.parent_checkpoint_sha256))))
    overflow_cache = fixture_cache('cubic',101,f0_override=torch.finfo(torch.float32).max)
    overflow = torch.load(checkpoint["path"],weights_only=True,map_location='cpu')
    overflow['output.bias'][0] = torch.finfo(torch.float32).max
    overflow_path = directory/'finite_parameters_overflowing_prediction.pt'
    torch.save(overflow,overflow_path)
    failed_row, failed_vectors = scoring.residual(overflow_cache,spec,choice,source(overflow_path),overflow_cache.parent_checkpoint_sha256)
    assert failed_row['status']=='failed_evaluation' and failed_row['metrics'] is None and failed_vectors is None
    assert failed_row['candidate_id']==spec['id']

    update_checks = []
    for domain,seed in (('cubic',100),('ligand_contact',42)):
        for batch in ((256,) if domain=='cubic' else (256,72)):
            for arm in ARMS:
                spec = current_spec(domain,seed,arm)
                train,val = fixture_cache(domain,seed,'train',batch),fixture_cache(domain,seed,'val',17)
                runtime = execution.make_runtime(spec,train,val,epoch_limit=2,batch_size=batch)
                for epoch in (1,2):
                    model = deepcopy(runtime['model'])
                    optimizer = costs.optimizer_for(model,spec['rate'])
                    optimizer.load_state_dict(deepcopy(runtime['optimizer'].state_dict()))
                    generator = torch.Generator()
                    generator.set_state(runtime['generator'].get_state())
                    order = torch.randperm(batch,generator=generator)
                    costs.optimizer_step(model,optimizer,train.f0[order],train.z[order],train.y_std[order])
                    execution.train_epoch(runtime,train,val)
                    assert costs.same(model.state_dict(),runtime['model'].state_dict())
                    assert costs.same(optimizer.state_dict()['state'],runtime['optimizer'].state_dict()['state'])
                update_checks.append(dict(domain=domain,arm=arm,batch_size=batch,epochs_checked=2,
                                          parameters_and_adamw_tensor_states_exact=True))
    assert len(update_checks)==9
    profile_rows = []
    profile_caches = {domain:fixture_cache(domain,seed,'train',256) for domain,seed in (('cubic',100),('ligand_contact',42))}
    for workload in costs.workload_plan():
        model=PairResidual(workload['arm'],workload['dimension'],seed=100 if workload['domain']=='cubic' else 42)
        model.eval().requires_grad_(False)
        row=costs.profile(model,profile_caches[workload['domain']],workload,warmups=1,repeats=2)
        assert row['source_model_and_cache_unchanged'] and row['caller_rng_unchanged']
        assert row['tensor_storage']['parameter_bytes']==model.parameter_count()*4
        assert row['tensor_storage']['constant_buffer_bytes']==28*2*8
        assert row['tensor_storage']['cached_prediction_and_bucket_input_bytes']==workload['batch_size']*(1+8*workload['dimension'])*4
        timings=row['repetitions_ms']
        assert row['summary']['median_ms']==statistics.median(timings)
        assert math.isclose(row['summary']['sample_sd_ms'],statistics.stdev(timings),rel_tol=1e-13,abs_tol=1e-13)
        if workload['mode']=='fresh_optimizer_step':
            assert set(row['optimizer_state_entry_counts_before_repetitions'])=={0}
        if workload['mode']=='steady_optimizer_step':
            assert min(row['optimizer_state_entry_counts_before_repetitions'])>0
        profile_rows.append(row)
    assert len(profile_rows)==36
    rejected=costs.workload_plan()[0]
    failures.append(dict(case="profiling a test cache",error=require_failure(lambda:costs.profile(PairResidual('product',12),caches['cubic',100],rejected,warmups=1,repeats=2))))
    assert not torch.cuda.is_initialized()
    profile_path=directory/'discarded_profiler_observations.json'
    profile_path.write_text(json.dumps(dict(scope='Manufactured profiler verification under concurrent unrelated training; not a scientific cost result',workloads=profile_rows),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    aggregate_path=directory/'manufactured_outcome_summary.json'
    aggregate_path.write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),scored_outcome_rows=60,primary_contrasts=6,
                maximum_scalar_metric_difference=largest_metric_difference,maximum_original_unit_prediction_difference=largest_prediction_difference,
                maximum_independent_interval_endpoint_difference=max_interval_difference,
                independent_t_quantile_method='60-decimal incomplete-beta CDF inversion using mpmath, distinct from SciPy quantile call',
                missing_comparison_primary_mean_suppressed=True,all_molecular_seeds_same_cohort_checked=True,
                rejected_cases=failures,failed_numerical_evaluation_preserved=True,
                independent_profile_update_cases=update_checks,profile_workloads_checked=36,
                source_model_cache_and_rng_preserved=True,cuda_initialized=False,retained_residual_fits=0,new_scientific_test_predictions=False,
                versions=dict(torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__),
                sources=[source(HERE/name) for name in ('scoring.py','costs.py','models.py','execution.py','selection.py','scoring_cost_specification.md',Path(__file__).name)],
                artifacts=[source(profile_path),source(aggregate_path)],
                remaining='Retained activation/source-lock and common real test-access controller, matched molecular parents/caches, native end-to-end and selected-model cost evaluation, and all retained residual fitting/evaluation')
    receipt_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('passed','scored_outcome_rows','primary_contrasts','profile_workloads_checked','maximum_scalar_metric_difference','maximum_independent_interval_endpoint_difference','retained_residual_fits','new_scientific_test_predictions')}))


if __name__=='__main__':
    main()
