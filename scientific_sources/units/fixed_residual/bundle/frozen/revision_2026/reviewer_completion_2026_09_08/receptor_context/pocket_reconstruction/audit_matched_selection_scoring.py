"""Independent selection/metric fixtures and actual CPU checkpoint scoring."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import copy,json,math,tempfile
import numpy as np
import torch
import matched_execution as execution
import matched_campaign as campaign
import matched_selection as selection
import matched_scoring as scoring
import matched_models as models
from audit_matched_execution import fixture,equal_tree

HERE=Path(__file__).resolve().parent

def rejects(function):
    try:function()
    except (ValueError,KeyError,FileNotFoundError):return
    raise AssertionError('Invalid selection or scoring input accepted')

def toy_groups(ids):
    full=set(ids);absence=dict(canonical_ligand=set(ids[:5]),complete_stored_string=set(ids[1:]),
        any_supplied_chain=set(ids[2:6]),both_ligand_and_any_chain=set(ids[2:5]))
    groups={'full':full}
    for name,absent in absence.items():
        groups[name+'_absent_from_train_and_val']=absent
        opposite='either_ligand_or_any_supplied_chain' if name=='both_ligand_and_any_chain' else name
        groups[opposite+'_overlap_with_train_or_val']=full-absent
    return dict(groups={k:sorted(v) for k,v in groups.items()},subset_sizes={k:len(v) for k,v in groups.items()})

def run():
    destination=HERE/'matched_selection_scoring_cpu_audit.json'
    if destination.exists():raise FileExistsError('Preserve completed selection/scoring audit')
    execution.configure('cpu');checks=[]
    config={selection.key(*p):i for i,p in enumerate(models.configurations())}
    plan=selection.candidate_plan()
    manufactured=[p|dict(status='valid',best_epoch=1,best_validation_rmse=1+config[selection.key(p['setting'],p['head'])]/10)
        for p in plan]
    locked=selection.select(manufactured)
    assert len(locked['selections'])==75 and len(locked['contrasts'])==30
    assert all(r['learning_rate']==.0003 for r in locked['selections'])
    assert locked['nominated_procedure']==selection.key(*models.configurations()[0])
    assert selection.select(manufactured[::-1])==locked
    assert selection.select([r|dict(test_rmse=1000-i) for i,r in enumerate(manufactured)])==locked
    rejects(lambda:selection.select(manufactured[:-1]))
    rejects(lambda:selection.select(manufactured[:-1]+[manufactured[0]]))
    for change in ({'status':'running'},{'best_validation_rmse':float('nan')},{'best_validation_rmse':True},
                   {'best_epoch':0},{'lr':.1}):
        rejects(lambda change=change:selection.select([manufactured[0]|change,*manufactured[1:]]))
    failed_config=selection.key('ligand_contact','lma2')
    failure_records=[r|dict(status='failed') if (selection.key(r['setting'],r['head']),r['seed'])==(failed_config,42) else r for r in manufactured]
    failed_lock=selection.select(failure_records)
    assert next(r for r in failed_lock['procedure_validation'] if r['configuration']==failed_config)['eligible'] is False
    all_failed=selection.select([r|dict(status='failed') for r in manufactured])
    assert all_failed['nominated_procedure'] is None and all(r['status']=='all_candidates_failed' for r in all_failed['selections'])
    checks.append('All 150 candidates required; 75 choices; rate and procedure ties, missing data, invalid records and test-independent nomination checked')

    y=np.array([-2.,-1.,0.,1.,2.]);p=np.array([-1.,0.,0.,0.,1.])
    computed=scoring.independent_metric(y,p)
    np.testing.assert_allclose([computed['rmse'],computed['mae'],computed['pearson']],
        [math.sqrt(.8),.8,4/math.sqrt(20)],rtol=1e-14,atol=1e-14)
    scoring.metric_delta(computed,execution.metric(y,p))
    assert scoring.independent_metric(y,np.zeros(5))['pearson'] is None
    assert scoring.independent_metric([],[])==dict(n=0,rmse=None,mae=None,pearson=None)
    assert scoring.independent_metric([1],[2])['pearson'] is None
    rejects(lambda:scoring.arrays(['x','x'],[1,2],[1,2]))
    rejects(lambda:scoring.arrays(['x'],[float('nan')],[1]))
    rejects(lambda:scoring.arrays(['x'],[1],[float('inf')]))
    ids=[f'fixture{i}' for i in range(7)];metadata=toy_groups(ids);groups=scoring.validate_groups(metadata,ids)
    bad=copy.deepcopy(metadata);bad['groups']['canonical_ligand_overlap_with_train_or_val']=ids[:2]
    rejects(lambda:scoring.validate_groups(bad,ids))
    checks.append('Analytic metrics, undefined correlation, empty subsets, identity checks and all nine complement/intersection rules')

    truth=np.arange(-3.,4.);rows=[];known={}
    for choice in locked['selections']:
        # Validation's first nominee deliberately has the largest test error.
        offset=(15-config[choice['configuration']])**2/8+(choice['seed']-42)/64
        known[(choice['configuration'],choice['seed'])]=offset
        rows.append(choice|dict(subsets=scoring.group_metrics(ids,truth,truth+offset,groups)))
    summary=scoring.summarize(rows,locked,groups)
    assert (len(summary['procedures']),len(summary['paired_contrasts']),len(summary['paired_seed_differences']))==(135,270,1350)
    assert scoring.summarize(rows[::-1],locked,groups)==summary and summary['nominated_procedure']==locked['nominated_procedure']
    for row in summary['procedures']:
        values=[known[(row['configuration'],s)] for s in selection.SEEDS]
        np.testing.assert_allclose([row['metrics']['rmse']['mean'],row['metrics']['rmse']['sample_sd']],
            [np.mean(values),np.std(values,ddof=1)],rtol=1e-13,atol=1e-13)
    for row in summary['paired_contrasts']:
        values=[sum(t['weight']*known[(t['configuration'],s)] for t in row['terms']) for s in selection.SEEDS]
        np.testing.assert_allclose([row['difference']['mean'],row['difference']['sample_sd']],
            [np.mean(values),np.std(values,ddof=1)],rtol=1e-13,atol=1e-13)
        assert (row['negative'],row['zero'],row['positive'])==(sum(v<0 for v in values),sum(v==0 for v in values),sum(v>0 for v in values))
    failure_choices={(r['configuration'],r['seed']):r for r in failed_lock['selections']}
    failure_rows=[failure_choices[(r['configuration'],r['seed'])]|dict(subsets=None)
        if (r['configuration'],r['seed'])==(failed_config,42) else r for r in rows]
    failed=scoring.summarize(failure_rows,failed_lock,groups)
    for row in failed['paired_contrasts']:
        if any(t['configuration']==failed_config for t in row['terms']):assert row['difference']['n']==4 and row['difference']['missing']==1
    rejects(lambda:scoring.summarize(rows[:-1],locked,groups))
    rejects(lambda:scoring.summarize(rows[:-1]+[rows[0]],locked,groups))
    changed=copy.deepcopy(rows);changed[0]['subsets']['full']['n']-=1
    rejects(lambda:scoring.summarize(changed,locked,groups))
    checks.append('All 135 procedure/subset rows, 270 signed contrasts and 1,350 paired seed rows independently reconciled; missing seeds remain visible')

    scratch=HERE/'tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='matched_scoring_',dir=scratch) as directory:
        folder=Path(directory).resolve()
        assert folder.parent==scratch.resolve() and folder.is_relative_to(HERE.resolve())
        store,data=fixture();mean,sd=execution.scalers(data)
        for setting,head in [('ligand_only','cp_pool'),('ligand_sequence','lma2'),('ligand_contact','lma2')]:
            candidate=next(r for r in plan if (r['setting'],r['head'],r['seed'],r['lr_index'])==(setting,head,42,0))
            record=campaign.fit_candidate(candidate,data,data,store,mean,sd,'0'*64,root=folder,
                device='cpu',epoch_limit=2,data_hash='fixture',effective_batch=5,max_records=2)
            choice=dict(configuration=selection.key(setting,head),setting=setting,head=head,seed=42,status='valid',
                selected_id=candidate['id'],validation_rmse=record['best_validation_rmse'],best_epoch=record['best_epoch'],learning_rate=candidate['lr'])
            with patch.object(scoring,'HERE',folder):
                observed=scoring.score_selected(choice,record,data,data,store,groups,device='cpu')
                assert scoring.score_selected(choice,record,data,data,store,groups,device='cpu',retain=False)==observed
                runtime=execution.make_runtime(42,setting,head,candidate['lr'],device='cpu')
                runtime['model'].load_state_dict(torch.load(folder/'checkpoints'/(candidate['id']+'.pt'),map_location='cpu',weights_only=True))
                runtime['model'].eval();manual=[]
                with torch.no_grad():
                    for chunk in store.chunks(data['ids'],max_records=2):
                        arguments,_=execution.collate(runtime['model'],data,store,chunk)
                        manual.extend((runtime['model'](**arguments).double()*sd+mean).tolist())
                saved=scoring.read_predictions(folder/observed['test_prediction_file'],data)
                np.testing.assert_array_equal(np.asarray(manual),saved[2])
                rejects(lambda:scoring.score_selected(choice|dict(validation_rmse=choice['validation_rmse']+.1),
                    record,data,data,store,groups,device='cpu',retain=False))
                initial=execution.to_cpu(runtime['model'].state_dict());optim=execution.to_cpu(runtime['optimizer'].state_dict())
                for mode in ('forward','training_step'):
                    profile=scoring.profile_workload(runtime,data,store,mean,sd,7,mode,repeats=2)
                    assert profile['checkpoint_restored'] and len(profile['seconds'])==2
                    equal_tree(initial,runtime['model'].state_dict());equal_tree(optim,runtime['optimizer'].state_dict())
                after=execution.predict(runtime['model'],data,store,mean,sd,max_records=2)
                np.testing.assert_array_equal(after,saved[2])
        checks.append('Actual CPU candidate checkpoint scoring in all three settings; independent inverse scaling and repeated file reload; profiling restores optimizer, weights and predictions')
        with (patch.object(scoring,'HERE',folder/'guard'),patch.object(scoring,'selection_context',side_effect=ValueError('Incomplete choices')),
             patch.object(execution,'load_split',side_effect=AssertionError('Test loaded before the selection gate')),
             patch.object(scoring,'gpu_start',side_effect=AssertionError('GPU touched before the selection gate'))):
            rejects(scoring.evaluate)
        checks.append('The evaluation entry point rejects incomplete selection before loading test tensors or starting CUDA')

    real_metadata=scoring.load_metadata()
    scoring.validate_groups(real_metadata,real_metadata['groups']['full'])
    assert real_metadata['subset_sizes']['full']==366 and real_metadata['subset_sizes']['both_ligand_and_any_chain_absent_from_train_and_val']==332
    paths=[Path(__file__),Path(campaign.__file__),Path(execution.__file__),Path(selection.__file__),Path(scoring.__file__),
        HERE/'matched_models.py',HERE/'audit_matched_execution.py',HERE/'matched_inputs/audit.json']
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks),
        candidates=150,selections=75,procedure_subset_rows=135,contrast_subset_rows=270,paired_seed_rows=1350,
        source_closure=[dict(path=str(p),sha256=campaign.sha(p)) for p in paths],
        retained_affinity_fit=False,real_test_predictions=False,gpu_accessed=False,
        qualification='Manufactured selection/metric fixtures and actual CPU checkpoint lifecycle; no matched-cohort performance conclusion.')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    panel=dict(created_utc=utc_now(),candidates=plan,contrasts=selection.planned_contrasts(),primary_contrast='ligand_contact_lma2_minus_lma1',
        source_sha256=campaign.sha(Path(selection.__file__)),outcomes_used=False,retained_source_lock_pending=True)
    campaign.write_json(HERE/'matched_candidate_plan.json',panel,immutable=True)
    print(json.dumps({k:result[k] for k in ('passed','check_count','candidates','selections','procedure_subset_rows','contrast_subset_rows','paired_seed_rows')}),flush=True)

def utc_now():return datetime.now(timezone.utc).isoformat()
if __name__=='__main__':run()
