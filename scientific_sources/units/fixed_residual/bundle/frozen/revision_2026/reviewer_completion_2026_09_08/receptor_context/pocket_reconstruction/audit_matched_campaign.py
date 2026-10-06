"""Manufactured complete-candidate recovery, failures and checkpoint selection."""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,tempfile
from unittest.mock import patch
import numpy as np
import torch
import matched_execution as execution
import matched_campaign as campaign
import matched_selection as selection
from audit_matched_execution import fixture,equal_tree

HERE=Path(__file__).resolve().parent
class Interrupted(BaseException):pass

def rejects(function):
    try:function()
    except (ValueError,FileNotFoundError,FileExistsError):return
    raise AssertionError('Invalid lifecycle input accepted')

def run():
    destination=HERE/'matched_campaign_cpu_audit.json'
    if destination.exists():raise FileExistsError('Preserve completed lifecycle audit')
    execution.configure('cpu');scratch=HERE/'tmp';scratch.mkdir(exist_ok=True);checks=[]
    with tempfile.TemporaryDirectory(prefix='matched_campaign_',dir=scratch) as directory:
        folder=Path(directory).resolve()
        assert folder.parent==scratch.resolve() and folder.is_relative_to(HERE.resolve())
        store,data=fixture();mean,sd=execution.scalers(data)
        plan=next(r for r in selection.candidate_plan() if (r['setting'],r['head'],r['seed'],r['lr_index'])==('ligand_contact','lma2',42,0))
        lock_sha='0'*64
        def fit(root):return campaign.fit_candidate(plan,data,data,store,mean,sd,lock_sha,root=root,device='cpu',
            epoch_limit=2,data_hash='fixture',effective_batch=5,max_records=2)
        complete_root=folder/'complete';complete=fit(complete_root)
        assert complete['status']=='valid' and complete['epochs_completed']==2 and complete['optimizer_steps']==4
        checkpoint=Path('checkpoints')/(plan['id']+'.pt')
        predictions=Path('validation_predictions')/(plan['id']+'.npz')
        with np.load(complete_root/predictions,allow_pickle=False) as v:
            expected_prediction=v['prediction'].copy();truth=v['truth'].copy()
        assert abs(np.sqrt(np.mean((expected_prediction-truth)**2))-complete['best_validation_rmse'])<1e-12
        checks.append('Real two-epoch contact candidate, independent validation RMSE and retained best checkpoint')
        with patch.object(execution,'finish_epoch',side_effect=AssertionError('Completed candidate refitted')):
            assert fit(complete_root)==complete
        assert len(list((complete_root/'attempts'/plan['id']).glob('attempt_*.json')))==1
        changed=copy.deepcopy(complete);changed['artifacts'][0]['sha256']='f'*64
        rejects(lambda:campaign.verify_candidate(changed,plan,lock_sha,complete_root))
        checks.append('Completed candidate preserved without refitting; changed artifact rejected')

        # Interrupt after computing epoch two, before its snapshot is committed.
        finish=execution.finish_epoch
        def interrupted_epoch(runtime,*args,**kwargs):
            row=finish(runtime,*args,**kwargs)
            if runtime['epochs_completed']==2:raise Interrupted('Discarded uncommitted epoch tail')
            return row
        resumed_root=folder/'resumed'
        try:
            with patch.object(execution,'finish_epoch',side_effect=interrupted_epoch):fit(resumed_root)
        except Interrupted:pass
        else:raise AssertionError('Interruption fixture failed')
        assert not (resumed_root/'candidates'/(plan['id']+'.json')).exists()
        snapshot=torch.load(resumed_root/'continuation'/(plan['id']+'.pt'),map_location='cpu',weights_only=True)
        assert snapshot['bookkeeping']['epochs_completed']==1
        resumed=fit(resumed_root)
        equal_tree(torch.load(complete_root/checkpoint,map_location='cpu',weights_only=True),
                   torch.load(resumed_root/checkpoint,map_location='cpu',weights_only=True))
        with np.load(resumed_root/predictions,allow_pickle=False) as v:np.testing.assert_array_equal(v['prediction'],expected_prediction)
        assert len(resumed['attempt_records'])==2 and resumed['optimizer_steps']==4
        attempts=[campaign.read(resumed_root/a['path']) for a in resumed['attempt_records']]
        assert 'finished_utc' not in attempts[0] and attempts[1]['resumed_from_epoch']==1
        for a,b in zip(complete['history'],resumed['history']):
            equal_tree({k:v for k,v in a.items() if k!='completed_epoch_seconds'},
                       {k:v for k,v in b.items() if k!='completed_epoch_seconds'})
        checks.append('Actual interruption discards the uncommitted tail, preserves the old attempt and resumes with exact final state/predictions')

        for name,exception in [('numerical',FloatingPointError('manufactured nonfinite loss')),
                               ('resource',torch.cuda.OutOfMemoryError('manufactured resource failure; no CUDA allocation'))]:
            root=folder/name
            with patch.object(execution,'finish_epoch',side_effect=exception):failed=fit(root)
            assert failed['status']=='failed' and failed['stopping_reason']=='numerical_or_resource_failure' and not failed['artifacts']
            with patch.object(execution,'finish_epoch',side_effect=AssertionError('Failed candidate restarted')):
                assert fit(root)==failed
        checks.append('Numerical and resource failures remain terminal and cannot be replaced by fresh fits')
        with patch.object(execution,'finish_epoch',side_effect=RuntimeError('manufactured implementation error')):
            try:fit(folder/'error')
            except RuntimeError:pass
            else:raise AssertionError('Programming error incorrectly treated as a candidate result')
        assert not list((folder/'error'/'candidates').glob('*.json'))
        error_attempt=campaign.read(next((folder/'error'/'attempts'/plan['id']).glob('*.json')))
        assert error_attempt['status']=='execution_error'
        checks.append('Implementation errors stop execution separately from prescribed scientific failure outcomes')

        # A genuine parameterized model with an exactly constant validation score
        # checks tie handling through finish_epoch, without modifying its logic.
        runtime=execution.make_runtime(42,'ligand_contact','lma2',.001,epochs=2,effective_batch=5,max_records=2)
        fixed=np.zeros(len(data['ids']))
        with patch.object(execution,'predict',return_value=fixed):
            execution.finish_epoch(runtime,data,data,store,mean,sd)
            first=copy.deepcopy(runtime['best_state'])
            execution.finish_epoch(runtime,data,data,store,mean,sd)
        assert runtime['best_epoch']==1 and runtime['bad_checks']==1
        equal_tree(first,runtime['best_state'])
        checks.append('An exact validation tie retains the earliest actual checkpoint')
    paths=[Path(__file__),Path(campaign.__file__),Path(execution.__file__),Path(selection.__file__),
        HERE/'audit_matched_execution.py',HERE/'matched_models.py',HERE.parent/'sequence_campaign.py',HERE.parent/'sequence_execution.py']
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks),
        source_closure=[dict(path=str(p),sha256=campaign.sha(p)) for p in paths],
        gpu_accessed=False,retained_affinity_fit=False,
        qualification='Manufactured CPU lifecycle and fault injection. The actual source lock, GPU checks and all real-data fitting remain separate.')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','check_count','gpu_accessed','retained_affinity_fit')}),flush=True)

if __name__=='__main__':run()
