"""Disposable whole-optimizer and serialized-continuation checks, on CPU only."""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,random,tempfile
import numpy as np
import torch
import matched_execution as execution
import matched_models as models

HERE=Path(__file__).resolve().parent

def equal_tree(left,right):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_tree(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right):equal_tree(a,b)
    else:assert left==right,(left,right)

def near_tree(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=1e-8,atol=2e-10)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:near_tree(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right):near_tree(a,b)
    else:assert left==right

def fixture(n=7):
    generator=torch.Generator().manual_seed(2026090841+n)
    strings=['AX','GG:WY','ACDEFGHIK','TT:ACDX:W','K','YVV:GX','WWWW:AC']
    counts=torch.tensor([[5,3,2,4,1,3,2][i%7] for i in range(n)])
    mask=torch.arange(5)[None]<counts[:,None]
    x=torch.randn(n,5,53,dtype=torch.float64,generator=generator)*mask[:,:,None]
    contact=torch.rand(n,5,216,dtype=torch.float64,generator=generator)*mask[:,:,None]
    adj=(mask[:,:,None]&mask[:,None,:]).double()/counts[:,None,None]
    y=torch.linspace(-1.3,1.7,n,dtype=torch.float64)
    blob=execution.indexed_blob(dict(X=x,mask=mask,adj=adj,contact=contact,y=y,
        sequences=[strings[i%7] for i in range(n)],ids=[f'fixture{i}' for i in range(n)]))
    return execution.CommonStore([blob]),blob

def independent_whole_step(runtime,blob,store,mean,sd):
    model=runtime['model'];model.train();runtime['optimizer'].zero_grad(set_to_none=True)
    args=dict(x=blob['X'],mask=blob['mask'],adj=blob['adj'])
    if model.sequence is not None:args['sequence_batch']=store.collate(blob['ids'])
    if model.contact_map is not None:args['contact']=blob['contact']
    prediction=model(**args)
    target=(blob['y']-mean)/sd
    loss=((prediction-target)**2).mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
    runtime['optimizer'].step()
    return float(norm)

def run():
    output=HERE/'matched_execution_cpu_audit.json'
    if output.exists():raise FileExistsError('Preserve completed execution audit')
    execution.configure('cpu');checks=[]
    scratch=HERE/'tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='matched_execution_',dir=scratch) as directory:
        folder=Path(directory).resolve()
        assert folder.parent==scratch.resolve() and folder.is_relative_to(HERE.resolve())
        store,blob=fixture();mean,sd=execution.scalers(blob)
        for setting,head in models.configurations():
            def fresh(**changes):
                args=dict(dtype=torch.float64,epochs=2,effective_batch=5,max_records=2,data_manifest_sha256='fixture-v1')
                args.update(changes)
                return execution.make_runtime(42,setting,head,.001,**args)
            whole,micro=fresh(),fresh()
            maximum=0.;norm_delta=0.
            for _ in range(2):
                reference_norm=independent_whole_step(whole,blob,store,mean,sd)
                step=execution.effective_step(micro['model'],micro['optimizer'],blob,store,blob['ids'],mean,sd,max_records=2)
                assert step['records']==7 and step['microbatches']==4 and step['optimizer_steps']==1
                for p,q in zip(whole['model'].parameters(),micro['model'].parameters()):
                    torch.testing.assert_close(p,q,atol=2e-10,rtol=1e-8)
                    maximum=max(maximum,float((p-q).detach().abs().max()))
                near_tree(whole['optimizer'].state_dict(),micro['optimizer'].state_dict())
                norm_delta=max(norm_delta,abs(reference_norm-step['preclip_gradient_norm']))
                assert norm_delta<1e-9
            # The actual 72-record final effective batch is checked independently
            # against one full, example-mean update for every complete model.
            s72,b72=fixture(72);m72,sd72=execution.scalers(b72)
            ref72,micro72=fresh(),fresh()
            independent_whole_step(ref72,b72,s72,m72,sd72)
            tail=execution.effective_step(micro72['model'],micro72['optimizer'],b72,s72,b72['ids'],m72,sd72)
            assert tail['records']==72 and tail['microbatches']==3
            for p,q in zip(ref72['model'].parameters(),micro72['model'].parameters()):
                torch.testing.assert_close(p,q,atol=2e-10,rtol=1e-8)
            near_tree(ref72['optimizer'].state_dict(),micro72['optimizer'].state_dict())

            uninterrupted=fresh()
            for _ in range(2):execution.finish_epoch(uninterrupted,blob,blob,store,mean,sd)
            expected=execution.predict(uninterrupted['model'],blob,store,mean,sd,max_records=2)
            expected_rng=(torch.rand(4),np.random.random(4),[random.random() for _ in range(4)])
            interrupted=fresh();execution.finish_epoch(interrupted,blob,blob,store,mean,sd)
            saved=folder/(setting+'_'+head+'.pt');execution.save_snapshot(saved,interrupted,mean,sd)
            torch.manual_seed(9);np.random.seed(9);random.seed(9)
            resumed=fresh();boundary=execution.restore_snapshot(saved,resumed,mean,sd)
            assert boundary=='complete epoch including its scheduled validation'
            execution.finish_epoch(resumed,blob,blob,store,mean,sd)
            actual=execution.predict(resumed['model'],blob,store,mean,sd,max_records=2)
            actual_rng=(torch.rand(4),np.random.random(4),[random.random() for _ in range(4)])
            for k in ('model','optimizer','scheduler'):equal_tree(uninterrupted[k].state_dict(),resumed[k].state_dict())
            equal_tree(uninterrupted['generator'].get_state(),resumed['generator'].get_state())
            equal_tree(uninterrupted['best_state'],resumed['best_state'])
            for a,b in zip(uninterrupted['history'],resumed['history']):
                equal_tree({k:v for k,v in a.items() if k!='completed_epoch_seconds'},
                           {k:v for k,v in b.items() if k!='completed_epoch_seconds'})
                assert b['effective_batch_records']==[5,2]
            assert resumed['optimizer_steps']==4 and resumed['scheduler'].last_epoch==2
            np.testing.assert_array_equal(expected,actual)
            torch.testing.assert_close(expected_rng[0],actual_rng[0],rtol=0,atol=0)
            np.testing.assert_array_equal(expected_rng[1],actual_rng[1]);assert expected_rng[2]==actual_rng[2]
            rejected=0
            for changes,bad_mean in [({'data_manifest_sha256':'different'},mean),({'max_records':3},mean),({},mean+.01)]:
                try:execution.restore_snapshot(saved,fresh(**changes),bad_mean,sd)
                except ValueError:rejected+=1
            assert rejected==3
            independent=execution.metric(blob['y'].numpy(),actual)
            assert abs(independent['rmse']-float(np.sqrt(np.mean((blob['y'].numpy()-actual)**2))))<1e-12
            # Evaluation partitioning can change rounding, but not input access.
            all_at_once=execution.predict(resumed['model'],blob,store,mean,sd,max_records=7)
            np.testing.assert_allclose(actual,all_at_once,atol=1e-11,rtol=1e-10)
            checks.append(dict(setting=setting,head=head,maximum_independent_update_delta=maximum,
                maximum_gradient_norm_delta=norm_delta,independent_72_record_update=True,
                exact_serialized_continuation=True,optimizer_scheduler_generator_rngs_exact=True,
                target_and_batch_identity_rejections=rejected,prediction_partition_invariance=True))
            print(json.dumps({'checked':len(checks),'total':15,'setting':setting,'head':head}),flush=True)
        # Resource rules are input-dependent and identical for every setting.
        batches=list(store.chunks(blob['ids'],max_records=7,max_contact_slots=2*5*264))
        assert [k for batch in batches for k in batch]==blob['ids']
        assert max(store.footprint(b)['contact_slots'] for b in batches)<=2*5*264
        failures=0
        for limits in [{'max_tokens':1},{'max_contact_slots':264},{'max_records':0}]:
            try:list(store.chunks(blob['ids'],**limits))
            except ValueError:failures+=1
        assert failures==3

    manifest=json.loads((execution.CACHE/'manifest.json').read_text(encoding='utf-8'))
    # Full input coverage is a read-only shape check, with no model predictions.
    real=[execution.load_split(s,manifest) for s in ('train','val','test')]
    real_store=execution.CommonStore(real);envelopes={}
    for split,data in zip(('train','val','test'),real):
        parts=list(real_store.chunks(data['ids']))
        assert [k for p in parts for k in p]==data['ids']
        footprints=[real_store.footprint(p) for p in parts]
        envelopes[split]=dict(records=len(data['ids']),manifest_order_microbatches=len(parts),
            maximum_padded_tokens=max(p['padded_tokens'] for p in footprints),
            maximum_contact_slots=max(p['contact_slots'] for p in footprints),
            maximum_singleton_tokens=max(real_store.footprint([k])['padded_tokens'] for k in data['ids']))
    assert [len(real[0]['ids'][i:i+256]) for i in range(0,len(real[0]['ids']),256)]==[256,256,256,256,72]
    paths=[Path(__file__),Path(execution.__file__),Path(models.__file__),Path(execution.previous.__file__),
        Path(models.base.__file__),HERE.parent/'sequence_encoder.py',HERE.parent/'sequence_data.py',execution.CACHE/'manifest.json']
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),configurations=15,checks=checks,
        input_envelopes=envelopes,singleton_and_malformed_limits_rejected=failures,
        source_closure=[dict(path=str(p),sha256=execution.sha(p)) for p in paths],
        gpu_accessed=False,retained_fit_started=False,real_affinity_model_predictions=False,
        qualification='Manufactured CPU optimizer and continuation checks plus input-only full-cache coverage. GPU feasibility, candidate lifecycle, final source lock and retained matched fitting remain separate.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','configurations','input_envelopes','gpu_accessed','retained_fit_started')}),flush=True)

if __name__=='__main__':run()
