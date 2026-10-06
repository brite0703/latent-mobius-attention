"""Read-only records reconciliation and complete retained-model CPU replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import hashlib
import json
import math
import platform
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import replay_helpers as h
REPORTS = HERE/'replay_reports'


def metric(truth, prediction):
    y, p = [float(v) for v in truth], [float(v) for v in prediction]
    assert len(y) == len(p) and all(math.isfinite(v) for v in y+p)
    n = len(y)
    if n == 0:
        return dict(n=0,rmse=None,mae=None,pearson=None)
    dy = [a-b for a,b in zip(y,p)]
    ym,pm = math.fsum(y)/n,math.fsum(p)/n
    yc,pc = [a-ym for a in y],[a-pm for a in p]
    vy,vp = math.fsum(a*a for a in yc),math.fsum(a*a for a in pc)
    correlation = math.fsum(a*b for a,b in zip(yc,pc))/math.sqrt(vy*vp) if n>=2 and vy>0 and vp>0 else None
    return dict(n=n,rmse=math.sqrt(math.fsum(v*v for v in dy)/n),
                mae=math.fsum(abs(v) for v in dy)/n,pearson=correlation)


def compare(actual, expected, differences, label):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),label
        for key in expected:compare(actual[key],expected[key],differences,label+'/'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),label
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,differences,label+'/'+str(i))
    elif expected is None or isinstance(expected,(str,bool,int)):
        assert actual==expected,(label,actual,expected)
    else:
        assert isinstance(actual,(float,int)) and math.isfinite(actual),label
        delta=abs(float(actual)-expected)
        differences.append(delta)
        assert delta<=1e-11,(label,actual,expected,delta)


def stats(values):
    finite=[float(v) for v in values if v is not None]
    n=len(finite);mean=math.fsum(finite)/n if n else None
    return dict(prescribed=5,n=n,missing=5-n,mean=mean,
                sample_sd=math.sqrt(math.fsum((v-mean)**2 for v in finite)/(n-1)) if n>1 else None,
                minimum=min(finite) if n else None,maximum=max(finite) if n else None)


def load_npz(path, blob):
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files)=={'ids','truth','prediction'}
        ids,y,p=saved['ids'].copy(),saved['truth'].copy(),saved['prediction'].copy()
    assert ids.shape==y.shape==p.shape==(len(blob['ids']),)
    assert np.array_equal(ids,np.asarray(blob['ids'],dtype=str))
    assert np.array_equal(y,blob['y'].numpy())
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return ids,y,p


def setup(manifest):
    root=HERE/'frozen'
    seq=root/'revision_2026/reviewer_completion_2026_09_08/receptor_context'
    pocket=seq/'pocket_reconstruction'
    sys.path.insert(0,str(seq));sys.path.insert(0,str(pocket))
    import sequence_models,sequence_execution,sequence_data,sequence_selection
    import matched_models,matched_execution,matched_selection
    result={}
    for spec in manifest['studies']:
        base=h.safe_relative(root,spec['root'])
        if spec['name']=='full_sequence':
            store=sequence_data.SequenceStore()
            data=root/'revision_2026/data/lp_pdbbind/tensors_reconstructed'
            blobs={split:sequence_execution.indexed_blob(torch.load(data/f'pdbbind_{split}.pt',map_location='cpu',weights_only=True)) for split in ('train','val','test')}
            for split,blob in blobs.items():
                assert set(blob['ids'])=={key for key,row in store.by_id.items() if row['split']==split}
            models,execution,selector=sequence_models,sequence_execution,sequence_selection
            metadata=h.read(seq/'sequence_metadata.json')
        else:
            blobs={split:matched_execution.load_split(split) for split in ('train','val','test')}
            store=matched_execution.CommonStore(list(blobs.values()))
            models,execution,selector=matched_models,matched_execution,matched_selection
            audit=h.read(pocket/'matched_inputs/audit.json')
            metadata=dict(groups=audit['test_groups'],subset_sizes=audit['test_subset_sizes'])
        idsets=[set(blob['ids']) for blob in blobs.values()]
        assert all(not a&b for i,a in enumerate(idsets) for b in idsets[i+1:])
        target=[float(v) for v in blobs['train']['y']]
        mean=math.fsum(target)/len(target)
        sd=math.sqrt(math.fsum((v-mean)**2 for v in target)/len(target))
        groups={name:set(ids) for name,ids in metadata['groups'].items()}
        assert len(groups)==9 and groups['full']==set(blobs['test']['ids'])
        for name,ids in groups.items():
            assert ids<=groups['full'] and len(ids)==metadata['subset_sizes'][name]
            assert len(ids)==len(metadata['groups'][name])
        for name in ('canonical_ligand','complete_stored_string','any_supplied_chain','both_ligand_and_any_chain'):
            other='either_ligand_or_any_supplied_chain' if name=='both_ligand_and_any_chain' else name
            assert groups[other+'_overlap_with_train_or_val']==groups['full']-groups[name+'_absent_from_train_and_val']
        assert groups['both_ligand_and_any_chain_absent_from_train_and_val']==groups['canonical_ligand_absent_from_train_and_val']&groups['any_supplied_chain_absent_from_train_and_val']
        result[spec['name']]=dict(spec=spec,base=base,models=models,execution=execution,
                                  selector=selector,blobs=blobs,store=store,groups=groups,mean=mean,sd=sd)
    return result


def records(manifest, contexts, binding):
    reports=[];differences=[]
    for name,c in contexts.items():
        spec,base=c['spec'],c['base'];prefix=spec['prefix']
        locked=h.read(base/(prefix+'selection_lock.json'))
        implementation=h.read(base/(prefix+'implementation_lock.json'))
        retained=h.read(base/(prefix+'evaluation.json'))
        summary=h.read(base/(prefix+'summary.json'))
        candidates=[];margins=[];history_rows=0
        for item in locked['candidate_records']:
            path=h.safe_relative(base,item['path']);assert h.sha(path)==item['sha256']
            row=h.read(path);assert row['status']=='valid'
            assert row['implementation_lock_sha256']==spec['implementation_lock_sha256']
            for artifact in row['artifacts']+row['attempt_records']:
                assert h.sha(h.safe_relative(base,artifact['path']))==artifact['sha256']
            assert h.sha(h.safe_relative(base,row['continuation_file']))==row['continuation_sha256']
            ids,y,p=load_npz(base/(prefix+'validation_predictions')/(row['id']+'.npz'),c['blobs']['val'])
            calculated=metric(y,p)
            compare(calculated['rmse'],row['best_validation_rmse'],differences,name+'/'+row['id']+'/validation_rmse')
            assert abs(c['mean']-row['target_mean'])<=1e-13
            assert abs(c['sd']-row['target_population_sd'])<=1e-13
            history=row['history'];history_rows+=len(history)
            assert [r['epoch'] for r in history]==list(range(1,row['epochs_completed']+1))
            scheduled=[r for r in history if 'validation_rmse' in r]
            assert [r['epoch'] for r in scheduled]==[i for i in range(1,row['epochs_completed']+1) if i==1 or i%5==0 or i==100]
            earliest=min(scheduled,key=lambda r:(r['validation_rmse'],r['epoch']))
            assert earliest['epoch']==row['best_epoch'] and earliest['validation_rmse']==row['best_validation_rmse']
            candidates.append(row)
        assert len(candidates)==spec['candidates']==len(implementation['candidates'])
        for actual,planned in zip(sorted(candidates,key=lambda r:r['id']),sorted(implementation['candidates'],key=lambda r:r['id'])):
            for field,value in planned.items():assert actual[field]==value,(name,actual['id'],field)
        selected=c['selector'].select(candidates)
        for key,value in selected.items():compare(value,locked[key],differences,name+'/selection/'+key)
        by_id={row['id']:row for row in candidates}
        for choice in locked['selections']:
            available=[by_id[key] for key in choice['candidate_ids']]
            ordered=sorted(available,key=lambda r:(r['best_validation_rmse'],r['lr']))
            assert len(ordered)==2 and ordered[0]['id']==choice['selected_id']
            margins.append(dict(configuration=choice['configuration'],seed=choice['seed'],
                                difference=ordered[1]['best_validation_rmse']-ordered[0]['best_validation_rmse']))
        nomination=[]
        for row in locked['procedure_validation']:
            values=[x['validation_rmse'] for x in locked['selections'] if x['configuration']==row['configuration']]
            assert len(values)==5
            mean=math.fsum(values)/5
            compare(mean,row['mean_selected_validation_rmse'],differences,name+'/nomination/'+row['configuration'])
            nomination.append((mean,row['configuration']))
        assert min(enumerate(nomination),key=lambda p:(p[1][0],p[0]))[1][1]==locked['nominated_procedure']
        metric_rows={};all_selected=[]
        for old in retained['rows']:
            choice=next(r for r in locked['selections'] if r['configuration']==old['configuration'] and r['seed']==old['seed'])
            for key in choice:assert old[key]==choice[key]
            row=by_id[choice['selected_id']]
            checkpoint=base/(prefix+'checkpoints')/(row['id']+'.pt')
            assert h.sha(checkpoint)==old['checkpoint_sha256']
            _,vy,vp=load_npz(base/(prefix+'validation_predictions')/(row['id']+'.npz'),c['blobs']['val'])
            compare(metric(vy,vp),old['validation'],differences,name+'/selected_validation/'+row['id'])
            ids,y,p=load_npz(h.safe_relative(base,old['test_prediction_file']),c['blobs']['test'])
            assert h.sha(h.safe_relative(base,old['test_prediction_file']))==old['test_prediction_sha256']
            for group,members in c['groups'].items():
                mask=np.asarray([key in members for key in ids])
                values=metric(y[mask],p[mask])
                compare(values,old['subsets'][group],differences,name+'/'+row['id']+'/'+group)
                metric_rows[(choice['configuration'],choice['seed'],group)]=values
            all_selected.append((choice,row,old))
        assert len(all_selected)==spec['selected']
        for aggregate in summary['procedures']:
            for key in ('rmse','mae','pearson'):
                values=[metric_rows[(aggregate['configuration'],seed,aggregate['subset'])][key] for seed in range(42,47)]
                compare(stats(values),aggregate['metrics'][key],differences,name+'/summary/'+aggregate['configuration']+'/'+key)
        pair_values={}
        for contrast in summary['paired_contrasts']:
            values=[]
            for seed in range(42,47):
                delta=math.fsum(term['weight']*metric_rows[(term['configuration'],seed,contrast['subset'])]['rmse'] for term in contrast['terms'])
                values.append(delta);pair_values[(contrast['contrast'],contrast['subset'],seed)]=delta
            compare(stats(values),contrast['difference'],differences,name+'/contrast/'+contrast['contrast'])
            for key,number in [('negative',sum(v<0 for v in values)),('zero',sum(v==0 for v in values)),('positive',sum(v>0 for v in values))]:
                assert number==contrast[key]
        for row in summary['paired_seed_differences']:
            compare(pair_values[(row['contrast'],row['subset'],row['seed'])],row['difference'],differences,name+'/paired_seed')
        c['selected']=all_selected
        reports.append(dict(study=name,candidates=len(candidates),selected=len(all_selected),
                            terminal_history_rows=history_rows,procedure_subset_summaries=len(summary['procedures']),
                            paired_contrasts=len(summary['paired_contrasts']),paired_seed_differences=len(summary['paired_seed_differences']),
                            selected_subset_metrics=len(metric_rows),nomination=locked['nominated_procedure'],
                            selection_margins=margins,split_counts={key:len(blob['ids']) for key,blob in c['blobs'].items()}))
    result=dict(completed_utc=h.now(),passed=True,binding=binding,studies=reports,
                maximum_metric_or_aggregate_difference=max(differences,default=0),
                scalar_numeric_comparisons=len(differences),
                source_note='Recorded GPU vectors remain authoritative; no model was fitted or selected anew.')
    assert sum(r['candidates'] for r in reports)==260
    assert sum(r['selected'] for r in reports)==130
    assert sum(r['procedure_subset_summaries'] for r in reports)==234
    assert sum(r['paired_contrasts'] for r in reports)==441
    h.write(REPORTS/'records.json',result)
    return result


def replay(manifest,contexts,binding):
    results=[]
    for name,c in contexts.items():
        base=c['base'];prefix=c['spec']['prefix']
        for choice,record,old in c['selected']:
            output=REPORTS/name/record['id'];output.mkdir(parents=True,exist_ok=True)
            current=output/'complete.json'
            if current.exists():
                saved=h.read(current)
                assert saved['binding']==binding
                for item in saved['prediction_files']:
                    assert h.sha(h.safe_relative(HERE,item['path']))==item['sha256']
                results.append(saved)
                continue
            trial=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            model=c['models'].build_model(choice['seed'],choice['setting'],choice['head']).cpu()
            checkpoint=base/(prefix+'checkpoints')/(record['id']+'.pt')
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            assert model.parameter_counts()==record['parameters']==old['parameters']
            before={key:value.detach().clone() for key,value in model.state_dict().items()}
            assert set(before)==set(state)
            for key,value in before.items():
                assert value.dtype==state[key].dtype and value.shape==state[key].shape and torch.equal(value,state[key])
            comparisons=[];predictions=[]
            for split in ('val','test'):
                blob=c['blobs'][split]
                original=base/(prefix+('validation_predictions' if split=='val' else 'test_predictions'))/(record['id']+'.npz')
                ids,y,reference=load_npz(original,blob)
                with torch.no_grad():
                    kwargs={'max_records':record['maximum_microbatch_records']} if name=='matched_context' else {}
                    predicted=c['execution'].predict(model,blob,c['store'],record['target_mean'],record['target_population_sd'],**kwargs)
                delta=np.abs(predicted-reference)
                tolerance=manifest['prediction_tolerance']['atol']+manifest['prediction_tolerance']['rtol']*np.abs(reference)
                valid=np.isfinite(predicted)&(delta<=tolerance)
                path=output/(split+'_'+trial+'.npz')
                np.savez_compressed(path,ids=ids,truth=y,prediction=predicted)
                predictions.append(dict(path=path.relative_to(HERE).as_posix(),sha256=h.sha(path)))
                groups={'full':set(ids)} if split=='val' else c['groups']
                subset_metrics={}
                for group,members in groups.items():
                    mask=np.asarray([key in members for key in ids])
                    subset_metrics[group]=dict(cpu=metric(y[mask],predicted[mask]),saved_gpu=metric(y[mask],reference[mask]))
                comparisons.append(dict(split=split,rows=len(ids),passed=bool(valid.all()),
                                        maximum_absolute_difference=float(delta.max()),
                                        maximum_tolerance_ratio=float((delta/tolerance).max()),
                                        exceedance_count=int((~valid).sum()),
                                        exceedance_ids=ids[~valid].tolist(),metrics=subset_metrics))
            after=model.state_dict()
            assert set(before)==set(after)
            for key in before:
                assert before[key].dtype==after[key].dtype and before[key].shape==after[key].shape and torch.equal(before[key],after[key])
            saved=dict(completed_utc=h.now(),binding=binding,study=name,id=record['id'],
                       passed=all(row['passed'] for row in comparisons),comparisons=comparisons,
                       prediction_files=predictions,checkpoint_sha256=h.sha(checkpoint),
                       complete_model_state_unchanged=True,parameters=record['parameters'])
            h.write(current,saved);results.append(saved)
            h.write(REPORTS/'progress.json',dict(updated_utc=h.now(),completed=len(results),total=130,
                                                 passed=sum(r['passed'] for r in results),last_id=record['id']))
            print(json.dumps(dict(study=name,completed=len(results),total=130,id=record['id'],passed=saved['passed'])),flush=True)
            del model,state,before,after
    assert len(results)==130
    return dict(completed_utc=h.now(),binding=binding,complete=True,passed=all(r['passed'] for r in results),
                models=len(results),prediction_arrays=sum(len(r['comparisons']) for r in results),
                scalar_predictions=sum(c['rows'] for r in results for c in r['comparisons']),
                maximum_absolute_difference=max(c['maximum_absolute_difference'] for r in results for c in r['comparisons']),
                maximum_tolerance_ratio=max(c['maximum_tolerance_ratio'] for r in results for c in r['comparisons']),
                failed_models=[r['study']+'/'+r['id'] for r in results if not r['passed']],
                per_model_records=[dict(path=(REPORTS/r['study']/r['id']/'complete.json').relative_to(HERE).as_posix(),
                                        sha256=h.sha(REPORTS/r['study']/r['id']/'complete.json')) for r in results])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['records','replay']);args=parser.parse_args()
    manifest=h.read(HERE/'bundle_manifest.json')
    integrity_before=h.verify_bundle(manifest)
    h.deny_original_workspace(manifest)
    global np,torch
    import numpy as np
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1' and not torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,
                     torch=torch.__version__,torch_configuration=torch.__config__.show(),cpu_threads=torch.get_num_threads(),
                     isolated=bool(sys.flags.isolated),no_bytecode=bool(sys.dont_write_bytecode),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    assert environment['isolated'] and environment['no_bytecode']
    binding=dict(manifest_sha256=h.sha(HERE/'bundle_manifest.json'),
                 environment_sha256=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest())
    REPORTS.mkdir(parents=True,exist_ok=True);h.write(REPORTS/'environment.json',environment)
    contexts=setup(manifest)
    checked=records(manifest,contexts,binding)
    result=replay(manifest,contexts,binding) if args.mode=='replay' else checked
    integrity_after=h.verify_bundle(manifest)
    assert integrity_before==integrity_after and not h.BLOCKED_READS
    result.update(bundle_integrity_before=integrity_before,bundle_integrity_after=integrity_after,
                  original_workspace_reads_rejected=h.BLOCKED_READS,
                  compiled_scientific_sources=sorted(h.SCIENTIFIC_COMPILES),
                  public_deposition=False,new_fitting=False)
    h.write(REPORTS/(args.mode+'_complete.json'),result)
    print(json.dumps({k:result[k] for k in ('passed','models','prediction_arrays','scalar_predictions','maximum_absolute_difference','maximum_tolerance_ratio') if k in result}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        REPORTS.mkdir(parents=True,exist_ok=True)
        h.write(REPORTS/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                dict(failed_utc=h.now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
