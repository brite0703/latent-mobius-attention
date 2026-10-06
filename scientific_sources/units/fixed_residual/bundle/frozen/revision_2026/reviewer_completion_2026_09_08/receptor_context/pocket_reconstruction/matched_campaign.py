"""Matched candidate lifecycle; retained fitting requires all prerequisite audits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,math,os,time,traceback
import torch
import matched_execution as execution
import matched_selection as selection
import matched_models as models
import sequence_campaign as predecessor

HERE=Path(__file__).resolve().parent
STUDY=HERE/'matched_study'
utc=predecessor.utc
sha=predecessor.sha
read=predecessor.read
write_json=predecessor.write_json
retain_state=predecessor.retain_state
retain_predictions=predecessor.retain_predictions
exclusive_process=predecessor.exclusive_process

def verify_candidate(row,plan,lock_sha,root=STUDY):
    if any(row.get(k)!=v for k,v in plan.items()) or row['implementation_lock_sha256']!=lock_sha:raise ValueError('Candidate or source-lock mismatch')
    if row['status'] not in ('valid','failed'):raise ValueError('Candidate is not terminal')
    for artifact in row['artifacts']+row['attempt_records']:
        path=(root/artifact['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=artifact['sha256']:raise ValueError('Preserved artifact changed')
    if sha(root/row['continuation_file'])!=row['continuation_sha256']:raise ValueError('Continuation changed')

def fit_candidate(plan,train,validation,store,mean,sd,lock_sha,*,root=STUDY,device='cuda',
                  epoch_limit=execution.EPOCHS,data_hash=None,effective_batch=execution.EFFECTIVE_BATCH,max_records=execution.MAX_RECORDS):
    root=Path(root)
    if root.resolve()==STUDY.resolve() and (epoch_limit!=100 or effective_batch!=256 or max_records!=32 or device!='cuda'):
        raise ValueError('Retained candidates require the declared CUDA schedule')
    if root.resolve()==STUDY.resolve():
        locked=read(STUDY/'implementation_lock.json')
        if sha(STUDY/'implementation_lock.json')!=lock_sha or data_hash!=locked['data_manifest_sha256'] or plan not in locked['candidates']:
            raise ValueError('Retained fitting requires the actual implementation lock and candidate identity')
    identifier=plan['id'];result_path=root/'candidates'/(identifier+'.json')
    expected_runtime=dict(epoch_limit=epoch_limit,data_manifest_sha256=data_hash,
        effective_batch=effective_batch,maximum_microbatch_records=max_records)
    if result_path.exists():
        row=read(result_path)
        if any(row.get(k)!=v for k,v in expected_runtime.items()):raise ValueError('Completed runtime differs')
        verify_candidate(row,plan,lock_sha,root)
        return row
    continuation=root/'continuation'/(identifier+'.pt')
    attempts=root/'attempts'/identifier;attempts.mkdir(parents=True,exist_ok=True)
    previous=sorted(attempts.glob('attempt_*.json'))
    runtime=execution.make_runtime(plan['seed'],plan['setting'],plan['head'],plan['lr'],device=device,epochs=epoch_limit,
        data_manifest_sha256=data_hash,effective_batch=effective_batch,max_records=max_records)
    if continuation.exists():boundary=execution.restore_snapshot(continuation,runtime,mean,sd)
    else:
        if previous:raise ValueError('Interrupted candidate lacks its recoverable initial/epoch snapshot')
        execution.save_snapshot(continuation,runtime,mean,sd)
        boundary='initialized state before the first epoch'
    attempt_path=attempts/f'attempt_{len(previous)+1:03d}.json'
    attempt=dict(started_utc=utc(),process_id=os.getpid(),candidate_id=identifier,implementation_lock_sha256=lock_sha,
        start_boundary=boundary,resumed_from_epoch=runtime['epochs_completed'],continuation_sha256_at_start=sha(continuation),
        previous_attempts=[dict(path=str(p.relative_to(root)),sha256=sha(p),finished='finished_utc' in read(p)) for p in previous],
        discarded_partial_work='Work beyond the saved complete-epoch boundary is replayed and is not reported as preserved progress.')
    write_json(attempt_path,attempt,immutable=True)
    started=time.perf_counter();status='valid';error=None;artifacts=[]
    try:
        while runtime['epochs_completed']<epoch_limit and runtime['bad_checks']<execution.PATIENCE:
            row=execution.finish_epoch(runtime,train,validation,store,mean,sd)
            execution.save_snapshot(continuation,runtime,mean,sd)
            write_json(root/'active_candidate.json',dict(updated_utc=utc(),candidate_id=identifier,completed_epoch=row['epoch'],
                best_epoch=runtime['best_epoch'],best_validation_rmse=runtime['best_validation_rmse'],attempt=len(previous)+1))
        if runtime['best_state'] is None:raise FloatingPointError('No finite validation checkpoint')
        runtime['model'].load_state_dict(runtime['best_state'],strict=True)
        prediction=execution.predict(runtime['model'],validation,store,mean,sd,**runtime['identity']['microbatch_policy'])
        checked=execution.metric(validation['y'].numpy(),prediction)['rmse']
        if checked!=runtime['best_validation_rmse']:raise RuntimeError('Best-checkpoint validation prediction changed')
        checkpoint=root/'checkpoints'/(identifier+'.pt');prediction_file=root/'validation_predictions'/(identifier+'.npz')
        retain_state(checkpoint,runtime['best_state']);retain_predictions(prediction_file,validation['ids'],validation['y'].numpy(),prediction)
        artifacts=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in (checkpoint,prediction_file)]
    except (FloatingPointError,torch.cuda.OutOfMemoryError) as exception:
        status='failed';error=dict(type=type(exception).__name__,message=str(exception),traceback=traceback.format_exc())
    except Exception:
        attempt.update(finished_utc=utc(),elapsed_seconds=time.perf_counter()-started,status='execution_error',traceback=traceback.format_exc())
        write_json(attempt_path,attempt)
        raise
    attempt.update(finished_utc=utc(),elapsed_seconds=time.perf_counter()-started,status=status,
        completed_epoch=runtime['epochs_completed'],error=error,continuation_sha256_at_finish=sha(continuation))
    write_json(attempt_path,attempt)
    row=plan|expected_runtime|dict(status=status,finished_utc=utc(),implementation_lock_sha256=lock_sha,
        best_validation_rmse=runtime['best_validation_rmse'] if math.isfinite(runtime['best_validation_rmse']) else None,
        best_epoch=runtime['best_epoch'],epochs_completed=runtime['epochs_completed'],optimizer_steps=runtime['optimizer_steps'],
        stopping_reason='numerical_or_resource_failure' if status=='failed' else 'validation_patience' if runtime['bad_checks']>=execution.PATIENCE else 'epoch_cap',
        target_mean=mean,target_population_sd=sd,parameters=runtime['model'].parameter_counts(),
        completed_epoch_seconds=math.fsum(h['completed_epoch_seconds'] for h in runtime['history']),
        history=runtime['history'],error=error,artifacts=artifacts,
        attempt_records=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in sorted(attempts.glob('attempt_*.json'))],
        continuation_file=str(continuation.relative_to(root)),continuation_sha256=sha(continuation),
        timing_scope='Completed-epoch training and scheduled validation, excluding snapshot I/O and replayed partial tails. Attempt elapsed times are separate.')
    write_json(result_path,row,immutable=True)
    del runtime
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return row

def verify_closure(record):
    for source in record:
        if sha(source['path'])!=source['sha256']:raise ValueError('Source changed: '+source['path'])

def verify_lock():
    lock=read(STUDY/'implementation_lock.json');verify_closure(lock['sources'])
    if lock['candidates']!=selection.candidate_plan() or lock['contrasts']!=selection.planned_contrasts():raise ValueError('Fixed plan changed')
    return lock

def lock():
    if (STUDY/'implementation_lock.json').exists():
        verify_lock();return
    # No CUDA fit is allowed simply because model/data preparation succeeded.
    required=('matched_models_cpu_audit.json','matched_execution_cpu_audit.json','matched_campaign_cpu_audit.json',
              'matched_selection_scoring_cpu_audit.json','matched_gpu_audit.json')
    paths=set()
    for name in required:
        path=HERE/name;receipt=read(path)
        if not receipt.get('passed'):raise ValueError('Incomplete prerequisite: '+name)
        closure=receipt.get('source_closure',receipt.get('sources',[]))
        if not closure:raise ValueError('Prerequisite lacks a source closure: '+name)
        if isinstance(closure,dict):closure=[dict(path=str(HERE/k),sha256=v) for k,v in closure.items()]
        verify_closure(closure);paths.add(path);paths.update(Path(s['path']) for s in closure)
    for path in HERE.glob('matched_*.py'):paths.add(path)
    paths.update([HERE/'matched_protocol_draft.md',execution.CACHE/'manifest.json',HERE/'matched_inputs/audit.json'])
    inputs=read(execution.CACHE/'manifest.json');verify_closure(inputs['sources'])
    paths.update(Path(s['path']) for s in inputs['sources'])
    paths.update(Path(s['path']) for s in inputs['files'])
    # The predecessor closure includes transitive LMA/GCN/CP definitions.
    prior=read(HERE.parent/'sequence_implementation_lock.json');verify_closure(prior['sources'])
    paths.update(Path(s['path']) for s in prior['sources'] if str(s['path']).endswith('.py'))
    record=dict(created_utc=utc(),candidates=selection.candidate_plan(),contrasts=selection.planned_contrasts(),
        source_scope='Audited model, execution, selection/scoring, GPU feasibility and common-input closure before retained fitting.',
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(paths)],
        target_scale=inputs['target_scale'],data_manifest_sha256=sha(execution.CACHE/'manifest.json'),
        primary_contrast='ligand_contact_lma2_minus_lma1',all_selections_before_new_test_scoring=True)
    write_json(STUDY/'implementation_lock.json',record,immutable=True)

def train():
    lock_record=verify_lock();execution.configure('cuda');lock_sha=sha(STUDY/'implementation_lock.json')
    manifest=read(execution.CACHE/'manifest.json')
    train_data=execution.load_split('train',manifest);validation=execution.load_split('val',manifest)
    store=execution.CommonStore([train_data,validation]);mean=manifest['target_scale']['mean'];sd=manifest['target_scale']['population_sd']
    with exclusive_process(STUDY):
        rows=[]
        for plan in lock_record['candidates']:
            rows.append(fit_candidate(plan,train_data,validation,store,mean,sd,lock_sha,data_hash=lock_record['data_manifest_sha256']))
            write_json(STUDY/'progress.json',dict(updated_utc=utc(),completed=len(rows),total=150,
                valid=sum(r['status']=='valid' for r in rows),failed=sum(r['status']=='failed' for r in rows),
                completed_epoch_seconds=math.fsum(r['completed_epoch_seconds'] for r in rows),last_id=plan['id']))
        chosen=selection.select(rows)
        receipt=chosen|dict(created_utc=utc(),implementation_lock_sha256=lock_sha,
            candidate_records=[dict(path=str((Path('candidates')/(r['id']+'.json'))),sha256=sha(STUDY/'candidates'/(r['id']+'.json'))) for r in rows])
        destination=STUDY/'selection_lock.json'
        if destination.exists():
            old=read(destination)
            for k,v in receipt.items():
                if k!='created_utc' and old[k]!=v:raise ValueError('Existing selection differs')
        else:write_json(destination,receipt,immutable=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','train']);args=parser.parse_args()
    {'lock':lock,'train':train}[args.action]()
