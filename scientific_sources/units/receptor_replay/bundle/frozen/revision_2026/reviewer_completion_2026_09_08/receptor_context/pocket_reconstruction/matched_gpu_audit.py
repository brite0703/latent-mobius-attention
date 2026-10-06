"""Discarded CUDA feasibility/continuation checks after the prior GPU pipeline."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,math,os,subprocess,time
import numpy as np
import torch
import matched_execution as execution
import matched_campaign as campaign
import matched_models as models
from sequence_data import SequenceStore
from audit_matched_execution import fixture,equal_tree

HERE=Path(__file__).resolve().parent
PREREQUISITES=('matched_models_cpu_audit.json','matched_execution_cpu_audit.json',
    'matched_campaign_cpu_audit.json','matched_selection_scoring_cpu_audit.json')

def assert_inputs():
    # Complete train/evaluate/audit/analyze/profile before changing GPU workload.
    parent=HERE.parent
    for name,flag in [('sequence_final_audit.json','passed'),('sequence_profiles.json','complete')]:
        record=campaign.read(parent/name)
        if not record.get(flag):raise RuntimeError('The full-sequence pipeline is still incomplete: '+name)
    final=campaign.read(parent/'sequence_final_audit.json')
    if final['evaluation_sha256']!=campaign.sha(parent/'sequence_evaluation.json'):
        raise ValueError('The preceding sequence evaluation changed')
    for name in PREREQUISITES:
        receipt=campaign.read(HERE/name)
        if not receipt.get('passed'):raise ValueError('CPU prerequisite incomplete: '+name)
        closure=receipt.get('source_closure',receipt.get('sources',[]))
        campaign.verify_closure(closure)
    panel=campaign.read(HERE/'matched_candidate_plan.json')
    if panel['source_sha256']!=campaign.sha(HERE/'matched_selection.py'):
        raise ValueError('Candidate/contrast source changed after its audit')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
        check=True,capture_output=True,text=True)
    others=[]
    for line in query.stdout.splitlines():
        parts=line.split(',',1)
        if len(parts)==2 and parts[0].strip().isdigit() and int(parts[0].strip())!=os.getpid() and 'python' in parts[1].lower():others.append(line.strip())
    if others:raise RuntimeError('Another Python GPU workload is active; wait: '+repr(others))

def artificial_envelopes(manifest):
    # Read supplied strings and input dimensions; no held-out tensors or targets.
    sequences=SequenceStore();rows=manifest['rows'];bounds={}
    for split in ('train','val','test'):
        selected=[r for r in rows if r['split']==split]
        bounds[split]=dict(records=len(selected),maximum_atoms=max(r['atom_count'] for r in selected),
            maximum_chain_length=max(sequences.sizes[r['pdbid']][1] for r in selected),
            maximum_supplied_chains=max(sequences.sizes[r['pdbid']][0] for r in selected),
            maximum_singleton_tokens=max(math.prod(sequences.sizes[r['pdbid']]) for r in selected))
    atoms=max(r['maximum_atoms'] for r in bounds.values())
    longest=max(r['maximum_chain_length'] for r in bounds.values())
    chains=max(r['maximum_supplied_chains'] for r in bounds.values())
    specs=[('record_cap',32,1,execution.MAX_PADDED_TOKENS//32),
        ('longest_chain',min(32,execution.MAX_PADDED_TOKENS//longest),1,longest),
        ('chain_count',1,chains,min(longest,execution.MAX_PADDED_TOKENS//chains))]
    blobs={}
    for name,n,c,l in specs:
        assert n>=1 and l>=1 and n*c*l<=execution.MAX_PADDED_TOKENS and atoms==57
        blobs[name]=execution.indexed_blob(dict(ids=[f'shape_{name}_{i}' for i in range(n)],
            sequences=[':'.join(['A'*l]*c)]*n,y=torch.zeros(n),mask=torch.ones(n,atoms,dtype=torch.bool),
            X=torch.ones(n,atoms,53),adj=torch.eye(atoms).unsqueeze(0).repeat(n,1,1),contact=torch.ones(n,atoms,216)))
    store=execution.CommonStore(list(blobs.values()))
    for blob in blobs.values():assert list(store.chunks(blob['ids']))==[blob['ids']]
    return store,blobs,dict(bounds_by_split=bounds,artificial_cases=[dict(name=name,records=n,chains=c,chain_length=l,
        ligand_atoms=atoms,contact_slots=n*atoms*264,padded_tokens=n*c*l) for name,n,c,l in specs],
        heldout_tensor_files_loaded=False,heldout_numeric_targets_accessed=False)

def run():
    destination=HERE/'matched_gpu_audit.json'
    assert_inputs()
    if destination.exists():
        receipt=campaign.read(destination)
        campaign.verify_closure(receipt['source_closure'])
        if not receipt.get('passed'):raise ValueError('Existing GPU receipt is incomplete')
        print(json.dumps({'passed':True,'preserved_existing_audit':True}));return
    # The old pipeline uses this same lock for every GPU stage.
    with campaign.exclusive_process(HERE.parent/'sequence_run_control'):
        execution.configure('cuda')
        manifest=campaign.read(execution.CACHE/'manifest.json');train=execution.load_split('train',manifest)
        store=execution.CommonStore([train]);mean=manifest['target_scale']['mean'];sd=manifest['target_scale']['population_sd']
        ranked=sorted(train['ids'],key=lambda k:(-math.prod(store.sizes[k]),k))[:256]
        anchors=sorted({ranked[0],max(train['ids'],key=lambda k:store.sizes[k][0]),
            max(train['ids'],key=lambda k:store.sizes[k][1]),max(train['ids'],key=lambda k:store.atom_sizes[k]),
            train['ids'][int(train['contact'].abs().flatten(1).amax(1).argmax())]})
        envelope_store,envelopes,shape_guard=artificial_envelopes(manifest)
        toy_store,toy=fixture();toy_mean,toy_sd=execution.scalers(toy)
        scratch=HERE/'tmp/matched_gpu_audit';scratch.mkdir(parents=True,exist_ok=True)
        results=[]
        for setting,head in models.configurations():
            runtime=execution.make_runtime(42,setting,head,.001,device='cuda')
            model,optimizer=runtime['model'],runtime['optimizer']
            for key in anchors:execution.effective_step(model,optimizer,train,store,[key],mean,sd)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated();began=time.perf_counter()
            worst=execution.effective_step(model,optimizer,train,store,ranked,mean,sd)
            torch.cuda.synchronize();worst_seconds=time.perf_counter()-began;worst_peak=torch.cuda.max_memory_allocated()
            times=[]
            for _ in range(5):
                began=time.perf_counter();ordinary=execution.effective_step(model,optimizer,train,store,train['ids'][:256],mean,sd)
                torch.cuda.synchronize();times.append(time.perf_counter()-began)
            tail=execution.effective_step(model,optimizer,train,store,train['ids'][-72:],mean,sd)
            assert tail['records']==72
            small=execution.indexed_blob({k:train[k][:64] for k in ('ids','sequences','X','mask','adj','contact','y')})
            began=time.perf_counter();prediction=execution.predict(model,small,store,mean,sd)
            torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-began
            assert prediction.shape==(64,) and np.isfinite(prediction).all()
            envelope_results=[]
            for name,blob in envelopes.items():
                torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
                # Both reverse-mode fitting and forward inference are tested.
                step=execution.effective_step(model,optimizer,blob,envelope_store,blob['ids'],0.,1.)
                values=execution.predict(model,blob,envelope_store,0.,1.);torch.cuda.synchronize()
                assert values.shape==(len(blob['ids']),) and np.isfinite(values).all()
                envelope_results.append(dict(name=name,seconds=time.perf_counter()-began,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),step=step))
            def fresh():return execution.make_runtime(42,setting,head,.001,device='cuda',epochs=2,effective_batch=5,max_records=2,data_manifest_sha256='gpu-fixture')
            full=fresh()
            for _ in range(2):execution.finish_epoch(full,toy,toy,toy_store,toy_mean,toy_sd)
            partial=fresh();execution.finish_epoch(partial,toy,toy,toy_store,toy_mean,toy_sd)
            path=scratch/(setting+'_'+head+'.pt');execution.save_snapshot(path,partial,toy_mean,toy_sd)
            restored=fresh();execution.restore_snapshot(path,restored,toy_mean,toy_sd)
            execution.finish_epoch(restored,toy,toy,toy_store,toy_mean,toy_sd)
            for key in ('model','optimizer','scheduler'):equal_tree(full[key].state_dict(),restored[key].state_dict())
            equal_tree(full['generator'].get_state(),restored['generator'].get_state());equal_tree(full['best_state'],restored['best_state'])
            for a,b in zip(full['history'],restored['history']):
                equal_tree({k:v for k,v in a.items() if k!='completed_epoch_seconds'},
                           {k:v for k,v in b.items() if k!='completed_epoch_seconds'})
            np.testing.assert_array_equal(execution.predict(full['model'],toy,toy_store,toy_mean,toy_sd),
                execution.predict(restored['model'],toy,toy_store,toy_mean,toy_sd))
            results.append(dict(setting=setting,head=head,parameters=model.parameter_counts(),anchor_ids=anchors,
                ranked_256_step=worst,ranked_256_seconds=worst_seconds,baseline_allocated_bytes=baseline,ranked_peak_allocated_bytes=worst_peak,
                ordinary_256_seconds=times,ordinary_256_median_seconds=float(np.median(times)),ordinary_step=ordinary,
                final_72_step=tail,evaluation_64_seconds=evaluation_seconds,artificial_envelopes=envelope_results,cuda_continuation_exact=True))
            print(json.dumps({'checked':len(results),'total':15,'setting':setting,'head':head,'ordinary_step_seconds':float(np.median(times))}),flush=True)
            del runtime,model,optimizer,full,partial,restored
            torch.cuda.empty_cache()
        paths=[Path(__file__),HERE/'matched_execution.py',HERE/'matched_models.py',HERE/'audit_matched_execution.py',
            HERE/'matched_candidate_plan.json',execution.CACHE/'manifest.json',execution.CACHE/'train.pt',
            HERE.parent/'sequence_data.py',HERE.parent/'published_sequences.csv',*[HERE/n for n in PREREQUISITES]]
        result=dict(passed=True,completed_utc=campaign.utc(),configurations=15,rows=results,shape_guard=shape_guard,
            target_mean=mean,target_population_sd=sd,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            source_closure=[dict(path=str(p),sha256=campaign.sha(p)) for p in paths],retained_candidate=False,
            projected_training_update_hours=100*5*10*sum(r['ordinary_256_median_seconds'] for r in results)/3600,
            qualification='Discarded fitting-only resource checks and artificial full-cohort shape envelopes. The projection treats all five epoch batches as 256 records and excludes validation/checkpoint I/O; it is not a completion guarantee. No held-out affinity prediction or retained procedure choice is made.')
        campaign.write_json(destination,result,immutable=True)
        print(json.dumps({'passed':True,'configurations':15,'projected_training_update_hours':result['projected_training_update_hours']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:assert_inputs();print('Prerequisites complete; GPU work is eligible to start.')
    else:run()
