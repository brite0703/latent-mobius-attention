"""Three retained random initial states; precision-only, aggregate900s, no retry."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time,math,traceback,subprocess,os,importlib.util
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'lboia_n20_witness_recovery_pilot_20261006'
PRECISION=HERE.parent/'lboia_n20_seed100_fp64_precision_pilot_20261006'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helpers=module('completed_precision_pure_helpers',PRECISION/'run_precision_fit.py')
sha,read,dump,append,now=helpers.sha,helpers.read,helpers.dump,helpers.append,helpers.now
diag,pm,pc,checks=helpers.diag,helpers.pm,helpers.pc,helpers.checks
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def gpu_idle_check():
 query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 rows=[s.strip() for s in query.splitlines() if s.strip()]
 unknown=[s for s in rows if '/gnome-remote-desktop-daemon' not in s and s.split(',')[0].strip()!=str(os.getpid())]
 dump(HERE/'evidence/gpu_preflight.json',{'utc':now(),'compute_processes':rows,'allowed_display_process':'gnome-remote-desktop-daemon','own_startup_context_pid':os.getpid(),'other_compute_processes':unknown,'status':'PASS' if not unknown else 'BLOCKED'})
 assert not unknown,'GPU already occupied by another compute process: '+str(unknown)
 assert torch.cuda.is_available(),'GPU unavailable; no implicit CPU fallback'

def direct(model,blob,deadline,savepath=None):
 deadline();was=model.training;model.eval();logits=[]
 with torch.inference_mode():
  for start in range(0,len(blob['y']),256):
   deadline();logits.append(model(blob['x'][start:start+256]).cpu().numpy())
 actual=np.concatenate(logits);y=blob['y'].cpu().numpy()
 assert np.isfinite(actual).all()
 result=pc.metrics(y,actual);margin=(2*y-1)*(actual[:,1]-actual[:,0])
 result.update(minimum_signed_margin=float(margin.min()),mean_signed_margin=float(margin.mean()),median_signed_margin=float(np.median(margin)),
  percentile05_signed_margin=float(np.quantile(margin,.05)),negative_margin_fraction=float(np.mean(margin<0)),
  fit_success_both=result['accuracy']>=.99 and result['cross_entropy']<=.05,scoring='native direct actual rows; no count lookup')
 if savepath:np.savez_compressed(savepath,truth=y,count=blob['count'],actual_row_logits=actual)
 model.train(was);return result

def instantiate(spec):
 seed=spec['seed'];source=torch.load(spec['initial_checkpoint_path'],map_location='cpu',weights_only=True)
 assert source['spec']['id']==f'n20_original_seed{seed}' and source['spec']['initializer']=='original'
 assert all(v.dtype==torch.float32 for v in source['state_dict'].values() if v.is_floating_point())
 model=pm.build_model(seed,'lma3',20).double()
 converted={k:v.double() if v.is_floating_point() else v.clone() for k,v in source['state_dict'].items()}
 model.load_state_dict(converted)
 assert all(torch.equal(model.state_dict()[k],v) for k,v in converted.items())
 assert all(torch.equal(model.state_dict()[k].float(),v) for k,v in source['state_dict'].items() if v.is_floating_point())
 model.cuda();torch.manual_seed(30000+seed)
 assert all(p.requires_grad and p.dtype==torch.float64 for p in model.parameters())
 assert sum(p.numel() for p in model.parameters())==12525
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100)
 gen=torch.Generator(device='cpu').manual_seed(20000+seed)
 coverage=checks.optimizer_coverage(model,opt)
 data={}
 for split in ['train','val']:
  blob=pc.load_data(20,seed,split,'cuda');before=blob['x'];blob['x']=before.double()
  assert torch.equal(blob['x'].float(),before)
  assert bool(((before==0)|(before==1)).all()) and before.shape[1]==20
  assert torch.equal(blob['y'],before.sum(1).long()%2)
  assert len(blob['y'])==(4800 if split=='train' else 600)
  data[split]=blob
 return model,opt,sched,gen,data,coverage,source['spec']

def fit(spec,deadline,campaign_start,index):
 folder=HERE/'runs'/spec['id'];folder.mkdir(parents=True,exist_ok=False)
 append(HERE/'evidence/started_runs.jsonl',{'utc':now(),'run_id':spec['id'],'seed':spec['seed'],'started_training_runs':index,'initial_sha256':spec['initial_checkpoint_sha256']})
 print(json.dumps({'stage':'ACTUAL_RANDOM_FLOAT64_RUN_START','utc':now(),'seed':spec['seed'],'run_id':spec['id'],'started_training_runs':index,'maximum_new_runs':3}),flush=True)
 model,opt,sched,gen,data,coverage,base_spec=instantiate(spec)
 torch.save({'state_dict':state(model),'dtype':'float64','source_sha256':spec['initial_checkpoint_sha256']},folder/'initial_exact_cast.pt')
 dump(folder/'identity_check.json',{'status':'PASS','seed':spec['seed'],'exact_cast_all_tensors':True,'float32_roundtrip_bitwise_equal':True,'no_initial_state_regeneration':True,'optimizer_coverage':coverage,'initial_float64_sha256':sha(folder/'initial_exact_cast.pt')})
 prior_epochs=[json.loads(s) for s in (PRIOR/'runs'/f"n20_original_seed{spec['seed']}"/'epochs.jsonl').read_text().splitlines()]
 assert len(prior_epochs)==100
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();history=[];steps=0;status='started';error=None;peak_rss=diag.rss_mib()
 try:
  deadline();initial={split:direct(model,data[split],deadline,folder/f'initial_{split}_predictions.npz') for split in ['train','val']}
  dump(folder/'initial_direct_metrics.json',initial)
  diag.snapshot(model,base_spec,data['train'],None,None,0,folder)
  for epoch in range(1,101):
   deadline();tick=time.perf_counter();model.train();order=torch.randperm(4800,generator=gen).cuda()
   batch_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
   assert batch_hash==prior_epochs[epoch-1]['batch_order_sha256'],'Matched batch identity failed'
   sum_loss,correct,maxnorm,clipped=0.,0,0.,0
   for offset in range(0,4800,256):
    deadline();idx=order[offset:offset+256];opt.zero_grad(set_to_none=True)
    out=model(data['train']['x'][idx]);loss=nn.functional.cross_entropy(out,data['train']['y'][idx]);loss.backward()
    norm=float(nn.utils.clip_grad_norm_(model.parameters(),10.))
    if not math.isfinite(float(loss.detach())) or not math.isfinite(norm):raise FloatingPointError('Nonfinite loss/gradient')
    flat_grad=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]);zeros=float((flat_grad==0).double().mean())
    before=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    captured=checks.capture_first_update(model,opt) if steps==0 else None
    opt.step();steps+=1
    update=torch.cat([p.detach().reshape(-1) for p in model.parameters()])-before
    if captured is not None:
     check=checks.check_first_update(model,opt,captured)
     assert max(x['adamw_first_step_maximum_absolute_error'] for x in check['rows'])<1e-10
     check['additional_float64_absolute_tolerance']=1e-10;dump(folder/'optimizer_update_step1.json',check)
    append(folder/'minibatches.jsonl',{'epoch':epoch,'batch_index':offset//256,'step':steps,'rows':len(idx),'prediction_loss':float(loss.detach()),
     'preclip_global_gradient_norm':norm,'clip_factor':min(1.,10./(norm+1e-6)),'gradient_zero_fraction':zeros,
     'committed_update_l2':float(update.norm()),'maximum_absolute_update':float(update.abs().max()),'learning_rate':opt.param_groups[0]['lr']})
    sum_loss+=float(loss.detach())*len(idx);correct+=int((out.detach().argmax(1)==data['train']['y'][idx]).sum());maxnorm=max(maxnorm,norm);clipped+=norm>10.
   sched.step();assert sched.get_last_lr()[0]==prior_epochs[epoch-1]['learning_rate_after_epoch']
   row={'epoch':epoch,'batch_order_sha256':batch_hash,'paired_order_exact':True,'online_training_cross_entropy':sum_loss/4800,'online_training_accuracy':correct/4800,
    'maximum_preclip_gradient_norm':maxnorm,'clipped_minibatches':int(clipped),'total_minibatches':19,'learning_rate_after_epoch':sched.get_last_lr()[0]}
   if epoch==1 or epoch%5==0:
    row['training_direct']=direct(model,data['train'],deadline);row['validation_direct']=direct(model,data['val'],deadline)
   if epoch==spec['matched_fp32_validation_epoch']:
    torch.save({'state_dict':state(model),'dtype':'float64','epoch':epoch,'selection':'matched prior float32-selected epoch; no new selection'},folder/'matched_validation_epoch.pt')
   diag.snapshot(model,base_spec,data['train'],None,None,epoch,folder);torch.cuda.synchronize()
   row.update(epoch_wall_seconds=time.perf_counter()-tick,cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,cuda_reserved_mib=torch.cuda.memory_reserved()/2**20,rss_mib=diag.rss_mib())
   peak_rss=max(peak_rss,row['rss_mib']);history.append(row);append(folder/'epochs.jsonl',row)
   dump(HERE/'evidence/progress.json',{'utc':now(),'run_id':spec['id'],'seed':spec['seed'],'epoch':epoch,'steps':steps,'started_training_runs':index,
    'aggregate_elapsed_seconds':time.perf_counter()-campaign_start,'maximum_aggregate_seconds':900})
   if epoch==1 or epoch%10==0:print(json.dumps({'stage':'random_float64_training','utc':now(),'seed':spec['seed'],'epoch':epoch,'aggregate_wall_seconds':time.perf_counter()-campaign_start}),flush=True)
  deadline();torch.save({'state_dict':state(model),'dtype':'float64','epoch':100},folder/'final_epoch100.pt')
  assert steps==1900 and len(history)==100
  status='completed'
 except Exception as exc:
  status='budget_terminated' if isinstance(exc,TimeoutError) else 'scientific_failure' if isinstance(exc,(FloatingPointError,torch.OutOfMemoryError)) else 'implementation_error'
  error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
  torch.save({'state_dict':state(model),'dtype':'float64','completed_epochs':len(history),'optimizer_steps':steps},folder/'failed_or_partial.pt')
 finally:
  torch.cuda.synchronize()
  result={'spec':spec,'status':status,'finished_utc':now(),'completed_epochs':len(history),'optimizer_steps':steps,'wall_seconds':time.perf_counter()-start,
   'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'peak_sampled_rss_mib':peak_rss,
   'initial_metrics':locals().get('initial'),'error':error,'equal_horizon_pair':status=='completed',
   'final_checkpoint_sha256':sha(folder/'final_epoch100.pt') if (folder/'final_epoch100.pt').exists() else None,
   'matched_validation_checkpoint_sha256':sha(folder/'matched_validation_epoch.pt') if (folder/'matched_validation_epoch.pt').exists() else None}
  dump(folder/'result.json',result);del model,opt,sched,gen,data;torch.cuda.empty_cache()
 print(json.dumps({'stage':'random_float64_fit_complete','utc':now(),'seed':spec['seed'],'status':status,'epochs':len(history),'wall_seconds':result['wall_seconds']}),flush=True)
 return result

def main():
 protocol=read(HERE/'evidence/protocol_frozen.json');lock=read(HERE/'evidence/protocol_lock.json')
 assert not (HERE/'evidence/budget.json').exists(),'Campaign executes once; no resume/retry'
 assert sha(HERE/'evidence/protocol_frozen.json')==lock['protocol_sha256']
 for e in lock['files']:assert sha(e['path'])==e['sha256'],'Protected input mismatch: '+e['path']
 gpu_idle_check();pc.configure();torch.use_deterministic_algorithms(True)
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
 assert sha(pm.source.__file__)=='5f62f77ef974c007ac837b427ec1fe1fa0600cf84de6d7fa4fcf44cfa3970b53'
 dump(HERE/'evidence/resources.json',{'utc':now(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA_build':torch.version.cuda,'python':sys.version,
  'cpu_threads':torch.get_num_threads(),'deterministic':True,'tf32':torch.backends.cuda.matmul.allow_tf32,
  'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()})
 start=time.perf_counter();wall_start=time.time();end=start+900
 def deadline():
  if time.perf_counter()>=end:raise TimeoutError('Frozen aggregate900-second cap reached')
 dump(HERE/'evidence/budget.json',{'start_utc':now(),'start_unix':wall_start,'maximum_aggregate_seconds':900,'maximum_new_fits':3})
 print(json.dumps({'stage':'AUTHORIZED_THREE_RANDOM_FLOAT64_CAMPAIGN_START','utc':now(),'pid':os.getpid(),'maximum_new_fits':3,'maximum_aggregate_seconds':900,'protocol_sha256':lock['protocol_sha256']}),flush=True)
 results=[];evaluations=[];error=None
 try:
  for index,spec in enumerate(protocol['run_specs'],1):
   deadline();result=fit(spec,deadline,start,index);results.append(result)
   if result['status'] in ['budget_terminated','implementation_error']:break
  dump(HERE/'evidence/all_fits_locked.json',{'utc':now(),'protocol_sha256':lock['protocol_sha256'],'results':results,'started_fits':len(results),'aggregate_elapsed_seconds':time.perf_counter()-start})
  for result in results:
   if result['status']!='completed':continue
   deadline();spec=result['spec'];folder=HERE/'runs'/spec['id'];model=pm.build_model(spec['seed'],'lma3',20).double().cuda()
   row={'spec':spec,'status':'direct_native_evaluation_completed'}
   for name,key,expected in [('final','final_epoch100.pt',result['final_checkpoint_sha256']),('matched_validation','matched_validation_epoch.pt',result['matched_validation_checkpoint_sha256'])]:
    path=folder/key;assert sha(path)==expected;model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);scores={}
    for split in ['train','val','test']:
     deadline();blob=pc.load_data(20,spec['seed'],split,'cuda');blob['x']=blob['x'].double()
     scores[split]=direct(model,blob,deadline,folder/f'{name}_{split}_predictions.npz');del blob
    row[name]=scores
   evaluations.append(row);del model;torch.cuda.empty_cache()
  dump(HERE/'evidence/direct_actual_evaluation.json',{'utc':now(),'all_fits_lock_sha256':sha(HERE/'evidence/all_fits_locked.json'),'rows':evaluations,'aggregate_elapsed_seconds':time.perf_counter()-start,'no_count_lookup_scoring':True})
 except Exception as exc:error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()};dump(HERE/'evidence/campaign_failure.json',error)
 finally:
  torch.cuda.synchronize();torch.cuda.empty_cache()
  finished={'utc':now(),'cuda_work_finished':True,'started_training_runs':len(list((HERE/'runs').iterdir())) if (HERE/'runs').exists() else 0,
   'completed_training_runs':sum(r['status']=='completed' for r in results),'evaluated_runs':len(evaluations),'aggregate_wall_seconds':time.perf_counter()-start,'maximum_seconds':900,
   'no_more_GPU_work_planned':True,'error':error}
  dump(HERE/'evidence/gpu_release.json',finished);print(json.dumps({'stage':'GPU_RELEASED',**finished}),flush=True)
 changed=[e['path'] for e in lock['files'] if sha(e['path'])!=e['sha256']];assert not changed,'Protected inputs changed'
 dump(HERE/'evidence/preservation_check.json',{'status':'PASS','utc':now(),'protected_files_unchanged':len(lock['files']),'changes':changed})
 return 0 if error is None and len(evaluations)==3 else 1
if __name__=='__main__':sys.exit(main())
