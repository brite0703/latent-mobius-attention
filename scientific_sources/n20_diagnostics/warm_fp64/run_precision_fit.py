"""One paired native N20 fit. The sole planned training factor is floating dtype."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, importlib.util, math, time, traceback, os, subprocess
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'lboia_n20_witness_recovery_pilot_20261006'
FIRST = HERE.parent / 'lboia_routing_parity_pilot_20261006'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)

def append(path, value):
    with Path(path).open('a') as f: f.write(json.dumps(value, allow_nan=False)+'\n')

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(FIRST))
diag = load_module('frozen_pilot_diagnostics', FIRST/'pilot.py')
checks = load_module('prior_update_checks', PRIOR/'training_chain_checks.py')
pm, pc = diag.pm, diag.pc

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def main():
    protocol = read(HERE/'evidence/protocol_frozen.json')
    lock = read(HERE/'evidence/protocol_lock.json')
    assert not (HERE/'evidence/started.json').exists(), 'No duplicate or resumed fit'
    assert sha(HERE/'evidence/protocol_frozen.json') == lock['protocol_sha256']
    for entry in lock['files']:
        assert sha(entry['path']) == entry['sha256'], 'Protected identity mismatch: '+entry['path']
    assert torch.cuda.is_available(), 'Verified GPU unavailable; no unapproved fallback'
    pc.configure(); torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert Path(pm.source.__file__).resolve() == (FIRST/'original_workspace/LMA/lma.py').resolve()
    assert sha(pm.source.__file__) == protocol['native_lma_sha256']
    checkpoint = torch.load(protocol['initial_checkpoint']['path'], map_location='cpu', weights_only=True)
    assert checkpoint['spec']['id'] == 'n20_perturbed_witness_seed100'
    assert all(v.dtype == torch.float32 for v in checkpoint['state_dict'].values() if v.is_floating_point())
    model = pm.build_model(100, 'lma3', 20).double()
    converted = {k:v.double() if v.is_floating_point() else v.clone() for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(converted)
    equal = all(torch.equal(model.state_dict()[k].float(), v) for k,v in checkpoint['state_dict'].items() if v.is_floating_point())
    assert equal and all(torch.equal(model.state_dict()[k],v) for k,v in converted.items())
    model.cuda()
    assert sum(p.numel() for p in model.parameters()) == 12525
    assert all(p.dtype == torch.float64 and p.requires_grad for p in model.parameters())
    torch.manual_seed(30100)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    gen = torch.Generator(device='cpu').manual_seed(20100)
    coverage = checks.optimizer_coverage(model, opt)
    data = {}
    for split in ['train','val','test']:
        blob = pc.load_data(20,100,split,'cuda')
        original = blob['x']; blob['x'] = original.double()
        assert torch.equal(blob['x'].float(), original)
        assert len(blob['y']) == (4800 if split=='train' else 600)
        assert bool(((original == 0) | (original == 1)).all())
        assert torch.equal(blob['y'], original.sum(1).long()%2)
        data[split] = blob
    prior_orders = [json.loads(s) for s in (PRIOR/'runs/n20_perturbed_witness_seed100/epochs.jsonl').read_text().splitlines()]
    assert len(prior_orders) == 100
    folder = HERE/'run'; folder.mkdir(exist_ok=False)
    torch.save({'state_dict':cpu_state(model),'source_checkpoint_sha256':protocol['initial_checkpoint']['sha256'],'dtype':'float64'},folder/'initial_exact_cast.pt')
    dump(HERE/'evidence/identity_check.json', {'status':'PASS','utc':now(),'state_exactly_float32_to_float64':True,
         'float32_roundtrip_bitwise_equal':equal,'no_noise_regeneration':True,'all12525_trainable_float64':True,
         'optimizer_coverage':coverage,'splits_rows':{k:len(v['y']) for k,v in data.items()},
         'source_and_protected_hashes_checked':len(lock['files']),'initial_cast_checkpoint_sha256':sha(folder/'initial_exact_cast.pt')})
    resources = {'utc':now(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                 'cuda_build':torch.version.cuda,'python':sys.version,'logical_cpu':os.cpu_count(),
                 'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
                 'cpu_threads':torch.get_num_threads(),'deterministic':True,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}
    dump(HERE/'evidence/resources.json',resources)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter(); wall_start = time.time(); end = start+900
    def deadline():
        if time.perf_counter() >= end: raise TimeoutError('Frozen 900-second local execution cap reached')
    def direct_score(split, tag, save=False):
        deadline(); blob=data[split]; was=model.training; model.eval()
        logits=[]
        with torch.inference_mode():
            for offset in range(0,len(blob['y']),256):
                deadline(); logits.append(model(blob['x'][offset:offset+256]).cpu().numpy())
        values=np.concatenate(logits); y=blob['y'].cpu().numpy()
        assert np.isfinite(values).all()
        m=pc.metrics(y,values); margin=(2*y-1)*(values[:,1]-values[:,0])
        m.update(minimum_signed_margin=float(margin.min()),mean_signed_margin=float(margin.mean()),
                 median_signed_margin=float(np.median(margin)),percentile05_signed_margin=float(np.quantile(margin,.05)),
                 negative_margin_fraction=float(np.mean(margin<0)),fit_success_both=m['accuracy']>=.99 and m['cross_entropy']<=.05,
                 scoring='direct native inference on actual original rows; no count lookup')
        if save: np.savez_compressed(folder/(tag+'_'+split+'_predictions.npz'),truth=y,actual_row_logits=values,count=blob['count'])
        model.train(was); return m
    dump(HERE/'evidence/started.json',{'started_utc':now(),'pid':os.getpid(),'run_id':'n20_perturbed_witness_seed100_float64',
         'started_training_runs':1,'maximum_new_fits':1,'maximum_wall_seconds':900,'start_unix':wall_start,
         'protocol_sha256':lock['protocol_sha256'],'initial_checkpoint_sha256':protocol['initial_checkpoint']['sha256']})
    print(json.dumps({'stage':'ACTUAL_FLOAT64_RUN_START','utc':now(),'pid':os.getpid(),'gpu':resources['gpu'],'maximum_new_fits':1,'maximum_seconds':900}),flush=True)
    history=[]; steps=0; peak_rss=diag.rss_mib(); status='started'; error=None
    try:
        initial={split:direct_score(split,'initial',True) for split in ['train','val']}
        dump(folder/'initial_direct_metrics.json',initial)
        spec=checkpoint['spec']
        diag.snapshot(model,spec,data['train'],None,None,0,folder)
        for epoch in range(1,101):
            deadline(); tick=time.perf_counter(); model.train()
            order=torch.randperm(4800,generator=gen).cuda()
            batch_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            assert batch_hash==prior_orders[epoch-1]['batch_order_sha256'], 'Paired batch order mismatch'
            sum_loss,correct,max_norm,clipped,update_sum=0.,0,0.,0,0.
            for offset in range(0,4800,256):
                deadline(); idx=order[offset:offset+256]
                opt.zero_grad(set_to_none=True)
                out=model(data['train']['x'][idx]); loss=nn.functional.cross_entropy(out,data['train']['y'][idx]); loss.backward()
                norm=float(nn.utils.clip_grad_norm_(model.parameters(),10.))
                assert math.isfinite(float(loss.detach())) and math.isfinite(norm),'Nonfinite training'
                flat_grad=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])
                zero_fraction=float((flat_grad==0).double().mean())
                first=checks.capture_first_update(model,opt) if steps==0 else None
                before=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
                opt.step(); steps+=1
                delta=torch.cat([p.detach().reshape(-1) for p in model.parameters()])-before
                update_norm=float(delta.norm()); update_max=float(delta.abs().max()); update_sum+=update_norm
                if first is not None:
                    rec=checks.check_first_update(model,opt,first)
                    assert max(x['adamw_first_step_maximum_absolute_error'] for x in rec['rows'])<1e-10
                    rec['additional_float64_absolute_tolerance']=1e-10
                    dump(folder/'optimizer_update_step1.json',rec)
                batch={'epoch':epoch,'batch_index':offset//256,'step':steps,'rows':len(idx),'prediction_loss':float(loss.detach()),
                       'preclip_global_gradient_norm':norm,'clip_factor':min(1.,10./(norm+1e-6)),
                       'gradient_zero_fraction':zero_fraction,'committed_update_l2':update_norm,'maximum_absolute_update':update_max,
                       'learning_rate':opt.param_groups[0]['lr']}
                append(folder/'minibatches.jsonl',batch)
                sum_loss+=float(loss.detach())*len(idx); correct+=int((out.detach().argmax(1)==data['train']['y'][idx]).sum())
                max_norm=max(max_norm,norm); clipped+=norm>10.
            sched.step()
            row={'epoch':epoch,'batch_order_sha256':batch_hash,'paired_order_exact':True,'online_training_cross_entropy':sum_loss/4800,
                 'online_training_accuracy':correct/4800,'maximum_preclip_gradient_norm':max_norm,'clipped_minibatches':int(clipped),
                 'total_minibatches':19,'sum_update_l2':update_sum,'learning_rate_after_epoch':sched.get_last_lr()[0]}
            if epoch==1 or epoch%5==0:
                row['training_direct']=direct_score('train',f'epoch{epoch}')
                row['validation_direct']=direct_score('val',f'epoch{epoch}')
            diag.snapshot(model,spec,data['train'],None,None,epoch,folder)
            torch.cuda.synchronize()
            row.update(epoch_wall_seconds=time.perf_counter()-tick,cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,
                       cuda_reserved_mib=torch.cuda.memory_reserved()/2**20,rss_mib=diag.rss_mib())
            peak_rss=max(peak_rss,row['rss_mib']); history.append(row); append(folder/'epochs.jsonl',row)
            dump(HERE/'evidence/progress.json',{'utc':now(),'epoch':epoch,'steps':steps,'maximum_epochs':100,
                 'started_training_runs':1,'elapsed_wall_seconds':time.perf_counter()-start,'maximum_seconds':900})
            if epoch==1 or epoch%10==0:
                print(json.dumps({'stage':'float64_training','utc':now(),'epoch':epoch,'steps':steps,'wall_seconds':time.perf_counter()-start,
                                 'training_direct':row.get('training_direct')}),flush=True)
        deadline(); torch.save({'state_dict':cpu_state(model),'dtype':'float64','source_checkpoint_sha256':protocol['initial_checkpoint']['sha256']},folder/'final_epoch100.pt')
        dump(HERE/'evidence/fit_locked.json',{'utc':now(),'completed_epochs':100,'optimizer_steps':steps,'checkpoint_sha256':sha(folder/'final_epoch100.pt')})
        final={split:direct_score(split,'final',True) for split in ['train','val','test']}
        dump(folder/'final_direct_metrics.json',final)
        assert steps==1900 and len(history)==100
        status='completed'
    except Exception as exc:
        status='budget_terminated' if isinstance(exc,TimeoutError) else 'failed'
        error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
        torch.save({'state_dict':cpu_state(model),'dtype':'float64','completed_epochs':len(history),'optimizer_steps':steps},folder/'failed_or_partial.pt')
        dump(HERE/'evidence/failure.json',error)
    finally:
        torch.cuda.synchronize()
        result={'status':status,'finished_utc':now(),'completed_epochs':len(history),'optimizer_steps':steps,
                'started_training_runs':1,'wall_seconds':time.perf_counter()-start,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'peak_sampled_rss_mib':peak_rss,
                'initial_metrics':locals().get('initial'),'final_metrics':locals().get('final'),'error':error,
                'equal_horizon_pair':status=='completed','selection':'fixed epoch100, equal to prior fp32 final and validation-best epoch; no new checkpoint selection'}
        dump(folder/'result.json',result)
        del model,opt,sched,data; torch.cuda.empty_cache()
        dump(HERE/'evidence/gpu_release.json',{'utc':now(),'cuda_work_finished':True,'no_more_training_or_GPU_checks_planned':True,
             'completed_epochs':len(history),'optimizer_steps':steps,'status':status})
        print(json.dumps({'stage':'GPU_RELEASED','utc':now(),'status':status,'epochs':len(history),'wall_seconds':result['wall_seconds']}),flush=True)
    changed=[entry['path'] for entry in lock['files'] if sha(entry['path'])!=entry['sha256']]
    assert not changed,'Protected inputs changed'
    dump(HERE/'evidence/preservation_check.json',{'utc':now(),'status':'PASS','protected_files':len(lock['files']),'changes':changed})
    return 0 if status=='completed' else 1

if __name__=='__main__': sys.exit(main())
