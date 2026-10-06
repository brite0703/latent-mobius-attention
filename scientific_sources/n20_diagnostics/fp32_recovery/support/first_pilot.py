"""Bounded matched pilot. Unmodified original models/data; independent output locks."""
from pathlib import Path
import argparse, json, math, time, traceback, sys, hashlib, copy
import numpy as np
import torch
from torch import nn
from verify_inputs import HERE,WORK,COMP,PAR,DATA,sha,scientific_imports

OUT=HERE/'runs'; EVID=HERE/'evidence'
cm,base,pm,pc=scientific_imports()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')

def read(p):return json.loads(Path(p).read_text())

def state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def check_inputs():
    m=read(EVID/'input_manifest.json')
    for e in m['required_items']+m['projected_review_documents']:
        assert sha(e['path'])==e['sha256'], 'Input changed: '+e['path']
    for e in read(EVID/'protocol_prefreeze.json')['pilot_code']:
        assert sha(e['path'])==e['sha256'], 'Pilot code changed after freeze: '+e['path']
    base.configure()

def specs():
    r=[]
    for s in [42,43,44]:
        for head in ['m8_k2_learned0','m8_k2_uniform']:
            r.append(dict(id=f'routing_{head}_seed{s}',kind='routing',seed=s,head=head,depth=1,lr=.001,clip=10.))
    for s in [100,101,102]:
        for head in ['lma3','lma3_clip1']:
            r.append(dict(id=f'parity_n20_{head}_seed{s}',kind='parity',seed=s,head=head,n=20,lr=.001,clip=pm.clip_threshold(head)))
    return r

def prefreeze():
    assert not (EVID/'protocol_prefreeze.json').exists()
    m=read(EVID/'input_manifest.json')
    protocol=dict(frozen_utc=pc.now(),authorization='Only routing/parity diagnosis and limited improvement; no MoST, no all-baseline retraining, no submission/publication',purpose='Distinguish early/endpoint training fit from held-out degradation under a matched single intervention, with trajectories; not universal causal identification',maximum_started_training_runs=12,maximum_wall_seconds_from_smoke_start=7200,run_specs=specs(),routing=dict(architecture='Unmodified original ComponentModel, one-layer GCN, d_model16 d_latent8 M8 k2, concatenated 36 memory tokens and query readout',training='AdamW decay1e-4, batch256, cosine T_max150, global clip10, original paired initialization and CPU minibatch stream 20000+seed; RNG30000+seed',preprocessing='Unchanged retained reconstructed ligand-only tensors, 7384/958/2171, train-only float64 mean/population SD, float32 standardized targets',primary='Full fitting-split standardized MSE at fixed endpoint, continuous paired uniform-minus-learned difference',fit_success='Endpoint fitting standardized MSE <= 0.5 (descriptive pilot threshold: halve constant-baseline loss; not a historical acceptance threshold)',smoke='First two epochs of seed42 learned baseline, exact original optimizer/batch/scheduler mechanics, retained and resumed with same state as run1; never an extra or discarded fit',horizon_rule='After smoke, lock E=min(150,max(2,floor(4800/(6*1.25*max(mean_smoke_epoch_seconds,1))))). Depends on timing only; scheduler remains T_max150. Reserve at least 2400 seconds for parity/evidence. No comparison fits before this lock.',primary_difference_rule='A consistent direction in all three paired endpoint MSE differences is descriptive support; no significance test. Report every difference regardless.',resource_caveat='Uniform removes unused routing maps; branch width and memory/computation structure matched, total parameter count differs and is reported.'),parity=dict(architecture='Unmodified original LMANetwork N20 D32 DL16 M8 k3 depth1 no positional features, 92 memory tokens, 12525 parameters',status='Training-fit diagnostic. No constructive representability proof under actual residual/LayerNorm/gated architecture; scalar theorem is not used as such a proof.',data='Original n20_seed100/101/102.npz verified byte identity and frozen generator equality; 4800/600/600 split',training='AdamW decay1e-4, batch256, cosine T_max100, LR .001 fixed before fitting; clip10 versus clip1 is the only intervention; original paired state and minibatch/RNG seeds',epochs=100,primary='Endpoint actual sampled fitting CE and accuracy, checked against original canonical count convention',fit_success='At least .99 fitting accuracy and CE <= .05; both reported separately as well',secondary='Validation/test CE, accuracy and original exact-binomial population metric; no universal remedy claim',intervention_basis='Existing retained N20 clip1 versus10 had partial successes; N40/80 could fail without crossing threshold10; two prior snapshots were not trajectories'),shared=dict(validation='Epoch1, every5, final; earliest strict minimum retained as secondary. Disable validation early stopping for equal fixed training budget; explicitly differs from old campaigns.',test='No test outcome scoring until all finished fits and checkpoint hashes are locked; endpoint and validation-selected predictors both reported',diagnostics='Fixed first64 original fitting rows at epoch0 and after every epoch: route entropy/assignment mass/global+graph load, product and module activation RMS/max/zero/nonfinite fractions, parameter and activation gradient norms/zero/None fractions, gates; every training minibatch preclip norm and clip factor; walltime and CUDA allocated/reserved plus sampled RSS.',diagnostic_scope='Fixed convenience batch is not entire data/population; no optimizer update, state/RNG preserved; uniform hard argmax tie collapse not interpreted semantically.',failure='Record all numeric/OOM failures and interruptions; no scientific-failure retries, seed replacement or outcome-dependent factor changes; implementation error stops for diagnosis.',budget='Hard deadline checked every minibatch. Priority routing; pending parity pairs can be skipped for budget, never silently extend. Smoke continuation remains run1.'),input_manifest_sha256=sha(EVID/'input_manifest.json'),resource_manifest_sha256=sha(EVID/'resources.json'),pilot_code=[dict(path=str(p),sha256=sha(p)) for p in [HERE/'pilot.py',HERE/'verify_inputs.py']],scientific_source_policy='Independent experiment manifest. Never edit old source/data/locks or pretend this new campaign matches original candidate artifacts.')
    dump(EVID/'protocol_prefreeze.json',protocol)
    print(json.dumps(dict(stage='protocol_prefrozen',sha256=sha(EVID/'protocol_prefreeze.json'),maximum_runs=12,maximum_wall_seconds=7200)))

def deadline():
    if time.time() >= read(EVID/'budget.json')['deadline_unix']:raise TimeoutError('Prespecified 7200-second wallclock budget exhausted')

def tensorstats(t):
    t=t.detach();finite=torch.isfinite(t);v=t[finite].float()
    return dict(entries=t.numel(),nonfinite_fraction=float((~finite).float().mean()),zero_fraction=float((t==0).float().mean()),rms=float(v.square().mean().sqrt()) if v.numel() else None,max_abs=float(v.abs().max()) if v.numel() else None)

def pgroup(n):
    if 'W_H' in n or 'W_k' in n:return 'routing'
    if 'interaction_' in n:
        a=n.split('interaction_',1)[1].split('.');return 'interaction_order'+str(int(a[1])+1)
    if 'order_gates' in n:return 'order_gates'
    if n.startswith('encoder') or n.startswith('embedding'):return 'encoder_or_embedding'
    if 'W_q' in n:return 'query'
    if 'W_v' in n:return 'value'
    if 'W_out' in n:return 'attention_output'
    if 'layer_norm' in n:return 'layer_norm'
    if 'ffn' in n:return 'ffn'
    return 'readout'

def snapshot(model,spec,train,mean,sd,epoch,folder):
    was=model.training;model.eval();model.zero_grad(set_to_none=True)
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    activ={};grads={};handles=[];routing_logits=[]
    def observe(name,t):
        if torch.is_tensor(t):
            activ[name]=tensorstats(t)
            if t.requires_grad:t.register_hook(lambda g,n=name:grads.__setitem__(n,tensorstats(g)))
    for name,module in model.named_modules():
        if isinstance(module,(nn.Linear,nn.LayerNorm,nn.GELU)) or name.endswith(('encoder','embedding','ffn')):
            handles.append(module.register_forward_hook(lambda m,i,o,n=name:observe(n+'.output',o)))
        if 'interaction_mlps.' in name:
            # Sequential order module; its input is the actual gathered Hadamard product.
            if isinstance(module,nn.Sequential):handles.append(module.register_forward_pre_hook(lambda m,i,n=name:observe(n+'.actual_product_input',i[0])))
        if name.endswith('W_H'):handles.append(module.register_forward_hook(lambda m,i,o:routing_logits.append(o.detach())))
        if name.endswith('interaction_projs.0'):handles.append(module.register_forward_pre_hook(lambda m,i:observe('bucket_z',i[0])))
    idx=torch.arange(64,device='cuda')
    if spec['kind']=='routing':
        args=base.get_batch(train,idx);mask=args[1];out=model(*args);target=((train['y'][idx].double()-mean)/sd).float();loss=(out-target).square().mean()
        layer=model.head.layers[0];pi=layer.last_routing.detach()
    else:
        mask=torch.ones_like(train['x'][idx],dtype=torch.bool);out=model(train['x'][idx]);loss=nn.functional.cross_entropy(out,train['y'][idx]);layer=model.layers[0]
        pi=torch.softmax(routing_logits[-1],dim=-1)
    loss.backward();bygroup={};parameters=[]
    for n,p in model.named_parameters():
        g=p.grad;row=dict(name=n,group=pgroup(n),numel=p.numel(),grad_none=g is None)
        if g is not None:row.update(tensorstats(g));row['l2']=float(g.norm())
        parameters.append(row);v=bygroup.setdefault(pgroup(n),dict(squared_l2=0.,zeros=0,entries=0,none_parameters=0))
        if g is None:v['none_parameters']+=1
        else:v['squared_l2']+=float(g.square().sum());v['zeros']+=int((g==0).sum());v['entries']+=g.numel()
    for v in bygroup.values():
        v['l2']=math.sqrt(v.pop('squared_l2'));v['zero_fraction']=v['zeros']/v['entries'] if v['entries'] else None
    valid=mask.unsqueeze(-1);counts=mask.sum(1);mass=(pi*valid).sum((0,1))/counts.sum()
    graph_mass=(pi*valid).sum(1)/counts[:,None]
    entropy=-(pi.clamp_min(torch.finfo(pi.dtype).tiny).log()*pi).sum(-1)
    loads=dict(entropy_nats=float(entropy[mask].mean()),entropy_normalized=float(entropy[mask].mean()/math.log(layer.M)),assignment_mass=mass.cpu().tolist(),global_load_penalty=float(layer.M*(mass-1/layer.M).square().sum()),mean_graph_load_penalty=float((layer.M*(graph_mass-1/layer.M).square().sum(-1)).mean()),probability_zero_fraction=float((pi[mask]==0).float().mean()),probability_one_fraction=float((pi[mask]==1).float().mean()))
    for h in handles:h.remove()
    assert sum('.actual_product_input' in n for n in activ)==layer.k, 'Missing actual order-product trajectory'
    hook_equivalence=None
    if epoch==0:
        saved_grads={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
        saved_out=out.detach().clone();saved_loss=loss.detach().clone()
        model.zero_grad(set_to_none=True)
        if spec['kind']=='routing':plain=model(*args);plain_loss=(plain-target).square().mean()
        else:plain=model(train['x'][idx]);plain_loss=nn.functional.cross_entropy(plain,train['y'][idx])
        plain_loss.backward()
        assert torch.equal(saved_out,plain.detach()) and torch.equal(saved_loss,plain_loss.detach())
        for n,p in model.named_parameters():
            a=saved_grads[n];b=p.grad
            assert (a is None)==(b is None)
            if a is not None:assert torch.allclose(a,b,atol=1e-7,rtol=1e-6), 'Instrumentation gradient mismatch: '+n
        hook_equivalence=dict(outputs_bitwise_equal=True,loss_bitwise_equal=True,parameter_gradients_allclose=True,atol=1e-7,rtol=1e-6,no_optimizer_updates=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()), 'Snapshot mutated model state'
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()), 'Snapshot consumed RNG'
    model.zero_grad(set_to_none=True);model.train(was)
    append(folder/'snapshots.jsonl',dict(epoch=epoch,fixed_training_rows=list(range(64)),loss=float(loss.detach()),routes=loads,order_gates=layer.order_gates.detach().cpu().tolist(),parameter_groups=bygroup,parameters=parameters,activations=activ,activation_gradients=grads,state_and_rng_preserved=True,initial_instrumentation_equivalence=hook_equivalence))

def build(spec):
    if spec['kind']=='routing':
        model=cm.build_model(spec['seed'],spec['head'],1).cuda();train=base.load_split('train');val=base.load_split('val')
        mean=float(train['y'].double().mean());sd=float(train['y'].double().std(unbiased=False))
    else:
        model=pm.build_model(spec['seed'],spec['head'],20).cuda();train=pc.load_data(20,spec['seed'],'train','cuda');val=pc.load_data(20,spec['seed'],'val')
        mean=sd=None
    torch.manual_seed(30000+spec['seed'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['lr'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=150 if spec['kind']=='routing' else 100)
    gen=torch.Generator().manual_seed(20000+spec['seed'])
    return model,optimizer,scheduler,gen,train,val,mean,sd

def validation(model,spec,val,mean,sd):
    if spec['kind']=='routing':return base.metrics(val['y'].double().cpu().numpy(),base.predict(model,val,mean,sd))
    logits=pc.canonical_logits(model,20);return pc.metrics(val['y'].numpy(),logits[val['count']])

def rss_mib():
    return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])/1024

def run_epochs(spec,folder,end,resume=False,smoke=False):
    model,opt,scheduler,gen,train,val,mean,sd=build(spec)
    history=[];best=math.inf;best_epoch=0;best_state=None;first=1;wall_before=0.
    if resume:
        ck=torch.load(folder/'smoke_continuation.pt',map_location='cpu',weights_only=True)
        assert ck['spec']==spec
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);gen.set_state(ck['generator'])
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
        history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];best_state=ck['best_state'];first=3;wall_before=ck['wall_seconds']
    else:
        assert not folder.exists();folder.mkdir(parents=True)
        append(EVID/'started_runs.jsonl',dict(spec=spec,started_utc=pc.now(),count=len(list(OUT.iterdir()))))
        assert len(list(OUT.iterdir()))<=12
        snapshot(model,spec,train,mean,sd,0,folder)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();peak_rss=rss_mib()
    dump(folder/('resumed.json' if resume else 'started.json'),dict(spec=spec,started_utc=pc.now(),resume_same_smoke_run=resume,mean=mean,sd=sd,total_parameters=sum(p.numel() for p in model.parameters())))
    try:
        for epoch in range(first,end+1):
            deadline();t=time.perf_counter();model.train();order=torch.randperm(len(train['y']),generator=gen).cuda()
            sumloss=0.;correct=0;maxnorm=0.;clipped=0
            for batch_start in range(0,len(order),256):
                deadline();idx=order[batch_start:batch_start+256];opt.zero_grad(set_to_none=True)
                if spec['kind']=='routing':
                    out=model(*base.get_batch(train,idx));target=((train['y'][idx].double()-mean)/sd).float();loss=(out-target).square().mean()
                else:out=model(train['x'][idx]);loss=nn.functional.cross_entropy(out,train['y'][idx]);correct+=int((out.detach().argmax(1)==train['y'][idx]).sum())
                loss.backward();norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],spec['clip']);f=float(norm)
                if not math.isfinite(float(loss)) or not math.isfinite(f):raise FloatingPointError('Nonfinite minibatch loss/gradient')
                opt.step();sumloss+=float(loss.detach())*len(idx);maxnorm=max(maxnorm,f);clipped+=f>spec['clip']
                append(folder/'minibatches.jsonl',dict(epoch=epoch,batch_index=batch_start//256,rows=len(idx),prediction_loss=float(loss.detach()),preclip_global_gradient_norm=f,clip_factor=min(1.,spec['clip']/(f+1e-6))))
            scheduler.step()
            row=dict(epoch=epoch,online_training_loss=sumloss/len(order),maximum_preclip_gradient_norm=maxnorm,clipped_minibatches=clipped,total_minibatches=math.ceil(len(order)/256),learning_rate_after_epoch=scheduler.get_last_lr()[0])
            if spec['kind']=='parity':row['online_training_accuracy']=correct/len(order)
            if epoch==1 or epoch%5==0 or (epoch==end and not smoke):
                score=validation(model,spec,val,mean,sd);value=score['rmse' if spec['kind']=='routing' else 'cross_entropy'];row['validation']=score
                if value<best:best=value;best_epoch=epoch;best_state=state_cpu(model)
            snapshot(model,spec,train,mean,sd,epoch,folder)
            torch.cuda.synchronize();row.update(epoch_wall_seconds=time.perf_counter()-t,cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,cuda_reserved_mib=torch.cuda.memory_reserved()/2**20,rss_mib=rss_mib());peak_rss=max(peak_rss,row['rss_mib'])
            history.append(row);append(folder/'epochs.jsonl',row)
            dump(EVID/'progress.json',dict(updated_utc=pc.now(),run_id=spec['id'],epoch=epoch,end_epoch=end,started_runs=len(list(OUT.iterdir())),elapsed_budget_seconds=time.time()-read(EVID/'budget.json')['start_unix']))
            if not smoke and (epoch%10==0 or epoch==end):print(json.dumps(dict(stage='training',id=spec['id'],epoch=epoch,cap=end,wall_seconds=wall_before+time.perf_counter()-start)),flush=True)
        wall=wall_before+time.perf_counter()-start
        if smoke:
            torch.save(dict(spec=spec,model=state_cpu(model),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),generator=gen.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,best=best,best_epoch=best_epoch,best_state=best_state,wall_seconds=wall),folder/'smoke_continuation.pt')
            dump(EVID/'smoke_timing.json',dict(run_id=spec['id'],started_runs=1,epochs=2,epoch_seconds=[r['epoch_wall_seconds'] for r in history],wall_seconds=wall,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_sampled_rss_mib=peak_rss,continuation_sha256=sha(folder/'smoke_continuation.pt'),note='Same run continues; no metrics used to set comparisons/horizon'))
            print(json.dumps(dict(stage='original_baseline_smoke_complete',id=spec['id'],counted_runs=1,epoch_seconds=[r['epoch_wall_seconds'] for r in history],wall_seconds=wall,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
            return
        torch.save(dict(state_dict=state_cpu(model),spec=spec),folder/'final.pt');torch.save(dict(state_dict=best_state,spec=spec),folder/'best_validation.pt')
        result=dict(spec=spec,status='completed',epochs=len(history),history=history,wall_seconds=wall,total_parameters=sum(p.numel() for p in model.parameters()),target_mean=mean,target_sd=sd,best_validation_value=best,best_validation_epoch=best_epoch,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_sampled_rss_mib=peak_rss,final_checkpoint_sha256=sha(folder/'final.pt'),best_validation_checkpoint_sha256=sha(folder/'best_validation.pt'),finished_utc=pc.now())
        dump(folder/'result.json',result)
        print(json.dumps(dict(stage='run_complete',id=spec['id'],epochs=len(history),wall_seconds=wall)),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=state_cpu(model),spec=spec),folder/'interrupted_or_failed_state.pt')
        scientific=isinstance(exc,(FloatingPointError,torch.OutOfMemoryError));budget=isinstance(exc,TimeoutError)
        dump(folder/'result.json',dict(spec=spec,status='failed' if scientific else 'budget_terminated' if budget else 'implementation_error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),epochs=len(history),history=history,wall_seconds=wall_before+time.perf_counter()-start,finished_utc=pc.now()))
        if not scientific:raise
    finally:
        del model,opt,scheduler,gen,train,val;torch.cuda.empty_cache()

def smoke():
    check_inputs();assert not (EVID/'budget.json').exists();OUT.mkdir(exist_ok=False)
    start=time.time();dump(EVID/'budget.json',dict(start_unix=start,start_utc=pc.now(),deadline_unix=start+7200,maximum_runs=12))
    print(json.dumps(dict(stage='ACTUAL_TRAINING_START',spec=specs()[0],gpu=torch.cuda.get_device_name(0),time_utc=pc.now(),deadline_utc_unix=start+7200)),flush=True)
    run_epochs(specs()[0],OUT/specs()[0]['id'],2,smoke=True)

def lock():
    assert not (EVID/'protocol_locked.json').exists();s=read(EVID/'smoke_timing.json');mean=np.mean(s['epoch_seconds'])
    epochs=min(150,max(2,math.floor(4800/(6*1.25*max(mean,1.)))))
    dump(EVID/'protocol_locked.json',dict(locked_utc=pc.now(),prefreeze_sha256=sha(EVID/'protocol_prefreeze.json'),smoke_timing_sha256=sha(EVID/'smoke_timing.json'),routing_epochs=epochs,parity_epochs=100,routing_cosine_T_max=150,parity_cosine_T_max=100,mean_smoke_epoch_seconds=float(mean),forecast_routing_seconds=6*epochs*1.25*max(mean,1.),horizon_rule='Timing-only formula predeclared before any fitting outcomes; no comparison fit started',run_specs=specs()))
    print(json.dumps(dict(stage='comparisons_locked',routing_epochs=epochs,parity_epochs=100,forecast_routing_seconds=6*epochs*1.25*max(mean,1.),maximum_runs=12)),flush=True)

def run():
    check_inputs();p=read(EVID/'protocol_locked.json');assert p['prefreeze_sha256']==sha(EVID/'protocol_prefreeze.json')
    assert p['run_specs']==specs()
    for i,spec in enumerate(specs()):
        deadline();folder=OUT/spec['id']
        if (folder/'result.json').exists():continue
        cap=p['routing_epochs' if spec['kind']=='routing' else 'parity_epochs']
        run_epochs(spec,folder,cap,resume=(i==0))
    results=[read(OUT/s['id']/'result.json') for s in specs()]
    dump(EVID/'all_fits_locked.json',dict(locked_utc=pc.now(),protocol_sha256=sha(EVID/'protocol_locked.json'),started_runs=len(list(OUT.iterdir())),results=[dict(id=r['spec']['id'],status=r['status'],record_sha256=sha(OUT/r['spec']['id']/'result.json'),final_checkpoint_sha256=r.get('final_checkpoint_sha256'),best_validation_checkpoint_sha256=r.get('best_validation_checkpoint_sha256')) for r in results],elapsed_budget_seconds=time.time()-read(EVID/'budget.json')['start_unix']))
    print(json.dumps(dict(stage='all_fits_locked',runs=len(results),elapsed_seconds=time.time()-read(EVID/'budget.json')['start_unix'])),flush=True)

@torch.no_grad()
def evaluation():
    check_inputs();lock=read(EVID/'all_fits_locked.json');rows=[]
    for item in lock['results']:
        deadline();folder=OUT/item['id'];assert sha(folder/'result.json')==item['record_sha256'];r=read(folder/'result.json');spec=r['spec']
        if r['status']!='completed':rows.append(dict(spec=spec,status=r['status']));continue
        model,opt,sched,gen,train,val,mean,sd=build(spec);del opt,sched,gen
        row=dict(spec=spec,status=r['status'],total_parameters=r['total_parameters'],epochs=r['epochs'],wall_seconds=r['wall_seconds'],peak_allocated_mib=r['peak_allocated_mib'])
        for ckname in ['final','best_validation']:
            cp=folder/(ckname+'.pt');assert sha(cp)==item[ckname+'_checkpoint_sha256'];model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']);model.eval();scores={}
            if spec['kind']=='routing':
                for split,blob in [('train',train),('val',val),('test',base.load_split('test'))]:
                    y=blob['y'].double().cpu().numpy();pred=base.predict(model,blob,mean,sd);scores[split]=base.metrics(y,pred)
                    if split=='train':scores[split]['standardized_mse']=float(np.mean(((pred-y)/sd)**2));scores[split]['fit_success_mse_le_05']=scores[split]['standardized_mse']<=.5
                    np.savez_compressed(folder/f'{ckname}_{split}_predictions.npz',truth=y,prediction=pred)
            else:
                canonical=pc.canonical_logits(model,20)
                for split in ['train','val','test']:
                    blob=pc.load_data(20,spec['seed'],split,'cuda');y=blob['y'].cpu().numpy();logits=canonical[blob['count']];scores[split]=pc.metrics(y,logits)
                    actual=[]
                    for start in range(0,len(y),256):actual.append(model(blob['x'][start:start+256]).double().cpu().numpy())
                    actual=np.concatenate(actual);assert np.allclose(actual,logits,atol=5e-4,rtol=5e-4), 'Canonical/actual row disagreement'
                    scores[split]['actual_row_metrics']=pc.metrics(y,actual);scores[split]['max_actual_canonical_logit_difference']=float(np.max(np.abs(actual-logits)))
                    if split=='train':scores[split].update(fit_success_accuracy_ge_099=scores[split]['accuracy']>=.99,fit_success_ce_le_005=scores[split]['cross_entropy']<=.05,fit_success_both=scores[split]['accuracy']>=.99 and scores[split]['cross_entropy']<=.05)
                    np.savez_compressed(folder/f'{ckname}_{split}_predictions.npz',truth=y,count=blob['count'],logits=logits,actual_row_logits=actual,canonical_logits=canonical)
                scores['population']=pc.population_metrics(20,canonical)
            row[ckname]=scores
        rows.append(row);del model,train,val;torch.cuda.empty_cache()
    dump(EVID/'evaluation.json',dict(evaluated_utc=pc.now(),all_fits_lock_sha256=sha(EVID/'all_fits_locked.json'),rows=rows,elapsed_budget_seconds=time.time()-read(EVID/'budget.json')['start_unix']))
    print(json.dumps(dict(stage='evaluation_complete',rows=rows,elapsed_seconds=time.time()-read(EVID/'budget.json')['start_unix']),indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prefreeze','smoke','lock','run','evaluate']);arg=a.parse_args()
    {'prefreeze':prefreeze,'smoke':smoke,'lock':lock,'run':run,'evaluate':evaluation}[arg.stage]()
