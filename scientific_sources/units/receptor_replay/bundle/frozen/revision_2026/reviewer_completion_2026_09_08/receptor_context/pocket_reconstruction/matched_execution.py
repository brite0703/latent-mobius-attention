"""Common-cohort accumulated updates and recoverable complete-epoch boundaries.

No fit or CUDA allocation occurs at import. Frozen predecessor sources are read,
not patched; generic checkpoint/RNG operations are reused by explicit import.
"""
from pathlib import Path
import sys,hashlib,json,math,random,time
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import sequence_execution as previous
from sequence_data import SequenceStore
from sequence_encoder import VOCAB
import matched_models as models

CACHE=HERE/'matched_inputs/tensors_v1'
EPOCHS=100
PATIENCE=8
EFFECTIVE_BATCH=256
MAX_RECORDS=32
MAX_PADDED_TOKENS=32768
# Padded contact inputs plus the two mapped activation shapes. This is an
# explicit shape bound, not an estimate of total training or GPU memory.
MAX_CONTACT_SLOTS=32*57*(216+32+16)
EXECUTION_VERSION='matched_common_cohort_v1'
configure=previous.configure
indexed_blob=previous.indexed_blob
scalers=previous.scalers
to_cpu=previous.to_cpu

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_split(split,manifest=None):
    if split not in ('train','val','test'):raise ValueError('Unknown split')
    manifest=manifest or json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
    receipt=next(row for row in manifest['files'] if row['split']==split)
    path=CACHE/(split+'.pt')
    if sha(path)!=receipt['sha256']:raise ValueError('Cached split changed')
    blob=indexed_blob(torch.load(path,map_location='cpu',weights_only=True))
    n=len(blob['ids'])
    if n!=manifest['split_counts'][split] or n!=len(blob['sequences']):raise ValueError('Split count mismatch')
    if blob['X'].shape!=(n,57,53) or blob['contact'].shape!=(n,57,216):raise ValueError('Common feature shapes changed')
    if blob['mask'].shape!=(n,57) or blob['mask'].dtype!=torch.bool or blob['adj'].shape!=(n,57,57):raise ValueError('Graph masks or adjacency changed')
    for key in ('X','contact','adj','y'):
        if blob[key].dtype!=torch.float32 or not bool(torch.isfinite(blob[key]).all()):raise ValueError('Invalid cache dtype/value')
    if bool((blob['mask'].sum(1)<1).any()):raise ValueError('Empty ligand')
    if not torch.equal(blob['mask'],torch.arange(57)[None]<blob['mask'].sum(1)[:,None]):raise ValueError('Non-prefix atom mask')
    expected=[row['pdbid'] for row in manifest['rows'] if row['split']==split]
    if blob['ids']!=expected:raise ValueError('Canonical record order changed')
    if split=='train':
        mean,sd=scalers(blob)
        scale=manifest['target_scale']
        if mean!=scale['mean'] or not math.isclose(sd,scale['population_sd'],rel_tol=0,abs_tol=1e-14):
            raise ValueError('Fitting-only target scaling changed')
    return blob

class CommonStore(SequenceStore):
    """One input-only batch rule is shared by every setting and head."""
    def __init__(self,blobs):
        self.chains={};self.atom_sizes={};self.by_id={}
        for blob in blobs:
            for i,key in enumerate(blob['ids']):
                if key in self.chains:raise ValueError('An ID occurs in multiple common splits')
                sequence=blob['sequences'][i]
                if not isinstance(sequence,str):raise ValueError('Sequence must be the original stored string')
                parts=sequence.split(':')
                if not all(parts) or any(not set(p)<=set(VOCAB) for p in parts):raise ValueError('Invalid supplied sequence')
                self.chains[key]=tuple(torch.tensor([VOCAB[a] for a in p],dtype=torch.uint8) for p in parts)
                self.atom_sizes[key]=int(blob['mask'][i].sum())
                self.by_id[key]={'pdbid':key,'sequence':sequence}
        self.sizes={key:(len(chains),max(map(len,chains))) for key,chains in self.chains.items()}

    def footprint(self,ids):
        if not ids:raise ValueError('Empty batch footprint')
        return {'records':len(ids),
            'padded_tokens':sum(self.sizes[k][0] for k in ids)*max(self.sizes[k][1] for k in ids),
            'padded_atoms':len(ids)*max(self.atom_sizes[k] for k in ids),
            'contact_slots':len(ids)*max(self.atom_sizes[k] for k in ids)*(216+32+16)}

    def chunks(self,ids,max_records=MAX_RECORDS,max_tokens=MAX_PADDED_TOKENS,max_contact_slots=MAX_CONTACT_SLOTS):
        if any(type(x)!=int or x<1 for x in (max_records,max_tokens,max_contact_slots)):
            raise ValueError('Positive integer batch limits required')
        def permitted(items):
            f=self.footprint(items)
            return f['records']<=max_records and f['padded_tokens']<=max_tokens and f['contact_slots']<=max_contact_slots
        current=[]
        for key in ids:
            if not permitted([key]):raise ValueError('Single record exceeds the common input envelope: '+key)
            if current and not permitted(current+[key]):
                yield current
                current=[]
            current.append(key)
        if current:yield current

def collate(model,blob,store,ids):
    arguments,truth=previous.collate(model,blob,store,ids)
    if model.contact_map is not None:
        indices=torch.tensor([blob['row_by_id'][key] for key in ids],dtype=torch.long)
        end=arguments['mask'].shape[1]
        parameter=next(model.parameters())
        arguments['contact']=blob['contact'].index_select(0,indices)[:,:end].to(device=parameter.device,dtype=parameter.dtype)
    return arguments,truth

def finite_output(model,arguments):
    prediction,product=previous.finite_output(model,arguments)
    if len(prediction)!=len(arguments['mask']):raise FloatingPointError('Prediction batch length mismatch')
    return prediction,product

def effective_step(model,optimizer,blob,store,ids,mean,sd,*,max_records=MAX_RECORDS,
                   max_tokens=MAX_PADDED_TOKENS,max_contact_slots=MAX_CONTACT_SLOTS):
    if not ids or not math.isfinite(mean) or not math.isfinite(sd) or sd<=0:raise ValueError('Invalid batch/scale')
    model.train();optimizer.zero_grad(set_to_none=True)
    sse=[];products=[];footprints=[]
    for chunk in store.chunks(ids,max_records,max_tokens,max_contact_slots):
        arguments,truth=collate(model,blob,store,chunk)
        prediction,product=finite_output(model,arguments)
        squared=(prediction-(truth-mean)/sd).square().sum()
        if not bool(torch.isfinite(squared)):raise FloatingPointError('Nonfinite normalized loss')
        # A final effective batch of 72 divides by 72, not 256 or chunk length.
        (squared/len(ids)).backward()
        sse.append(float(squared.detach()));footprints.append(store.footprint(chunk))
        if product is not None:products.append(product)
    parameters=list(model.parameters())
    if any(p.grad is None for p in parameters):raise FloatingPointError('Absent whole-model gradient')
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):raise FloatingPointError('Nonfinite gradient')
    norm=nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise FloatingPointError('Nonfinite optimizer state output')
    return {'records':len(ids),'normalized_sse':math.fsum(sse),'preclip_gradient_norm':float(norm),
        'clipped':bool(norm>10),'microbatches':len(footprints),'optimizer_steps':1,
        'maximum_padded_tokens':max(f['padded_tokens'] for f in footprints),
        'maximum_contact_slots':max(f['contact_slots'] for f in footprints),
        'maximum_abs_cp_product':max(products) if products else None}

@torch.no_grad()
def predict(model,blob,store,mean,sd,*,max_records=MAX_RECORDS,max_tokens=MAX_PADDED_TOKENS,max_contact_slots=MAX_CONTACT_SLOTS):
    model.eval();values=[]
    for ids in store.chunks(blob['ids'],max_records,max_tokens,max_contact_slots):
        arguments,_=collate(model,blob,store,ids)
        prediction,_=finite_output(model,arguments)
        values.append(prediction.detach().cpu().double()*sd+mean)
    output=torch.cat(values).numpy()
    if output.shape!=(len(blob['ids']),) or not np.isfinite(output).all():raise FloatingPointError('Invalid inverse-scaled output')
    return output

def metric(truth,prediction):
    truth=np.asarray(truth,dtype=np.float64)
    if not np.isfinite(truth).all():raise ValueError('Nonfinite truth')
    return previous.metric(truth,prediction)

def make_runtime(seed,setting,head,lr,device='cpu',dtype=torch.float32,epochs=EPOCHS,
                 data_manifest_sha256=None,effective_batch=EFFECTIVE_BATCH,max_records=MAX_RECORDS,
                 max_tokens=MAX_PADDED_TOKENS,max_contact_slots=MAX_CONTACT_SLOTS):
    if not isinstance(epochs,int) or epochs<1 or not math.isfinite(lr) or lr<=0:raise ValueError('Invalid training schedule')
    if any(type(v)!=int or v<1 for v in (effective_batch,max_records,max_tokens,max_contact_slots)):
        raise ValueError('Invalid update or microbatch size')
    model=models.build_model(seed,setting,head).to(device=device,dtype=dtype)
    torch.manual_seed(30000+seed);random.seed(30000+seed);np.random.seed(30000+seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    policy=dict(max_records=max_records,max_tokens=max_tokens,max_contact_slots=max_contact_slots)
    return dict(model=model,optimizer=optimizer,scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs),
        generator=torch.Generator().manual_seed(20000+seed),
        identity=dict(seed=seed,setting=setting,head=head,lr=lr,epochs=epochs,dtype=str(dtype),
            execution_version=EXECUTION_VERSION,data_manifest_sha256=data_manifest_sha256,
            effective_batch=effective_batch,microbatch_policy=policy),
        epochs_completed=0,optimizer_steps=0,best_validation_rmse=float('inf'),best_epoch=None,
        best_state=None,bad_checks=0,history=[])

def finish_epoch(runtime,train,validation,store,mean,sd):
    start=time.perf_counter();model=runtime['model'];identity=runtime['identity'];policy=identity['microbatch_policy']
    permutation=torch.randperm(len(train['ids']),generator=runtime['generator']).tolist()
    steps=[]
    for offset in range(0,len(permutation),identity['effective_batch']):
        ids=[train['ids'][i] for i in permutation[offset:offset+identity['effective_batch']]]
        steps.append(effective_step(model,runtime['optimizer'],train,store,ids,mean,sd,**policy))
        runtime['optimizer_steps']+=1
    runtime['scheduler'].step();runtime['epochs_completed']+=1;epoch=runtime['epochs_completed']
    products=[s['maximum_abs_cp_product'] for s in steps if s['maximum_abs_cp_product'] is not None]
    row=dict(epoch=epoch,normalized_training_mse=math.fsum(s['normalized_sse'] for s in steps)/len(train['ids']),
        optimizer_steps=len(steps),effective_batch_records=[s['records'] for s in steps],
        microbatches=sum(s['microbatches'] for s in steps),clipped_effective_batches=sum(s['clipped'] for s in steps),
        maximum_preclip_gradient_norm=max(s['preclip_gradient_norm'] for s in steps),
        maximum_padded_tokens=max(s['maximum_padded_tokens'] for s in steps),
        maximum_contact_slots=max(s['maximum_contact_slots'] for s in steps),
        maximum_abs_cp_product=max(products) if products else None,next_learning_rate=float(runtime['optimizer'].param_groups[0]['lr']))
    if epoch==1 or epoch%5==0 or epoch==identity['epochs']:
        score=metric(validation['y'].numpy(),predict(model,validation,store,mean,sd,**policy))['rmse']
        row['validation_rmse']=score
        if score<runtime['best_validation_rmse']:
            runtime.update(best_validation_rmse=score,best_epoch=epoch,
                best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},bad_checks=0)
        else:runtime['bad_checks']+=1
    if next(model.parameters()).is_cuda:torch.cuda.synchronize()
    row['completed_epoch_seconds']=time.perf_counter()-start
    runtime['history'].append(row)
    return row

# These functions operate on a passed runtime, including its complete matched
# identity and data hash. They neither construct nor alter predecessor models.
snapshot=previous.snapshot
save_snapshot=previous.save_snapshot
restore_snapshot=previous.restore_snapshot
