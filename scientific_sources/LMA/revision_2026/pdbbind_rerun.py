"""Reproducible ligand-only LP-PDBBind head comparison with local data provenance.

The historical preprocessing and graph code are unavailable. This runner is a
declared reconstruction, not an exact reproduction of the submitted tables.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from lma_revision import original, MaskedLMAHead


class MeanHead(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
    def forward(self,h,mask):
        mean=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.net(mean).squeeze(-1)


class PMAHead(nn.Module):
    """One learned query with multihead pooling and residual feedforward output."""
    def __init__(self,d):
        super().__init__()
        self.query=nn.Parameter(torch.randn(1,1,d)/math.sqrt(d))
        self.attention=nn.MultiheadAttention(d,4,batch_first=True,dropout=0)
        self.norm=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.head=nn.Linear(d,1)
    def forward(self,h,mask):
        q=self.query.expand(h.shape[0],-1,-1)
        value,_=self.attention(q,h,h,key_padding_mask=~mask,need_weights=False)
        z=self.norm(q+value)
        return self.head(z+self.ffn(z)).flatten()


class GraphEncoder(nn.Module):
    """Declared GCN reconstruction: symmetric normalized adjacency, LN and GELU."""
    def __init__(self,d_in,d,depth):
        super().__init__()
        self.layers=nn.ModuleList(nn.Linear(d_in if i==0 else d,d) for i in range(depth))
        self.norms=nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))
    def forward(self,x,mask,adj):
        h=x
        for linear,norm in zip(self.layers,self.norms):
            h=torch.nn.functional.gelu(norm(linear(torch.bmm(adj,h))))*mask.unsqueeze(-1)
        return h


class AffinityModel(nn.Module):
    def __init__(self,encoder,head,backbone):
        super().__init__()
        self.encoder,self.head,self.backbone=encoder,head,backbone
    def forward(self,x,mask,adj=None):
        h=self.encoder(x) if self.backbone=="mlp" else self.encoder(x,mask,adj)
        return self.head(h,mask)


def build_model(seed,name,d_in,args):
    torch.manual_seed(seed)
    encoder=(original.NodeEncoder(d_in,args.d_model,dropout=.1) if args.backbone=="mlp"
             else GraphEncoder(d_in,args.d_model,args.graph_depth))
    torch.manual_seed(10000+seed)
    if name=="mean":head=MeanHead(args.d_model)
    elif name=="pma":head=PMAHead(args.d_model)
    elif name=="transformer":head=original.TransformerHead(args.d_model,4,1)
    elif name=="deepsets":
        ref=MaskedLMAHead(args.d_model,args.d_latent,args.buckets,2,1)
        budget=sum(p.numel() for p in ref.parameters())
        widths=range(4,129)
        width=min(widths,key=lambda w:abs(sum(p.numel() for p in original.DeepSetsHead(args.d_model,w).parameters())-budget))
        torch.manual_seed(10000+seed)
        head=original.DeepSetsHead(args.d_model,width)
    else:
        head=MaskedLMAHead(args.d_model,args.d_latent,args.buckets,int(name[-1]),1)
    return AffinityModel(encoder,head,args.backbone)


def validate_blob(blob):
    x,mask,y,adj=blob["X"],blob["mask"],blob["y"],blob["adj"]
    assert x.ndim==3 and y.ndim==1 and mask.ndim==2
    assert x.shape[:2]==mask.shape and len(y)==len(x)==len(blob["ids"])
    assert adj.shape==(len(x),x.shape[1],x.shape[1])
    assert mask.dtype==torch.bool and bool(mask.any(1).all())
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    assert bool(torch.isfinite(adj).all()) and torch.allclose(adj,adj.transpose(1,2))
    assert not bool((x[~mask]!=0).any())
    assert bool((adj*(~mask).unsqueeze(-1)==0).all())


def predict(model,blob,mean,sd,batch_size):
    model.eval()
    parts=[]
    with torch.no_grad():
        for start in range(0,len(blob["y"]),batch_size):
            stop=start+batch_size
            p=model(blob["X"][start:stop],blob["mask"][start:stop],blob["adj"][start:stop])
            assert p.ndim==1
            parts.append(p*sd+mean)
    return torch.cat(parts)


def metrics(y,p):
    y,p=np.asarray(y,dtype=np.float64),np.asarray(p,dtype=np.float64)
    if y.ndim!=1 or p.shape!=y.shape or not np.isfinite(p).all():
        raise ValueError("Invalid regression predictions or labels")
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(np.abs(y-p))),
                pearson=float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None)


def run(seed,name,args,blobs,mean,sd):
    model=build_model(seed,name,blobs["train"]["X"].shape[-1],args).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    generator=torch.Generator().manual_seed(20000+seed)
    train=blobs["train"]
    target=(train["y"]-mean)/sd
    best=float("inf");best_state=None;best_epoch=0;bad_checks=0
    history=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start_wall=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        order=torch.randperm(len(target),generator=generator).cuda()
        squared=torch.zeros((),device="cuda")
        largest_norm=torch.zeros((),device="cuda")
        for start in range(0,len(target),args.batch_size):
            idx=order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            p=model(train["X"][idx],train["mask"][idx],train["adj"][idx])
            assert p.shape==target[idx].shape
            loss=(p-target[idx]).square().mean()
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
            largest_norm=torch.maximum(largest_norm,norm.detach())
            optimizer.step()
            squared+=loss.detach()*len(idx)
        scheduler.step()
        if not bool(torch.isfinite(squared)) or not bool(torch.isfinite(largest_norm)):
            raise RuntimeError(f"Nonfinite training: {seed}/{name}/{epoch}")
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            val_prediction=predict(model,blobs["val"],mean,sd,args.batch_size)
            val=float(torch.sqrt((val_prediction-blobs["val"]["y"]).square().mean()))
            history.append(dict(epoch=epoch,training_normalized_mse=float(squared/len(target)),
                                validation_rmse=val,maximum_preclip_gradient_norm=float(largest_norm)))
            if val<best:
                best,best_epoch,bad_checks=val,epoch,0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:bad_checks+=1
            if epoch%25==0:
                print(json.dumps(dict(stage="training",seed=seed,model=name,backbone=args.backbone,
                                      epoch=epoch,val_rmse=val)),flush=True)
            if bad_checks>=args.patience_checks:break
    model.load_state_dict(best_state)
    # Test is accessed only after validation checkpoint selection is complete.
    test_prediction=predict(model,blobs["test"],mean,sd,args.batch_size).cpu().numpy()
    test_truth=blobs["test"]["y"].cpu().numpy()
    evaluation=metrics(test_truth,test_prediction)
    torch.cuda.synchronize()
    stem=f"{args.backbone}{args.graph_depth if args.backbone!='mlp' else ''}_{name}_seed{seed}"
    np.savez_compressed(args.output_dir/f"{stem}_predictions.npz",ids=np.asarray(blobs["test"]["ids"]),
                        truth=test_truth,prediction=test_prediction)
    torch.save(dict(state_dict=best_state,target_mean=float(mean),target_sd=float(sd)),args.output_dir/f"{stem}_checkpoint.pt")
    row=dict(seed=seed,model=name,backbone=args.backbone,graph_depth=args.graph_depth if args.backbone!="mlp" else None,
        best_epoch=best_epoch,epochs_run=epoch,validation_rmse=best,test=evaluation,
        wall_seconds=time.perf_counter()-start_wall,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        total_parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        head_parameters=sum(p.numel() for p in model.head.parameters()),history=history,
        prediction_file=f"{stem}_predictions.npz",checkpoint_file=f"{stem}_checkpoint.pt")
    (args.output_dir/f"{stem}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    print(json.dumps({k:row[k] for k in ("seed","model","backbone","test","wall_seconds")}),flush=True)
    del model,optimizer
    torch.cuda.empty_cache()
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--models",nargs="+",default=["mean","deepsets","pma","lma1","lma2","lma3"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--patience-checks",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--d-model",type=int,default=16)
    parser.add_argument("--d-latent",type=int,default=8)
    parser.add_argument("--buckets",type=int,default=8)
    parser.add_argument("--backbone",choices=["mlp","gcn"],default="mlp")
    parser.add_argument("--graph-depth",type=int,default=1)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA environment required")
    torch.set_num_threads(4)
    blobs={}
    for split in ("train","val","test"):
        blob=torch.load(args.data_dir/f"pdbbind_{split}.pt",weights_only=True,map_location="cpu")
        validate_blob(blob)
        blobs[split]={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in blob.items()}
    assert not set(blobs["train"]["ids"])&set(blobs["test"]["ids"])
    assert not set(blobs["train"]["ids"])&set(blobs["val"]["ids"])
    mean=blobs["train"]["y"].mean();sd=blobs["train"]["y"].std(unbiased=False)
    assert float(sd)>0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(created_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,device=torch.cuda.get_device_name(),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/"lma_revision.py"]},
        data_audit_sha256=hashlib.sha256((args.data_dir/"data_audit.json").read_bytes()).hexdigest(),
        split_sizes={k:len(v["y"]) for k,v in blobs.items()},target_mean=float(mean),target_sd=float(sd),
        balance_loss_weight=0,gradient_clip=10,weight_decay=1e-4,validation_interval=5,
        purpose="fresh ligand-only LP-PDBBind reconstruction with shared backbone per comparison; no claim of exact historical replication",
        parameter_matching="DeepSets head approximately matched to LMA k2; same backbone architecture and initialization across heads",
        null_test=metrics(blobs["test"]["y"].cpu().numpy(),np.full(len(blobs["test"]["y"]),float(mean))))
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    rows=[]
    for seed in args.seeds:
        for name in args.models:
            rows.append(run(seed,name,args,blobs,mean,sd))
            (args.output_dir/"summary.json").write_text(json.dumps(dict(config=config,rows=rows),indent=2),encoding="utf-8")


if __name__=="__main__":main()
