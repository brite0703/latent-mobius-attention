"""Exact native cache extraction on supplied inputs; no scientific data reader."""
from copy import deepcopy

import torch
from torch.nn import functional as F

import campaign_lifecycle as life
import costs
import execution
from models import FrozenLmaFeatures
from native_costs import module_snapshot


def extract(parent,layer_path,payload,parent_sha):
    if torch.get_num_threads()!=1 or torch.cuda.is_initialized():
        raise ValueError("Native caches use one CPU thread without CUDA initialization")
    if payload["split"] not in ("train","val","test") or len(payload["ids"])!=len(payload["truth"]):
        raise ValueError("Unaligned native cache payload")
    if any(p.device.type!="cpu" or p.dtype!=torch.float32 for p in parent.parameters()):
        raise ValueError("Native parent cache extraction requires CPU float32 parameters")
    before,saved,rng = module_snapshot(parent),deepcopy(payload),torch.get_rng_state().clone()
    predictions,buckets = [],[]
    try:
        model = deepcopy(parent).eval()
        frozen = FrozenLmaFeatures(model,layer_path).eval()
        layer = model.get_submodule(layer_path)
        for start in range(0,len(payload["ids"]),256):
            inputs = {k:v[start:start+256] for k,v in payload["inputs"].items()}
            f0,z = frozen(**inputs)
            observed = []
            handle = layer.register_forward_pre_hook(lambda _,args:observed.append(args))
            try:
                with torch.no_grad():
                    direct = model(**inputs)
                    if len(observed)!=1:
                        raise ValueError("Native layer did not execute exactly once")
                    h,mask = observed[0]
                    probability = F.softmax(F.linear(F.linear(h,layer.W_k.weight,layer.W_k.bias),layer.W_H.weight,layer.W_H.bias),dim=-1)
                    values = F.linear(h,layer.W_v.weight,layer.W_v.bias)
                    reference = probability.transpose(1,2) @ (values*mask.unsqueeze(-1))
                if not torch.equal(direct,f0) or not torch.equal(reference,z):
                    raise ValueError("Native cache does not equal independent same-shape prediction and bucket reconstruction")
                predictions.append(f0);buckets.append(z)
            finally:
                handle.remove()
        cache = execution.make_cache(payload["split"],payload["ids"],torch.cat(predictions),torch.cat(buckets),
            payload["truth"],payload["target_mean"],payload["target_sd"],parent_sha)
        return cache,dict(rows=len(cache.ids),chunks=(len(cache.ids)+255)//256,
            native_prediction_and_bucket_equal=True,source_state_inputs_gradients_hooks_and_rng_unchanged=True,
            target_mean=cache.target_mean,target_sd=cache.target_sd)
    finally:
        if not costs.same(before,module_snapshot(parent)) or not costs.same(saved,payload) or not torch.equal(rng,torch.get_rng_state()):
            raise ValueError("Native extraction changed its supplied model, inputs or CPU random state")


def save(cache,path):
    cache.verify()
    blob = dict(split=cache.split,ids=list(cache.ids),f0=cache.f0,z=cache.z,truth=cache.truth,
                target_mean=cache.target_mean,target_sd=cache.target_sd,
                parent_checkpoint_sha256=cache.parent_checkpoint_sha256,content_digest=cache.digest)
    life.atomic_tensor(path,blob,immutable=True)
    record = life.artifact(path)|dict(content_digest=cache.digest)
    if life.load_cache(record,cache.split,cache.parent_checkpoint_sha256).digest!=cache.digest:
        raise ValueError("Native cache serialization changed its content")
    return record
