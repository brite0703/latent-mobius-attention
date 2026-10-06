"""Actual fitting-input contracts and bounded native cache integration checks."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import campaign_lifecycle as life
import costs
import native_cache
import native_inputs

HERE=Path(__file__).resolve().parent


def main():
    destination=HERE/"native_inputs_cpu_audit.json"
    scratch=HERE/"discarded_native_input_audit_20260909"
    if destination.exists() or scratch.exists():
        raise FileExistsError("Preserve earlier native input audit outputs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    paths=[HERE/name for name in ("native_input_specification.md","native_inputs.py","native_cache.py","audit_native_inputs.py",
        "native_costs.py","native_costs_cpu_audit.json","models.py","execution.py","selection.py","costs.py",
        "campaign_control.py","campaign_lifecycle.py","parent_fragments/cubic_v2/fragment.json")]
    sources={str(p.resolve()):life.artifact(p) for p in paths}
    blocked=[]
    original_numpy_load,original_torch_load=np.load,torch.load
    def forbidden_data_read(*args,**kwargs):
        raise AssertionError("A scientific archive reader ran before test authorization")
    np.load,torch.load=forbidden_data_read,forbidden_data_read
    try:
        for domain,seed in (("cubic",100),("ligand_contact",42)):
            try:
                native_inputs.read_inputs(domain,seed,"test")
            except ValueError as error:
                blocked.append(dict(domain=domain,reason=str(error),archive_reader_called=False))
            else:
                raise AssertionError("Unselected scientific test inputs were accepted")
    finally:
        np.load,torch.load=original_numpy_load,original_torch_load
    fragment=life.read(HERE/"parent_fragments/cubic_v2/fragment.json")
    row=next(r for r in fragment["parents"] if r["seed"]==100)
    actual_cubic=[]
    for split in ("train","val"):
        rng=torch.get_rng_state().clone()
        parent,layer_path,payload=native_inputs.parent_and_inputs(row,split)
        assert torch.equal(rng,torch.get_rng_state())
        old=life.load_cache(row["caches"][split],split,row["parent_checkpoint"]["sha256"])
        assert tuple(payload["ids"])==old.ids and torch.equal(payload["truth"],old.truth)
        assert (payload["target_mean"],payload["target_sd"])==(0.,1.)
        for source in payload["sources"]+[row["caches"][split]]:
            sources[str(Path(source["path"]).resolve())]=source
        limited=deepcopy(payload)
        limited["ids"]=limited["ids"][:256]
        limited["truth"]=limited["truth"][:256]
        limited["inputs"]={k:v[:256] for k,v in limited["inputs"].items()}
        new,check=native_cache.extract(parent,layer_path,limited,row["parent_checkpoint"]["sha256"])
        assert torch.equal(new.f0,old.f0[:256]) and torch.equal(new.z,old.z[:256])
        assert torch.equal(new.truth,old.truth[:256])
        actual_cubic.append(dict(split=split,input_contract_rows=len(payload["ids"]),
            native_rows_checked=256,native_cache_exact=True,original_target_and_id_contract_exact=True))
    molecular=[]
    for split in ("train","val"):
        payload=native_inputs.read_inputs("ligand_contact",42,split)
        assert payload["inputs"]["x"].shape==(len(payload["ids"]),57,53)
        assert payload["inputs"]["contact"].shape==(len(payload["ids"]),57,216)
        for source in payload["sources"]:
            sources[str(Path(source["path"]).resolve())]=source
        molecular.append(dict(split=split,input_contract_rows=len(payload["ids"]),
            target_mean=payload["target_mean"],target_sd=payload["target_sd"],
            actual_matched_parent_loaded=False))
    del payload,parent
    fixture_checks=[]
    generator=torch.Generator().manual_seed(2026090847)
    for domain,seed in (("cubic",100),("ligand_contact",42)):
        module,_=native_inputs.modules(domain)
        with torch.random.fork_rng(devices=[]):
            parent=module.build(seed,"lma1") if domain=="cubic" else module.build_model(seed,"ligand_contact","lma1")
        molecular_domain=domain=="ligand_contact"
        nodes,features=(57,53) if molecular_domain else (12,12)
        inputs=dict(x=torch.randn(259,nodes,features,generator=generator),mask=torch.ones(259,nodes,dtype=torch.bool))
        inputs["mask"][::2,-3:]=False
        if molecular_domain:
            inputs.update(adj=torch.eye(nodes).expand(259,nodes,nodes).clone(),contact=torch.rand(259,nodes,216,generator=generator))
        payload=dict(split="train",ids=[f"manufactured:{domain}:{i}" for i in range(259)],inputs=inputs,
            truth=torch.linspace(-1,1,259,dtype=torch.float64),target_mean=0. if not molecular_domain else 6.3,
            target_sd=1. if not molecular_domain else 1.4)
        parent_sha=hashlib.sha256(("manufactured native cache "+domain).encode()).hexdigest()
        layer_path="pool.layers.0" if molecular_domain else "head.layers.0"
        cache,check=native_cache.extract(parent,layer_path,payload,parent_sha)
        assert check["rows"]==259 and check["chunks"]==2
        saved=native_cache.save(cache,scratch/(domain+".pt"))
        before=life.sha(saved["path"])
        assert native_cache.save(cache,saved["path"])==saved and life.sha(saved["path"])==before
        assert life.load_cache(saved,"train",parent_sha).digest==cache.digest
        fixture_checks.append(dict(domain=domain,rows=259,remainder_rows=3,exact_native_cache=True,
                                   serialization_and_idempotent_preservation=True,artifact=saved))
    synthetic,_=native_inputs.modules("cubic")
    for path in synthetic.sources()+[native_inputs.CUBIC/"generator.py",native_inputs.POCKET/"matched_models.py",
        native_inputs.POCKET/"matched_execution.py",native_inputs.POCKET.parent/"sequence_models.py",
        native_inputs.POCKET.parent/"sequence_execution.py",native_inputs.POCKET.parent/"sequence_data.py",
        native_inputs.POCKET.parent/"sequence_encoder.py"]:
        sources[str(Path(path).resolve())]=life.artifact(path)
    for source in sources.values():
        life.verify_artifact(source)
    receipt=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope="Actual fitting/validation input contracts, one saved cubic parent and manufactured native extraction fixtures",
        actual_cubic_input_rows=3072,actual_cubic_native_prediction_rows=512,
        actual_molecular_input_rows=1246,actual_matched_parents_loaded=0,
        actual_cubic_checks=actual_cubic,actual_molecular_input_checks=molecular,
        manufactured_architectures=2,manufactured_native_rows=518,fixture_checks=fixture_checks,
        test_access_rejections=blocked,scientific_test_archive_readers_called=False,
        residual_optimizer_updates=0,new_scientific_test_predictions=False,cuda_initialized=torch.cuda.is_initialized(),
        sources=list(sources.values()),artifacts=[r["artifact"] for r in fixture_checks],
        remaining="Actual matched parent/validation equivalence and full bundle, retained-gated test producer, source-bound cost orchestration, any activation and scientific outcomes")
    life.atomic_json(destination,receipt,immutable=True)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("sources","artifacts","fixture_checks","actual_cubic_checks","actual_molecular_input_checks","test_access_rejections")}),flush=True)


if __name__=="__main__":
    main()
