"""Bounded post-failure numerical diagnosis; original scoring guard stays failed."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import time,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_recovery_test import pm,pc,sha,dump,deadline,verify_inputs,scored_metrics

def main():
    verify_inputs();deadline()
    lock=json.loads((HERE/"evidence/all_fits_locked.json").read_text())
    entry=next(x for x in lock["results"] if x["id"]=="n20_perturbed_witness_seed100")
    cp=HERE/"runs"/entry["id"]/"final.pt"
    assert sha(cp)==entry["final_checkpoint_sha256"]
    state=torch.load(cp,map_location="cpu",weights_only=True)["state_dict"]
    blob=pc.load_data(20,100,"train","cuda")
    assert blob["x"].shape==(4800,20) and bool(((blob["x"]==0)|(blob["x"]==1)).all())
    assert torch.equal(blob["x"].sum(1).long()%2,blob["y"])
    assert np.array_equal(blob["count"],blob["x"].sum(1).long().cpu().numpy())
    original_guard={"atol":5e-4,"rtol":5e-4}
    record={"reason":"Original frozen scoring stopped at actual/canonical assertion",
            "source_and_data_hash_checks_passed":True,"binary_length20_labels_count_mod2":True,
            "checkpoint_sha256":sha(cp),"checkpoint":str(cp),"new_optimizer_steps":0,
            "new_fits":0,"original_guard_not_relaxed":original_guard,"dtypes":{},
            "elapsed_budget_seconds_before":time.time()-json.loads((HERE/"evidence/budget.json").read_text())["start_unix"]}
    for dtype_name,dtype in [("float32",torch.float32),("float64",torch.float64)]:
        deadline()
        model=pm.build_model(100,"lma3",20).cuda().to(dtype)
        model.load_state_dict({n:t.to(dtype) for n,t in state.items()})
        model.eval()
        with torch.inference_mode():
            canonical=pc.canonical_logits(model,20)
            parts=[]
            for start in range(0,4800,256):
                deadline()
                parts.append(model(blob["x"][start:start+256].to(dtype)).double().cpu().numpy())
            actual=np.concatenate(parts);lookup=canonical[blob["count"]]
        assert np.isfinite(actual).all() and np.isfinite(lookup).all()
        differences=np.abs(actual-lookup)
        tolerance=original_guard["atol"]+original_guard["rtol"]*np.abs(lookup)
        worst=np.unravel_index(np.argmax(differences-tolerance),differences.shape)
        entry_record={"max_actual_canonical_absolute_logit_difference":float(differences.max()),
                      "original_guard_pass":bool(np.allclose(actual,lookup,**original_guard)),
                      "entries_exceeding_original_guard":int((differences>tolerance).sum()),
                      "prediction_label_disagreements":int((actual.argmax(1)!=lookup.argmax(1)).sum()),
                      "actual_training_metrics":scored_metrics(blob["y"].cpu().numpy(),actual),
                      "canonical_lookup_training_metrics":scored_metrics(blob["y"].cpu().numpy(),lookup),
                      "largest_guard_excess_row":int(worst[0]),"largest_guard_excess_class":int(worst[1]),
                      "worst_actual_logit":float(actual[worst]),"worst_lookup_logit":float(lookup[worst]),
                      "worst_allowed_error":float(tolerance[worst]),
                      "float64_1e_8_consistency":bool(np.allclose(actual,lookup,atol=1e-8,rtol=1e-8))}
        record["dtypes"][dtype_name]=entry_record
        np.savez_compressed(HERE/"output"/("scoring_guard_diagnosis_"+dtype_name+".npz"),
                            truth=blob["y"].cpu().numpy(),count=blob["count"],
                            actual_row_logits=actual,canonical_logits=canonical)
        del model
        torch.cuda.empty_cache()
    record["source_permutation_consistency_verified_in_float64"]=record["dtypes"]["float64"]["float64_1e_8_consistency"]
    record["interpretation"]="Finite-precision count-lookup approximation failed in float32" if (
        not record["dtypes"]["float32"]["original_guard_pass"] and
        record["source_permutation_consistency_verified_in_float64"]) else "Unresolved or borderline scoring discrepancy; do not relax guard"
    record["elapsed_budget_seconds_after"]=time.time()-json.loads((HERE/"evidence/budget.json").read_text())["start_unix"]
    dump(HERE/"evidence/scoring_guard_diagnosis.json",record)
    print(json.dumps(record,indent=2))
    assert record["source_permutation_consistency_verified_in_float64"],"Unresolved source/numerical discrepancy"

if __name__=="__main__":main()

