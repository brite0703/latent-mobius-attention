"""Direct actual-row audit after numerical lookup guard failure; zero training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_recovery_test import pm,pc,sha,dump,deadline,verify_inputs,scored_metrics

@torch.no_grad()
def main():
    verify_inputs();deadline()
    diagnosis=json.loads((HERE/"evidence/scoring_guard_diagnosis.json").read_text())
    assert diagnosis["source_and_data_hash_checks_passed"]
    assert diagnosis["source_permutation_consistency_verified_in_float64"]
    lock=json.loads((HERE/"evidence/all_fits_locked.json").read_text())
    assert lock["started_training_runs"]==6 and all(x["status"]=="completed" for x in lock["results"])
    plan={"recorded_utc":pc.now(),"purpose":"Complete prespecified actual-row fp32 primary and secondary metrics after count-lookup approximation guard failed",
          "original_pipeline_status":"scoring_guard_failed_and_retained","original_guard_tolerance_unchanged":True,
          "training_or_checkpoint_changes":False,"new_fits":0,"new_optimizer_steps":0,
          "fit_lock_sha256":sha(HERE/"evidence/all_fits_locked.json"),
          "diagnosis_sha256":sha(HERE/"evidence/scoring_guard_diagnosis.json"),
          "primary_criterion_unchanged":"actual train accuracy>=.99 and CE<=.05 at epoch100",
          "dtype":"float32","evaluation":"actual original rows for train/val/test; canonical is discrepancy diagnostic only",
          "best_validation_checkpoint":"original canonical-validation-selected checkpoint remains unchanged; no reselection"}
    assert not (HERE/"evidence/direct_scoring_plan.json").exists()
    dump(HERE/"evidence/direct_scoring_plan.json",plan)
    rows=[]
    for item in lock["results"]:
        deadline()
        folder=HERE/"runs"/item["id"]
        assert sha(folder/"result.json")==item["record_sha256"]
        result=json.loads((folder/"result.json").read_text());spec=result["spec"]
        model=pm.build_model(spec["seed"],"lma3",20).cuda()
        destination=HERE/"direct_scoring"/spec["id"];destination.mkdir(parents=True,exist_ok=False)
        row={"spec":spec,"status":"fits_completed_direct_actual_rows_audited"}
        for checkpoint in ["final","best_validation"]:
            cp=folder/(checkpoint+".pt")
            assert sha(cp)==item[checkpoint+"_checkpoint_sha256"]
            model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True)["state_dict"])
            model.eval()
            assert next(model.parameters()).dtype==torch.float32
            canonical=pc.canonical_logits(model,20)
            scores={}
            for split in ["train","val","test"]:
                deadline()
                blob=pc.load_data(20,spec["seed"],split,"cuda")
                assert blob["x"].shape[1]==20 and bool(((blob["x"]==0)|(blob["x"]==1)).all())
                assert torch.equal(blob["x"].sum(1).long()%2,blob["y"])
                actual=np.concatenate([model(blob["x"][s:s+256]).double().cpu().numpy()
                                       for s in range(0,len(blob["y"]),256)])
                assert np.isfinite(actual).all()
                y=blob["y"].cpu().numpy();lookup=canonical[blob["count"]]
                scores[split]=scored_metrics(y,actual)
                scores[split].update(
                    actual_row_primary_not_lookup=True,
                    max_actual_canonical_logit_difference=float(np.abs(actual-lookup).max()),
                    original_lookup_guard_pass=bool(np.allclose(actual,lookup,atol=5e-4,rtol=5e-4)),
                    lookup_prediction_disagreements=int(np.sum(actual.argmax(1)!=lookup.argmax(1))),
                    count_lookup_diagnostic_metrics=scored_metrics(y,lookup))
                if split=="train":
                    scores[split].update(fit_success_accuracy_ge_099=scores[split]["accuracy"]>=.99,
                                         fit_success_ce_le_005=scores[split]["cross_entropy"]<=.05,
                                         fit_success_both=scores[split]["accuracy"]>=.99 and scores[split]["cross_entropy"]<=.05)
                np.savez_compressed(destination/(checkpoint+"_"+split+"_predictions.npz"),
                                    truth=y,count=blob["count"],actual_row_logits=actual,
                                    canonical_logits=canonical)
            scores["population_from_orbit_representatives_diagnostic"]=pc.population_metrics(20,canonical)
            scores["canonical_minimum_signed_margin"]=float(((2*(np.arange(21)%2)-1)*(canonical[:,1]-canonical[:,0])).min())
            row[checkpoint]=scores
        rows.append(row);del model;torch.cuda.empty_cache()
        print(json.dumps({"stage":"direct_actual_scoring","id":spec["id"],
                          "final_train":row["final"]["train"]}),flush=True)
    verify_inputs()
    elapsed=time.time()-json.loads((HERE/"evidence/budget.json").read_text())["start_unix"]
    assert elapsed<1800
    dump(HERE/"evidence/direct_actual_evaluation.json",
         {"evaluated_utc":pc.now(),"original_pipeline_guard_failed":True,
          "all_fits_lock_sha256":sha(HERE/"evidence/all_fits_locked.json"),
          "scoring_plan_sha256":sha(HERE/"evidence/direct_scoring_plan.json"),
          "rows":rows,"elapsed_budget_seconds":elapsed})
    dump(HERE/"evidence/direct_evaluation_completion.json",
         {"finished_utc":pc.now(),"started_training_runs":6,"completed_training_runs":6,
          "original_scoring_status":"guard_failed_retained",
          "direct_actual_row_audit":"completed","new_training_runs_added_by_audit":0,
          "protected_inputs_unchanged":True,"elapsed_wall_seconds":elapsed})
    print(json.dumps({"direct_actual_evaluation_complete":True,"elapsed_seconds":elapsed}),flush=True)

if __name__=="__main__":main()

