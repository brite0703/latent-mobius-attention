"""Independent CPU checks of all retained arrays, pairing and committed updates."""
from pathlib import Path
import json, hashlib, math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonlines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]

def main():
    lock=read(HERE/"evidence/all_fits_locked.json")
    evaluation=read(HERE/"evidence/direct_actual_evaluation.json")
    completion=read(HERE/"evidence/direct_evaluation_completion.json")
    assert completion["started_training_runs"]==completion["completed_training_runs"]==6
    assert completion["direct_actual_row_audit"]=="completed" and completion["elapsed_wall_seconds"]<1800
    assert evaluation["all_fits_lock_sha256"]==sha(HERE/"evidence/all_fits_locked.json")
    chain=read(HERE/"evidence/training_chain_checks.json")
    assert chain["status"]=="PASS" and len(chain["finite_difference_checks"])==30
    assert all(x["passed"] for x in chain["finite_difference_checks"])
    checks=[];maximum_metric_error=0.;maximum_fp32_ce_error=0.
    total_batches=0;total_snapshots=0
    for entry,row in zip(lock["results"],evaluation["rows"]):
        spec=row["spec"];folder=HERE/"runs"/spec["id"]
        assert entry["id"]==spec["id"] and entry["status"]=="completed"
        assert sha(folder/"result.json")==entry["record_sha256"]
        result=read(folder/"result.json")
        assert result["epochs"]==100 and result["optimizer_steps"]==1900
        assert read(folder/"optimizer_update_step1.json")["status"]=="PASS"
        batches=jsonlines(folder/"minibatches.jsonl");snapshots=jsonlines(folder/"snapshots.jsonl")
        assert len(batches)==1900 and len(snapshots)==101
        assert all(x["state_and_rng_preserved"] for x in snapshots)
        total_batches+=len(batches);total_snapshots+=len(snapshots)
        for checkpoint in ["final","best_validation"]:
            assert sha(folder/(checkpoint+".pt"))==entry[checkpoint+"_checkpoint_sha256"]
            state=torch.load(folder/(checkpoint+".pt"),map_location="cpu",weights_only=True)["state_dict"]
            assert sum(t.numel() for t in state.values())==12525
            assert all(bool(torch.isfinite(t).all()) for t in state.values())
            for split in ["train","val","test"]:
                with np.load(HERE/"direct_scoring"/spec["id"]/(checkpoint+"_"+split+"_predictions.npz")) as z:
                    y=z["truth"];logits=z["actual_row_logits"];count=z["count"]
                    assert np.array_equal(y,count%2) and logits.shape==(len(y),2)
                    assert np.isfinite(logits).all()
                    signed=(2*y-1)*(logits[:,1]-logits[:,0])
                    metrics={"accuracy":float(np.mean(logits.argmax(1)==y)),
                             "cross_entropy":float(np.logaddexp(0.,-signed).mean()),
                             "minimum_signed_margin":float(signed.min()),
                             "mean_signed_margin":float(signed.mean()),
                             "median_signed_margin":float(np.median(signed)),
                             "percentile05_signed_margin":float(np.quantile(signed,.05)),
                             "negative_margin_fraction":float(np.mean(signed<0))}
                    errors={k:abs(v-row[checkpoint][split][k]) for k,v in metrics.items()}
                    assert max(errors.values())<1e-12
                    maximum_metric_error=max(maximum_metric_error,max(errors.values()))
                    torch_ce=float(F.cross_entropy(torch.from_numpy(logits.astype(np.float32)),torch.from_numpy(y)))
                    ce_error=abs(torch_ce-metrics["cross_entropy"])
                    assert ce_error<2e-6
                    maximum_fp32_ce_error=max(maximum_fp32_ce_error,ce_error)
                    checks.append({"id":spec["id"],"checkpoint":checkpoint,"split":split,
                                   "rows":len(y),"maximum_saved_metric_error":max(errors.values()),
                                   "cpu_fp32_torch_ce_vs_manual_error":ce_error})
    batch_pairs=[]
    for seed in [100,101,102]:
        a=HERE/"runs"/f"n20_original_seed{seed}"
        b=HERE/"runs"/f"n20_perturbed_witness_seed{seed}"
        ha=[x["batch_order_sha256"] for x in jsonlines(a/"epochs.jsonl")]
        hb=[x["batch_order_sha256"] for x in jsonlines(b/"epochs.jsonl")]
        assert ha==hb and len(ha)==100
        batch_pairs.append({"seed":seed,"all100_epoch_batch_orders_identical":True})
    guard=read(HERE/"evidence/protocol_lock.json")
    for item in guard["files"]:assert sha(item["path"])==item["sha256"],item["path"]
    output={"status":"PASS","checked_utc":datetime.now(timezone.utc).isoformat(),
            "new_model_forwards":0,"new_optimizer_steps":0,"saved_metric_checks":checks,
            "maximum_metric_error":maximum_metric_error,
            "maximum_cpu_fp32_ce_vs_manual_error":maximum_fp32_ce_error,
            "minibatches":total_batches,"snapshots":total_snapshots,"batch_pairs":batch_pairs,
            "protected_file_hashes_unchanged":len(guard["files"])}
    p=HERE/"evidence/independent_saved_results_check.json";assert not p.exists()
    p.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k!="saved_metric_checks"},indent=2))

if __name__=="__main__":main()

