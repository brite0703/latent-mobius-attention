"""Decompose all retained finite-population predictions without fitting or selection."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import csv,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/"first_cubic"
HEADS=["deepsets_ln","deepsets_plain","deepsets_wide","janossy2","cp_pool","lma1","lma2","lma3","additive2","additive3"]
SUPPORTS=[(0,1,2),(0,1,3),(0,1,4),(0,5,6),(0,5,7),(8,9,10),(8,9,11),(2,6,10)]
SIGNS=[1,-1,1,1,-1,1,-1,1]
COMPONENTS=["target_amplitude_error","relative_target_coefficient_error","other_target_degree_energy","other_degree_energy"]
TOL=1e-11


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def raw_transform(values):
    result=np.asarray(values,dtype=np.float64).copy()
    width=1
    while width<len(result):
        blocks=result.reshape(-1,2*width)
        left,right=blocks[:,:width].copy(),blocks[:,width:].copy()
        blocks[:,:width]=left+right
        blocks[:,width:]=left-right
        width*=2
    return result


def stats(values):
    values=[float(x) for x in values]
    mean=math.fsum(values)/len(values)
    return dict(mean=mean,sample_sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)),
                negative_count=sum(x<0 for x in values),n=len(values))


def csv_write(path,rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    destination=HERE/"output/run_v1"
    if destination.exists():
        raise FileExistsError("Preserve the existing diagnosis rather than overwrite it")
    evaluation=read(STUDY/"evaluation.json")
    selection=read(STUDY/"selection_lock.json")
    assert read(STUDY/"final_audit.json")["passed"]
    assert evaluation["selection_lock_sha256"]==sha(STUDY/"selection_lock.json")
    assert selection["implementation_lock_sha256"]==sha(STUDY/"implementation_lock.json")
    assert len(evaluation["rows"])==len(selection["selections"])==200
    choices={r["selected_id"]:r for r in selection["selections"]}
    ids=np.arange(4096)
    bits=((ids[:,None]>>np.arange(12))&1).astype(np.int64)
    signs=2*bits-1
    degrees=np.array([int(i).bit_count() for i in ids])
    phase=np.where(degrees%2,-1.,1.)
    cubic_masks=[sum(1<<i for i in s) for s in SUPPORTS]
    probes=sorted({0,4095,*[1<<i for i in range(12)],*cubic_masks,
                   *map(int,np.random.default_rng(2026090901).choice(4096,16,replace=False))})
    probe_signs={mask:np.prod(signs[:,[i for i in range(12) if mask&(1<<i)]],axis=1) for mask in probes}
    target={}
    for task,degree,supports in [("first",1,[(i,) for i in range(8)]),("cubic",3,SUPPORTS)]:
        masks=[sum(1<<i for i in s) for s in supports]
        coefficients=np.zeros(4096)
        coefficients[masks]=np.array(SIGNS)/math.sqrt(8)
        truth=sum(s*np.prod(signs[:,support],axis=1) for s,support in zip(SIGNS,supports))/math.sqrt(8)
        assert np.max(np.abs(raw_transform(truth)*phase/4096-coefficients))<TOL
        assert abs(math.fsum(float(y*y) for y in truth)/4096-1)<TOL
        target[task]=(degree,masks,coefficients,truth)
    sources=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),HERE/"protocol.md",STUDY/"evaluation.json",
             STUDY/"selection_lock.json",STUDY/"implementation_lock.json",STUDY/"final_audit.json",STUDY/"generator.py"]]
    rows=[];coefficients=[];spectra=[];metadata=[]
    errors=dict(direct_coefficients=0.,inverse_transform=0.,parseval=0.,four_components=0.,covariance=0.,retained_mse=0.)
    seen=set()
    for entry in evaluation["rows"]:
        key=(entry["task"],entry["head"],entry["seed"])
        assert key not in seen
        seen.add(key)
        assert entry["status"]=="valid" and entry["checkpoint_sha256"]==choices[entry["selected_id"]]["checkpoint_sha256"]
        path=STUDY/entry["population_prediction_file"]
        assert sha(path)==entry["population_prediction_sha256"]
        sources.append(dict(path=str(path),sha256=sha(path)))
        degree,masks,target_coefficient,truth=target[entry["task"]]
        with np.load(path,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"],ids)
            np.testing.assert_array_equal(saved["truth"],truth)
            prediction=saved["prediction"].astype(np.float64)
        assert prediction.shape==(4096,) and np.isfinite(prediction).all()
        coefficient=raw_transform(prediction)*phase/4096
        reconstructed=raw_transform(coefficient*phase)
        inverse_error=float(np.max(np.abs(reconstructed-prediction)))
        direct_error=0.
        for mask in probes:
            expected=math.fsum(float(p)*int(s) for p,s in zip(prediction,probe_signs[mask]))/4096
            direct_error=max(direct_error,abs(expected-float(coefficient[mask])))
        alpha=float(coefficient@target_coefficient)
        direct_alpha=math.fsum(float(p)*float(y) for p,y in zip(prediction,truth))/4096
        delta=prediction-truth
        risk=math.fsum(float(x)*float(x) for x in delta)/4096
        residual_coefficients=coefficient-target_coefficient
        degree_errors=np.array([math.fsum(float(x)*float(x) for x in residual_coefficients[degrees==d]) for d in range(13)])
        active=np.zeros(4096,dtype=bool);active[masks]=True
        parts=dict(target_amplitude_error=(1-alpha)**2,
                   relative_target_coefficient_error=math.fsum(float(x)*float(x) for x in (coefficient[masks]-alpha*target_coefficient[masks])),
                   other_target_degree_energy=math.fsum(float(x)*float(x) for x in coefficient[(degrees==degree)&~active]),
                   other_degree_energy=math.fsum(float(x)*float(x) for x in coefficient[degrees!=degree]))
        case_errors=dict(direct_coefficients=direct_error,inverse_transform=inverse_error,
            parseval=abs(float(degree_errors.sum())-risk),four_components=abs(math.fsum(parts.values())-risk),
            covariance=abs(alpha-direct_alpha),retained_mse=abs(risk-entry["population"]["mse"]))
        assert all(v<=TOL for v in case_errors.values()),(key,case_errors)
        for k,v in case_errors.items():
            errors[k]=max(errors[k],v)
        row=dict(task=entry["task"],head=entry["head"],seed=entry["seed"],selected_id=entry["selected_id"],
                 population_mse=risk,retained_test_mse=entry["test"]["mse"],target_alignment=alpha,constant_bias_energy=float(coefficient[0]**2),**parts)
        row.update({f"degree_{d}_error":float(degree_errors[d]) for d in range(13)})
        rows.append(row);coefficients.append(coefficient);spectra.append(degree_errors)
        metadata.append(dict(task=entry["task"],head=entry["head"],seed=entry["seed"],selected_id=entry["selected_id"]))
    assert seen=={(task,head,seed) for task in ("first","cubic") for head in HEADS for seed in range(100,110)}
    measures=["population_mse","retained_test_mse","target_alignment","constant_bias_energy",*COMPONENTS,*[f"degree_{d}_error" for d in range(13)]]
    summaries=[]
    for task in ("first","cubic"):
        for head in HEADS:
            selected=[r for r in rows if r["task"]==task and r["head"]==head]
            summaries.append(dict(task=task,head=head,metrics={key:stats([r[key] for r in selected]) for key in measures}))
    index={(r["task"],r["head"],r["seed"]):r for r in rows}
    contrasts=[];paired=[]
    for right in ("lma1","additive2","cp_pool"):
        by_task={}
        for task in ("first","cubic"):
            values=[]
            for seed in range(100,110):
                left_row,right_row=index[task,"lma2",seed],index[task,right,seed]
                values.append(dict(task=task,left="lma2",right=right,seed=seed,
                                   **{key:left_row[key]-right_row[key] for key in measures}))
            paired+=values;by_task[task]=values
            contrasts.append(dict(task=task,left="lma2",right=right,metrics={key:stats([r[key] for r in values]) for key in measures}))
        differences=[dict(task="cubic_minus_first",left="lma2",right=right,seed=a["seed"],
                          **{key:b[key]-a[key] for key in measures}) for a,b in zip(by_task["first"],by_task["cubic"])]
        paired+=differences
        contrasts.append(dict(task="cubic_minus_first",left="lma2",right=right,metrics={key:stats([r[key] for r in differences]) for key in measures}))
    destination.mkdir(parents=True)
    csv_write(destination/"all_seed_components.csv",rows)
    csv_write(destination/"paired_components.csv",paired)
    np.savez_compressed(destination/"spectra.npz",coefficients=np.stack(coefficients),degree_error=np.stack(spectra),degrees=degrees,
                        task=np.array([r["task"] for r in metadata]),head=np.array([r["head"] for r in metadata]),seed=np.array([r["seed"] for r in metadata]))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,selected_models=200,states_per_model=4096,
                coefficient_probe_indices=probes,direct_coefficient_checks=200*len(probes),absolute_tolerance=TOL,
                maximum_discrepancies=errors,procedures=summaries,contrasts=contrasts,sources=sources,
                qualification="Post-outcome standard Fourier diagnosis on full finite populations including fitting/validation; no new fitting, selection, independent replication or theorem-level novelty")
    (destination/"diagnosis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    render(summaries,destination)
    findings(report,destination)
    receipt=dict(passed=True,created_utc=report["created_utc"],selected_models=200,
                 files=[dict(path=str(p),sha256=sha(p)) for p in sorted(destination.iterdir()) if p.is_file()])
    (HERE/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(passed=True,selected_models=200,direct_coefficient_checks=report["direct_coefficient_checks"],
        maximum_discrepancies=errors,contrasts=[dict(task=c["task"],right=c["right"],risk=c["metrics"]["population_mse"],
            components={k:c["metrics"][k]["mean"] for k in COMPONENTS}) for c in contrasts])),flush=True)


def render(summaries,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=["Sum + LN","Plain sum","Wide plain sum","Janossy-2","CP","LMA 1","LMA 2","LMA 3","Additive 2","Additive 3"]
    colors=["#264568","#548b8e","#cb9a47","#afb9c5"]
    labels=["Target amplitude error","Relative target-coefficient error","Other modes of target degree","Other degrees (including bias)"]
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    for ax,task,title in zip(axes,["first","cubic"],["First-degree target","Overlapping-cubic target"]):
        selected={r["head"]:r for r in summaries if r["task"]==task}
        left=np.zeros(10)
        for key,color,label in zip(COMPONENTS,colors,labels):
            values=np.array([selected[h]["metrics"][key]["mean"] for h in HEADS])
            ax.barh(np.arange(10),values,left=left,color=color,label=label,height=.68)
            left+=values
        ax.set_yticks(np.arange(10),names)
        ax.invert_yaxis();ax.set_xlabel("Full-population MSE (separate axis scales)");ax.set_title(title)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",alpha=.16);ax.set_axisbelow(True)
    fig.suptitle("Error components of the saved synthetic predictors",y=.99,fontsize=15)
    fig.text(.5,.928,"4,096 Boolean states; ten seeds per bar; fitting and validation states included",ha="center",fontsize=10)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=[0,.1,1,.92],w_pad=2.4)
    fig.savefig(destination/"target_error_components.png",dpi=180,bbox_inches="tight")
    fig.savefig(destination/"target_error_components.svg",bbox_inches="tight")
    plt.close(fig)


def findings(report,destination):
    text=["# Target-structure diagnosis of the retained predictors","",
          "All 200 selected neural predictors and their unchanged full-cube prediction vectors pass the decomposition checks. This is a post-outcome analysis of completed results. It adds no trained model, seed or independent dataset.","",
          "For a unit-variance target, population MSE separates into target-amplitude error, relative error among its eight nonzero coefficients, other modes of the same degree, and other degrees. This is the standard orthogonal-projection identity. Full-population error includes fitting and validation states.","",
          "| Target | Order two minus comparator | Population MSE difference | Paired SD | Lower-error seeds | Amplitude term | Relative target term | Other same-degree modes | Other degrees |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["contrasts"]:
        m=c["metrics"];s=m["population_mse"]
        text.append(f"| {c['task']} | {c['right']} | {s['mean']:+.6f} | {s['sample_sd']:.6f} | {s['negative_count']}/10 | "+" | ".join(f"{m[k]['mean']:+.6f}" for k in COMPONENTS)+" |")
    text += ["","The table retains all specified contrasts, including changes between the two targets. The full CSV and coefficient arrays retain every procedure and seed. These descriptive paired summaries do not provide a newly confirmatory significance test.","",
             "The two target families differ in degree, support overlap and active coordinates. This comparison therefore cannot isolate degree alone. The native first-order model is nonlinear: its cubic coefficients are measured, not assumed absent. A reduction in target-amplitude or coefficient error describes the fitted functions; it does not uniquely attribute that reduction to multiplication, training dynamics or a restricted scalar theorem. A real-data explanation requires its own evidence.","",
             "Numerical verification uses direct signed sums at fixed coordinates, inverse reconstruction of every output, direct scalar covariance/MSE and the original recorded population MSE. Maximum discrepancies: `"+json.dumps(report["maximum_discrepancies"])+"`. All original prediction and source hashes are preserved."]
    (destination/"findings.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
