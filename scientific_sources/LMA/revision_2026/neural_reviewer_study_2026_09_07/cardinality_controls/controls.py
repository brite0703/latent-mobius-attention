"""Pre-test baseline supplement; reuses the immutable core training function."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PRIMARY = HERE.parent
sys.path.insert(0, str(PRIMARY))
import models
import study as base

ORIGINAL_BUILDER = models.build_model
HEADS = ["deepsets_raw35", "deepsets_raw70", "deepsets_count35", "deepsets_count70", "pma_count"]


class CountDeepSets(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.phi, self.norm = head.phi, head.rho[0]
        old = head.rho[1]
        self.linear = nn.Linear(old.in_features+1, old.out_features)
        with torch.no_grad():
            self.linear.weight[:, :-1].copy_(old.weight)
            self.linear.weight[:, -1].zero_()
            self.linear.bias.copy_(old.bias)
        self.last = head.rho[3]

    def forward(self, h, mask):
        z = (self.phi(h)*mask.unsqueeze(-1)).sum(1)
        count = torch.log1p(mask.sum(1, keepdim=True).to(h.dtype))
        return self.last(F.gelu(self.linear(torch.cat([self.norm(z), count], -1)))).squeeze(-1)


class CountPMA(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.query, self.attention = head.query, head.attention
        self.norm, self.ffn = head.norm, head.ffn
        self.head = nn.Linear(head.head.in_features+1, 1)
        with torch.no_grad():
            self.head.weight[:, :-1].copy_(head.head.weight)
            self.head.weight[:, -1].zero_()
            self.head.bias.copy_(head.head.bias)

    def forward(self, h, mask):
        q = self.query.expand(len(h), -1, -1)
        value, _ = self.attention(q, h, h, key_padding_mask=~mask, need_weights=False)
        z = self.norm(q+value)
        z = (z+self.ffn(z)).squeeze(1)
        count = torch.log1p(mask.sum(1, keepdim=True).to(h.dtype))
        return self.head(torch.cat([z, count], -1)).squeeze(-1)


def build(seed, name, depth, d_in=53):
    assert depth == 1 and name in HEADS
    original_name = ("pma" if name=="pma_count" else "deepsets_wide" if name.endswith("70") else "deepsets")
    model = ORIGINAL_BUILDER(seed, original_name, depth, d_in)
    if name == "pma_count":
        model.head = CountPMA(model.head)
    elif "count" in name:
        model.head = CountDeepSets(model.head)
    else:
        # Keep inherited phi/rho linear parameters; remove only pooled LayerNorm.
        model.head.rho = nn.Sequential(*list(model.head.rho.children())[1:])
    return model


def specifications():
    rows = [dict(id=f"gcn1_{head}_seed{seed}_lr{j}", head=head, depth=1, seed=seed, lr=lr, lr_index=j)
            for head in HEADS for seed in base.SEEDS for j, lr in enumerate(base.LRS)]
    return [rows[i] for i in np.random.default_rng(2026090722).permutation(50)]


def audit():
    from prefit_audit import graph_batch
    x, mask, adj = graph_batch()
    checks = []
    for name in HEADS:
        for seed in base.SEEDS:
            model = build(seed, name, 1).double().eval()
            old_name = "pma" if name=="pma_count" else "deepsets_wide" if name.endswith("70") else "deepsets"
            old = ORIGINAL_BUILDER(seed, old_name, 1).double().eval()
            assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in old.encoder.state_dict().items())
            p = model(x, mask, adj)
            perm = torch.tensor([3, 0, 5, 1, 4, 2])
            q = model(x[:,perm], mask[:,perm], adj[:,perm][:,:,perm])
            padded = F.pad(x, (0,0,0,3))
            padded[:, -3:] = 13.
            padded.requires_grad_()
            pad_mask = F.pad(mask, (0,3))
            z = model(padded, pad_mask, F.pad(adj, (0,3,0,3)))
            singles = torch.cat([model(x[i:i+1],mask[i:i+1],adj[i:i+1]) for i in range(3)])
            error = max(float((p-q).abs().max().detach()), float((p-z).abs().max().detach()),
                        float((p-singles).abs().max().detach()))
            assert error < 1e-11
            z.sum().backward()
            assert float(padded.grad[~pad_mask].abs().max()) == 0.
            if "count" in name:
                initial_delta = float((p-old(x,mask,adj)).abs().max().detach())
                assert initial_delta < 1e-12, (name,seed,initial_delta)
                if name == "pma_count":
                    assert torch.equal(model.head.head.weight[:, :-1], old.head.head.weight)
                    assert torch.equal(model.head.head.bias, old.head.head.bias)
                    assert float(model.head.head.weight[:, -1].abs().max()) == 0.
                else:
                    assert torch.equal(model.head.linear.weight[:, :-1], old.head.rho[1].weight)
                    assert torch.equal(model.head.linear.bias, old.head.rho[1].bias)
                    assert float(model.head.linear.weight[:, -1].abs().max()) == 0.
                    assert all(torch.equal(v,model.head.phi.state_dict()[k]) for k,v in old.head.phi.state_dict().items())
                with torch.no_grad():
                    if name == "pma_count":
                        model.head.head.weight[:, -1] = .25
                    else:
                        model.head.linear.weight[:, -1] = torch.linspace(.1,.3,model.head.linear.out_features,dtype=torch.float64)
                after = model(x,mask,adj)
                probe_change = float((after-p).abs().max().detach())
                assert probe_change > 1e-6
                added = 1 if name=="pma_count" else model.head.linear.out_features
                assert models.parameter_count(model)-models.parameter_count(old) == added
                with torch.no_grad():
                    if name == "pma_count":
                        model.head.head.weight[:, -1].zero_()
                    else:
                        model.head.linear.weight[:, -1].zero_()
            else:
                h = old.encoder(x,mask,adj)
                pooled = (old.head.phi(h)*mask.unsqueeze(-1)).sum(1)
                explicit = old.head.rho[3](F.gelu(old.head.rho[1](pooled))).squeeze(-1)
                initial_delta = float((p-explicit).abs().max().detach())
                assert initial_delta < 1e-12
                probe_change = None
                assert models.parameter_count(model)-models.parameter_count(old) == -2*old.head.rho[0].normalized_shape[0]
            # This model is discarded. Verify a finite update reaches its newly
            # added count coefficient (where present) without benchmarking labels.
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
            optimizer.zero_grad()
            loss = (model(x,mask,adj)-torch.tensor([-.7,.2,1.1],dtype=torch.float64)).square().mean()
            loss.backward()
            count_gradient = None
            if "count" in name:
                gradient = model.head.head.weight.grad[:, -1] if name=="pma_count" else model.head.linear.weight.grad[:, -1]
                count_gradient = float(gradient.norm())
                assert count_gradient > 1e-10
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            checks.append(dict(head=name, seed=seed, invariance_max_delta=error,
                               initial_output_or_equation_delta=initial_delta, discarded_count_probe_change=probe_change,
                               initial_zero_column_gradient_norm=count_gradient,
                               total_parameters=models.parameter_count(model)))
    # Algebraic check of the finite-epsilon caveat; c multiplies the centered
    # sum and its SD, leaving epsilon/c^2 rather than exactly the same map.
    torch.manual_seed(881)
    z = torch.randn(3, 35, dtype=torch.float64)
    c = 2.
    eps = 1e-5
    reference = (z-z.mean(-1,keepdim=True))/torch.sqrt(z.var(-1,keepdim=True,unbiased=False)+eps/c**2)
    assert torch.allclose(F.layer_norm(c*z,(35,),eps=eps),reference,atol=1e-15,rtol=1e-15)
    output = dict(passed=True, audited_utc=base.now(), checks=checks, test_tensors_loaded=False,
                  layernorm_finite_epsilon_equation_verified=True, source_sha256=base.sha(__file__))
    base.write_json(HERE/"prefit_audit.json", output)
    print(json.dumps(dict(stage="supplement_audit", passed=True, checks=len(checks))), flush=True)


def lock():
    base.verify_lock()
    assert not (PRIMARY/"evaluation.json").exists()
    audit_record = json.loads((HERE/"prefit_audit.json").read_text(encoding="utf-8"))
    assert audit_record["passed"] and audit_record["source_sha256"] == base.sha(__file__)
    assert not (HERE/"implementation_lock.json").exists()
    files = [Path(__file__), HERE/"protocol.md", HERE/"prefit_audit.json", HERE/"web_review_22_disposition.md",
             PRIMARY/"study.py", PRIMARY/"models.py", PRIMARY/"implementation_lock.json"]
    base.write_json(HERE/"implementation_lock.json", dict(locked_utc=base.now(), candidates=specifications(),
        candidate_count=50, selection_count=25, primary_implementation_lock_sha256=base.sha(PRIMARY/"implementation_lock.json"),
        reason="Pooled LayerNorm scale suppression found by source inspection before any new test predictions.",
        development_status="Core validation-only outcomes already existed; no claim of development independence.",
        training="The unmodified core study.train_candidate function with a declared factory and output-directory override in this separate process.",
        files=[dict(path=str(p), sha256=base.sha(p)) for p in files]))
    print(json.dumps(dict(stage="supplement_locked", candidates=50, selected=25)), flush=True)


def verify():
    primary_context = base.HERE
    base.HERE = PRIMARY
    base.verify_lock()
    base.HERE = primary_context
    record = json.loads((HERE/"implementation_lock.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert base.sha(item["path"]) == item["sha256"], item["path"]
    assert record["candidates"] == specifications()
    return record


def train():
    record = verify()
    assert not (PRIMARY/"evaluation.json").exists()
    base.HERE = HERE
    models.build_model = build
    training, validation = base.load_split("train"), base.load_split("val")
    mean = float(training["y"].double().mean())
    sd = float(training["y"].double().std(unbiased=False))
    for item in record["candidates"]:
        base.train_candidate(item,training,validation,mean,sd,base.sha(HERE/"implementation_lock.json"))
        done = [json.loads(p.read_text(encoding="utf-8")) for p in (HERE/"candidates").glob("*.json")]
        base.write_json(HERE/"progress.json", dict(updated_utc=base.now(), completed=len(done), total=50,
             valid=sum(r["status"]=="valid" for r in done), failed=sum(r["status"]=="failed" for r in done),
             aggregate_fit_seconds=sum(r["wall_seconds"] for r in done)))
    selections, files = [], []
    for head in HEADS:
        for seed in base.SEEDS:
            rows = [r for r in done if r["spec"]["head"]==head and r["spec"]["seed"]==seed]
            assert len(rows)==2
            valid = [r for r in rows if r["status"]=="valid"]
            selected = min(valid,key=lambda r:(r["best_validation_rmse"],r["spec"]["lr"])) if valid else None
            selections.append(dict(depth=1,head=head,seed=seed,selected_id=selected["spec"]["id"] if selected else None,
                validation_rmse=selected["best_validation_rmse"] if selected else None,
                candidate_ids=[r["spec"]["id"] for r in rows],
                checkpoint_sha256=selected["checkpoint_sha256"] if selected else None))
            files += [dict(path=f'candidates/{r["spec"]["id"]}.json',sha256=base.sha(HERE/"candidates"/(r["spec"]["id"]+".json"))) for r in rows]
    assert not (HERE/"selection_lock.json").exists()
    base.write_json(HERE/"selection_lock.json",dict(locked_utc=base.now(),selections=selections,candidate_records=files,
         implementation_lock_sha256=base.sha(HERE/"implementation_lock.json")))
    print(json.dumps(dict(stage="supplement_selections_locked",selected=25)),flush=True)


def gate():
    verify()
    assert not (PRIMARY/"evaluation.json").exists() and not (HERE/"evaluation.json").exists()
    assert not (PRIMARY/"combined_selection_gate.json").exists()
    core = json.loads((PRIMARY/"selection_lock.json").read_text(encoding="utf-8"))
    supp = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    assert len(core["selections"])==85 and len(supp["selections"])==25
    for directory, record in [(PRIMARY,core),(HERE,supp)]:
        for item in record["candidate_records"]:
            assert base.sha(directory/item["path"]) == item["sha256"]
    base.write_json(PRIMARY/"combined_selection_gate.json",dict(locked_utc=base.now(),total_candidates=220,total_selections=110,
       primary_selection_sha256=base.sha(PRIMARY/"selection_lock.json"),
       supplement_selection_sha256=base.sha(HERE/"selection_lock.json"),
       qualification="Both new blocks fixed before their test scoring; the historical test-reuse limitation remains."))
    print(json.dumps(dict(stage="combined_test_gate_locked",candidates=220,selections=110)),flush=True)


def evaluate():
    verify()
    gate_record = json.loads((PRIMARY/"combined_selection_gate.json").read_text(encoding="utf-8"))
    assert gate_record["supplement_selection_sha256"] == base.sha(HERE/"selection_lock.json")
    selection = json.loads((HERE/"selection_lock.json").read_text(encoding="utf-8"))
    test = base.load_split("test")
    y = test["y"].double().cpu().numpy()
    output = HERE/"test_predictions"
    output.mkdir(exist_ok=True)
    rows = []
    for choice in selection["selections"]:
        if choice["selected_id"] is None:
            rows.append(dict(**choice,status="all_candidates_failed",test=None))
            continue
        candidate = json.loads((HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
        path = HERE/candidate["checkpoint"]
        assert base.sha(path)==choice["checkpoint_sha256"]
        checkpoint = torch.load(path,weights_only=True,map_location="cpu")
        model = build(choice["seed"],choice["head"],1).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        prediction = base.predict(model,test,checkpoint["target_mean"],checkpoint["target_sd"])
        destination = output/(choice["selected_id"]+".npz")
        np.savez_compressed(destination,ids=np.asarray(test["ids"]),truth=y,prediction=prediction)
        rows.append(dict(**choice,status="valid",test=base.metrics(y,prediction),total_parameters=models.parameter_count(model),
             selected_epochs=candidate["epochs_run"],best_epoch=candidate["best_epoch"],selected_lr=candidate["spec"]["lr"],
             prediction_file=str(destination.relative_to(HERE)),prediction_sha256=base.sha(destination)))
        del model
    base.write_json(HERE/"evaluation.json",dict(evaluated_utc=base.now(),rows=rows,
        combined_gate_sha256=base.sha(PRIMARY/"combined_selection_gate.json"),selection_lock_sha256=base.sha(HERE/"selection_lock.json")))
    print(json.dumps(dict(stage="supplement_evaluation_complete",selected=len(rows))),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","lock","train","gate","evaluate"])
    args = parser.parse_args()
    base.configure()
    {"audit":audit,"lock":lock,"train":train,"gate":gate,"evaluate":evaluate}[args.stage]()
