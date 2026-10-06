"""An elementary graph-encoder indistinguishability check, not a new theorem."""
import json
import torch
import models
import study


def main():
    torch.set_num_threads(4)
    selection = json.loads((study.HERE/"selection_lock.json").read_text(encoding="utf-8"))
    x = torch.zeros(2, 6, 53, dtype=torch.float64)
    x[:, :, 0] = 1.
    mask = torch.ones(2, 6, dtype=torch.bool)
    adjacency = torch.eye(6, dtype=torch.float64).repeat(2, 1, 1)
    edges = [[(i, (i+1)%6) for i in range(6)],
             [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]]
    for index, graph_edges in enumerate(edges):
        for i, j in graph_edges:
            adjacency[index, i, j] = adjacency[index, j, i] = 1.
    assert torch.equal(adjacency.sum(-1), torch.full((2, 6), 3., dtype=torch.float64))
    adjacency /= 3.
    rows = []
    for choice in selection["selections"]:
        if choice["selected_id"] is None:
            continue
        record = json.loads((study.HERE/"candidates"/(choice["selected_id"]+".json")).read_text(encoding="utf-8"))
        checkpoint = torch.load(study.HERE/record["checkpoint"], weights_only=True, map_location="cpu")
        model = models.build_model(choice["seed"], choice["head"], choice["depth"]).double().eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            h = model.encoder(x, mask, adjacency)
            p = model(x, mask, adjacency)
        encoder_delta = float((h[0]-h[1]).abs().max())
        prediction_delta = float((p[0]-p[1]).abs())
        assert encoder_delta < 1e-12 and prediction_delta < 1e-10
        rows.append(dict(id=choice["selected_id"], encoder_maximum_delta=encoder_delta,
                         head_output_absolute_delta=prediction_delta))
    study.write_json(study.HERE/"encoder_limitation_check.json",
        dict(checked_utc=study.now(), source_sha256=study.sha(__file__), passed=True, rows=rows,
             graphs="C6 and two disjoint C3; six constant-feature nodes; degree-normalized adjacency with self loops",
             claim="Elementary compositional limitation; not novelty, molecule chemistry, or a PDBbind error bound",
             toy_equal_probability_labels=[-1, 1], exact_minimum_toy_squared_risk=1.))
    print(json.dumps(dict(passed=True, checked=len(rows),
                          maximum_delta=max(r["head_output_absolute_delta"] for r in rows))), flush=True)


if __name__ == "__main__":
    main()
