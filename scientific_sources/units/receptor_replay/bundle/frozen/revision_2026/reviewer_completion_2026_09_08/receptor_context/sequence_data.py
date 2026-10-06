"""Outcome-independent chain collation, batch partitioning and overlap manifest."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from sequence_encoder import VOCAB

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parents[1]/"data/lp_pdbbind/tensors_reconstructed/sample_manifest.csv"
MAX_RECORDS = 32
MAX_PADDED_TOKENS = 32768


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SequenceStore:
    def __init__(self, path=HERE/"published_sequences.csv"):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        self.by_id = {r["pdbid"]: r for r in self.rows}
        assert len(self.rows) == len(self.by_id)
        self.chains = {}
        for row in self.rows:
            parts = row["sequence"].split(":")
            if not all(parts) or not all(set(part) <= set(VOCAB) for part in parts):
                raise ValueError("Malformed sequence for "+row["pdbid"])
            self.chains[row["pdbid"]] = tuple(torch.tensor([VOCAB[a] for a in p], dtype=torch.uint8) for p in parts)
        self.sizes = {key: (len(value), max(map(len, value))) for key, value in self.chains.items()}

    def chunks(self, ids, max_records=MAX_RECORDS, max_tokens=MAX_PADDED_TOKENS):
        if max_records < 1 or max_tokens < 1:
            raise ValueError("Positive batch limits are required")
        current, chain_count, longest = [], 0, 0
        for key in ids:
            number, length = self.sizes[key]
            if number*length > max_tokens:
                raise ValueError(f"Single record {key} exceeds declared padded-token limit")
            if current and (len(current) == max_records or (chain_count+number)*max(longest, length) > max_tokens):
                yield current
                current, chain_count, longest = [], 0, 0
            current.append(key)
            chain_count += number
            longest = max(longest, length)
        if current:
            yield current

    def collate(self, ids, device="cpu"):
        if not ids:
            raise ValueError("At least one record is required")
        chains = [chain for key in ids for chain in self.chains[key]]
        owners = [i for i, key in enumerate(ids) for _ in self.chains[key]]
        tokens = torch.zeros(len(chains), max(map(len, chains)), dtype=torch.long)
        for i, chain in enumerate(chains):
            tokens[i, :len(chain)] = chain
        tokens = tokens.to(device)
        return tokens, tokens != 0, torch.tensor(owners, dtype=torch.long, device=device), len(ids)


def prepare_metadata():
    store = SequenceStore()
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    by_id = {r["pdbid"]: r for r in entries}
    assert len(entries) == len(by_id) == len(store.rows) == 10513
    assert set(by_id) == set(store.by_id)
    assert all(by_id[key]["split"] == row["split"] for key, row in store.by_id.items())
    fitting = [r for r in entries if r["split"] == "train"]
    validation = [r for r in entries if r["split"] == "val"]
    test = [r for r in entries if r["split"] == "test"]
    assert tuple(map(len, (fitting, validation, test))) == (7384, 958, 2171)
    reference = fitting+validation
    ligands = {r["canonical_smiles"] for r in reference}
    strings = {store.by_id[r["pdbid"]]["sequence"] for r in reference}
    chains = {c for s in strings for c in s.split(":")}
    full = {r["pdbid"] for r in test}
    absent_ligand = {r["pdbid"] for r in test if r["canonical_smiles"] not in ligands}
    absent_string = {key for key in full if store.by_id[key]["sequence"] not in strings}
    absent_chain = {key for key in full if not set(store.by_id[key]["sequence"].split(":")) & chains}
    assert absent_chain <= absent_string
    groups = dict(full=full)
    for name, absent in [("canonical_ligand", absent_ligand), ("complete_stored_string", absent_string),
                         ("any_supplied_chain", absent_chain), ("both_ligand_and_any_chain", absent_ligand & absent_chain)]:
        groups[name+"_absent_from_train_and_val"] = absent
        overlap_name = "either_ligand_or_any_supplied_chain" if name == "both_ligand_and_any_chain" else name
        groups[overlap_name+"_overlap_with_train_or_val"] = full-absent
    chain_audit = json.loads((HERE/"chain_audit.json").read_text(encoding="utf-8"))
    for audit_key, absent in [("test_exact_sequence_overlap", absent_string), ("test_exact_chain_overlap", absent_chain)]:
        union = set(chain_audit[audit_key]["with_train"]) | set(chain_audit[audit_key]["with_val"])
        assert full-absent == union
    chunks_by_split = {}
    for split in ("train", "val", "test"):
        ids = [r["pdbid"] for r in entries if r["split"] == split]
        chunks = list(store.chunks(ids))
        assert [key for group in chunks for key in group] == ids
        work = [sum(store.sizes[key][0] for key in group)*max(store.sizes[key][1] for key in group) for group in chunks]
        assert max(work) <= MAX_PADDED_TOKENS and max(map(len, chunks)) <= MAX_RECORDS
        chunks_by_split[split] = dict(records=len(ids), manifest_order_microbatches=len(chunks), maximum_padded_tokens=max(work),
            maximum_record_padded_tokens=max(store.sizes[key][0]*store.sizes[key][1] for key in ids),
            maximum_chain_length=max(store.sizes[key][1] for key in ids),
            maximum_supplied_chains=max(store.sizes[key][0] for key in ids))
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, records=len(entries),
        numerical_targets_accessed=False, model_predictions_accessed=False,
        groups={name: sorted(value) for name, value in groups.items()}, subset_sizes={name: len(value) for name, value in groups.items()},
        batch_rule=dict(maximum_records=MAX_RECORDS, maximum_padded_chain_tokens=MAX_PADDED_TOKENS,
                        effective_batch=256, preserve_order=True, no_sequence_truncation=True),
        batching=chunks_by_split, sources=[dict(path=str(p), sha256=sha(p)) for p in [MANIFEST, HERE/"published_sequences.csv", HERE/"chain_audit.json", Path(__file__)]],
        qualifications=["Equality of supplied strings/chains is not biological receptor identity or similarity-threshold separation.",
                        "Subsets describe a reused test population; no refitting, reselection or causal leakage estimate is implied.",
                        "The joint-overlap complement means ligand OR any supplied chain has an exact match.",
                        "Per-split batching uses manifest order for coverage checks; training preserves each seed's random effective-batch order."])
    target = HERE/"sequence_metadata.json"
    if target.exists():
        raise FileExistsError("Preserve the existing metadata record; an amendment needs a new file")
    target.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    prepare_metadata()
