"""Independent Fraction verification of saved cubic representation witnesses.

This does not import the diagnostic's dyadic/common-denominator implementation.
No model fitting, target prediction, validation selection, or CUDA work occurs.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fraction(value):
    return Fraction.from_float(float(value))


def main():
    destination = HERE / "linear_access_independent_verification.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed independent verification")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    diagnostic_path = HERE / "cubic_linear_access.json"
    diagnostic = read(diagnostic_path)
    manifest_path = HERE / "caches/cubic_v1/manifest.json"
    manifest = read(manifest_path)
    for source in diagnostic["sources"] + diagnostic["artifacts"] + manifest["sources"] + manifest["files"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Changed source or representation artifact: " + source["path"])
    checks = []
    for record in diagnostic["checks"]:
        seed = record["seed"]
        with np.load(record["certificate_artifact"], allow_pickle=False) as saved:
            routing, values = saved["routing"], saved["values"]
            dictionary, inverse = saved["dictionary"], saved["inverse"]
        assert routing.shape == (12, 8) and values.shape == (12, 12)
        assert dictionary.shape == (96, 12) and inverse.shape == (12, 96)
        assert routing.dtype == values.dtype == np.float32
        assert dictionary.dtype == inverse.dtype == np.float64
        assert all(np.isfinite(a).all() for a in (routing, values, dictionary, inverse))
        k = [[fraction(value) for value in row] for row in dictionary]
        q = [[fraction(value) for value in row] for row in inverse]
        product_checks = 0
        for bucket in range(8):
            for coordinate in range(12):
                for token in range(12):
                    assert k[12 * bucket + coordinate][token] == (
                        fraction(routing[token, bucket]) * fraction(values[token, coordinate])
                    )
                    product_checks += 1
        residual = []
        for output in range(12):
            row = []
            for token in range(12):
                entry = sum((q[output][j] * k[j][token] for j in range(96)), Fraction(0))
                row.append(Fraction(int(output == token)) - entry)
            residual.append(row)
        bound = max(sum((abs(entry) for entry in row), Fraction(0)) for row in residual)
        expected = record["exact_neumann_residual_bound"]
        assert bound == Fraction(int(expected["numerator"]), int(expected["denominator"]))
        assert bool(bound < 1) == record["dictionary_full_column_rank_certified"]
        assert bool(bound < Fraction(1, 2)) == record["all_exact_dictionary_boolean_inputs_round_correctly"]
        assert bound < Fraction(1, 2)
        # Direct Fraction checks of a fixed, outcome-independent six-row sample.
        # The separate diagnostic retains the complete 30,720-row integer check.
        sample = []
        for split in ("train", "val"):
            cache = torch.load(HERE / f"caches/cubic_v1/seed{seed}_{split}.pt", weights_only=True, map_location="cpu")
            assert cache["parent_checkpoint_sha256"] == record["parent_checkpoint_sha256"]
            for position in (0, len(cache["ids"]) // 2, len(cache["ids"]) - 1):
                identifier = cache["ids"][position]
                integer_id = int(identifier.split(":")[1])
                z = [fraction(value) for value in cache["z"][position].reshape(-1).tolist()]
                estimates = [sum((entry * value for entry, value in zip(row, z)), Fraction(0)) for row in q]
                bits = [(integer_id >> j) & 1 for j in range(12)]
                assert [int(value >= Fraction(1, 2)) for value in estimates] == bits
                error = max(abs(value - bit) for value, bit in zip(estimates, bits))
                reported_split = next(row for row in record["native_cache_recovery"] if row["split"] == split)
                reported_max = reported_split["maximum_coordinate_error"]
                assert error <= Fraction(int(reported_max["numerator"]), int(reported_max["denominator"]))
                sample.append(dict(split=split, position=position, id=identifier, recovered_bits_exact=True))
        checks.append(dict(seed=seed, exact_dictionary_products_checked=product_checks,
                           matrix_entries_checked=144, exact_bound_matches=True,
                           exact_bound_numerator=str(bound.numerator), exact_bound_denominator=str(bound.denominator),
                           full_column_rank_certified=True, native_sample_checks=sample))
    assert len(checks) == 10 and not torch.cuda.is_initialized()
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  independent_method="Python Fraction per-entry multiplication and summation; no import or use of the original dyadic matrix implementation",
                  parent_certificates_verified=len(checks), matrix_entries_verified=144 * len(checks),
                  dictionary_products_verified=1152 * len(checks), native_sample_records_verified=6 * len(checks),
                  complete_native_recovery_source_records=diagnostic["native_record_parent_pairs"],
                  native_sampling_scope="Positions 0, floor(n/2), n-1 in each fitting/validation cache; the complete recovery claim comes from the separate exact-integer diagnostic",
                  new_test_predictions=False, optimizer_updates=0, cuda_initialized=False,
                  sources=[dict(path=str(path), sha256=sha(path)) for path in (Path(__file__), diagnostic_path, manifest_path)],
                  checks=checks)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "parent_certificates_verified", "matrix_entries_verified", "native_sample_records_verified", "new_test_predictions")}))


if __name__ == "__main__":
    main()
