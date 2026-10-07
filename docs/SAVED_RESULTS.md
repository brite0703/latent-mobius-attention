# Saved-result table and curve displays

This candidate extends source commit `0cefc8e29c4e587269a59aaac1cb646a9653cc02`. It formats retained scalar results and plots retained loss histories. It does not execute a model, load checkpoint arrays, train, run inference, profile, reconstruct upstream data, or validate full fresh all-comparison reproduction.

The entry is `tools/saved_display.py`. It reuses the unchanged table formatter and the recovered author-origin curve renderer. Historical `lma*` keys and release paths retain their identity; human-readable current labels use Latent Bounded-Order Interaction Attention (LBOIA). The `original` and `current` options select the expressly documented display variants, rather than different scientific results.

## Inputs and output coverage

The existing public release supplies Tables 1–8 and 10 and all nine learning-curve panels. The author approved the seven-file current scalar/reference package for the planned repository directory `data/saved_display_v15/`. It supplies Tables 9 and 11–14 and automatic comparison with the current reference cells. Publisher upload and public availability remain unverified at this preparation checkpoint. See [acquisition, hashes and schema](SAVED_RESULT_INPUTS.md).

| Table | Retained source | Extension needed | Output |
| --- | --- | --- | --- |
| 1, first-degree/cubic | public14 `first_cubic/evaluation.json` | no | 12 display rows |
| 2, parity | public14 `synthetic_parity/evaluation.json` | no | 11 display rows |
| 3, hierarchical sequence | public14 `hierarchical_sequence/evaluation.json` | no | 7 display rows |
| 4, sequence context | public14 `sequence_summary.json` | no | 5 display rows |
| 5, matched context | public14 matched-study `summary.json` | no | 5 display rows |
| 6, molecular comparisons | public14 `combined_results_summary.json` | no | 16 display rows |
| 7, components | public14 components/width `summary.json` | no | 6 display rows |
| 8, width | same components/width summary | no | 6 display rows |
| 9, retained accuracy/cost display | current saved projection `table_9.csv` | yes | 48 display rows |
| 10, routing | public14 `routing_diagnostics.json` and components summary | no | 12 display rows |
| 11, clipping contrast | saved `table_11.csv` | yes | 4 display rows |
| 12, multiplicity family summary | saved `table_12.csv` | yes | 14 display rows |
| 13, selected paired evidence | saved `table_13.csv` | yes | 20 display rows |
| 14, native N20 diagnostics | saved `table_14.csv` | yes | 7 display rows |

The three synthetic source paths above are relative to `units/retained_synthetic/frozen/revision_2026/reviewer_completion_2026_09_08/` inside public14. Exact other member paths are listed in the acquisition guide. Table output consists of `table_N_cells.csv`, a simple `table_N.tex` tabular fragment and `receipt.json`. These fragments do not reconstruct full manuscript captions, page layout or typography. Saved confidence intervals, multiplicity corrections, costs and native N20 values are read unchanged; they are not newly calculated.

The nine curve output stems are `first_cubic_first`, `first_cubic_cubic`, `synthetic_parity_10`, `synthetic_parity_20`, `synthetic_parity_40`, `synthetic_parity_80`, `hierarchical_sequence_depth2`, `hierarchical_sequence_depth3`, and `hierarchical_sequence_depth4`. Each has PNG and SVG output in the output root's `work/learning_curves/` directory, alongside `curve_points.csv` and `curve_manifest.json`. The recovered renderer writes the combined PDF at `work/learning_curves/output/pdf/synthetic_learning_curves.pdf`. Its saved-run selection is checked against the historical candidate locks. The CSV is compared byte for byte with the public09 reference. Whole-image pixel or SVG identity is not asserted.

## Commands

Run from the candidate source checkout. Set `ARCHIVE09` and `ARCHIVE14` to existing downloaded ZIP files. Set `NEW_OUTPUT` to a fresh directory outside the checkout. No implicit private path, network download or historical-input overwrite occurs.

~~~bash
python3 -I -B tools/saved_display.py plan

python3 -I -B tools/saved_display.py preflight \
  --historical-09 "$ARCHIVE09" --historical-14 "$ARCHIVE14"

# Default: Tables 1–8 and 10, using existing public inputs only.
python3 -I -B tools/saved_display.py tables \
  --historical-14 "$ARCHIVE14" --output-root "$NEW_OUTPUT"

# Nine saved-history panels; no current scalar extension required.
MPLCONFIGDIR="$MPL_CACHE" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -I -B tools/saved_display.py curves \
  --historical-09 "$ARCHIVE09" --historical-14 "$ARCHIVE14" \
  --presentation current --output-root "$NEW_CURVE_OUTPUT"
~~~

`MPL_CACHE` must name a writable cache directory; `NEW_CURVE_OUTPUT` must be a fresh output directory. Table display with `--presentation original` uses only Python's standard library. `--presentation current` uses NumPy for Table 3 aggregation. Curve rendering requires Matplotlib. The recorded candidate environment is Python 3.13.5, NumPy 2.3.4 and Matplotlib 3.10.8; it is a saved-display environment, not a validated training environment.

After the approved payload is present in the checkout, set `CURRENT_INPUTS="$PWD/data/saved_display_v15"`. This explicit directory must contain `manifest.json` and all six bound payload files:

~~~bash
python3 -I -B tools/saved_display.py preflight \
  --historical-14 "$ARCHIVE14" --current-inputs "$CURRENT_INPUTS"

OPENBLAS_NUM_THREADS=1 python3 -I -B tools/saved_display.py tables \
  --historical-14 "$ARCHIVE14" --current-inputs "$CURRENT_INPUTS" \
  --tables 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --presentation original --output-root "$NEW_ORIGINAL_TABLE_OUTPUT"

OPENBLAS_NUM_THREADS=1 python3 -I -B tools/saved_display.py tables \
  --historical-14 "$ARCHIVE14" --current-inputs "$CURRENT_INPUTS" \
  --tables 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --presentation current --output-root "$NEW_CURRENT_TABLE_OUTPUT"
~~~

The unchanged `plan` operation contains a static preparatory publication-status field, not a network availability check. Pass the explicit `--current-inputs` directory for the approved scalar package. Each table execution includes a per-table reference comparison when the extension is supplied. There is no separate `verify` operation. Selecting an extension table without its input fails. A successful metadata preflight alone does not establish result equivalence. Inspect `reference_comparison` in the emitted receipt; original-format differences remain recorded rather than being hidden by a relaxed comparison tolerance.

The earlier basic source/import validation remains a separate, completed record. [Saved-display validation](SAVED_RESULTS_VALIDATION.md) describes the added bounded execution and its remaining limits.
