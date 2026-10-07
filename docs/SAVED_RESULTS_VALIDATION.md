# Bounded saved-display validation, 7 October 2026

The candidate `tools/saved_display.py` was executed with the retained public09/public14 ZIPs and the separately prepared current scalar extension. No fresh training, model inference, cost profiling or upstream reconstruction was performed. The earlier completed 19 basic source/import checks at source commit `0cefc8e29c4e587269a59aaac1cb646a9653cc02` were not repeated or recast as scientific validation.

The executed candidate operations were `plan`, `preflight`, all-14-table `tables --presentation original`, all-14-table `tables --presentation current`, and `curves --presentation current`. All completed with exit status zero. The four guarded operations recorded no model imports or checkpoint/array reads; writes were restricted to the private candidate output directory. Original sources, historical archives and current manuscript deliveries were preserved.

## Tables

The initial/original display variant matches numeric displays in 13/14 tables and all normalized cells in 12/14 tables. Four cell differences remain explicitly recorded:

| Table and location | Original variant | Current reference | Treatment |
| --- | --- | --- | --- |
| 3, Transformer depth 3 | `0.806 ± 0.014` | `0.807 ± 0.014` | explicit NumPy aggregation in current variant |
| 3, Count lookup depth 4 | `0.737 ± 0.020` | `0.738 ± 0.020` | explicit NumPy aggregation in current variant |
| 9, row 14, GCN 1 mean/count head | `Count + mean` | `Mean + count` | display label only |
| 9, row 22, GCN 3 mean/count head | `Count + mean` | `Mean + count` | display label only |

The two Table 3 means differ at floating-point rounding boundaries: standard-library means are `0.8065` and `0.7374999999999999`, while the current NumPy view gives `0.8065000000000001` and `0.7375`. Each mean difference is about `1.11e-16`. Current mode computes Table 3 mean and sample standard deviation using NumPy `mean` and `std(ddof=1)` from the same retained ten-seed values. It does not alter those values. Table 9 row order and numbers are unchanged; `mean_count` remains the machine key. The original formatter bytes remain unchanged.

With these expressly named display adaptations, the current variant matches numeric displays and all normalized cells in 14/14 tables, comprising 173 data rows and 1,043 cells. Comparison uses row/column topology and exact normalized cell/token strings, rather than a relaxed numerical tolerance. A TeX page-control line previously mistaken for a Table 9 data row was excluded from reference parsing; Table 9 has 48 actual data rows.

This verifies the saved display view. It does not independently validate the input measurements, recompute saved inferential statistics or assert full manuscript layout equivalence. In particular, copying saved confidence intervals or adjusted p-values into a matching table is not a fresh statistical analysis.

## Curves

The candidate regenerated all nine panels from 780 selected saved histories and 76,392 points. Its `curve_points.csv` is byte-identical to the existing public09 reference, with zero point-value tolerance. The current curve variant changes only four displayed LMA labels to LBOIA in an isolated renderer copy; original historical keys, selection locks and renderer source retain their identity.

The prior retained-reference comparison established equality of plotted data/marker geometry in 9/9 panels, with maximum coordinate difference zero under a 0.01-point tolerance. That earlier comparison did not establish whole-image equality: 0/9 complete SVG comparisons and 0/9 PNG pixel comparisons passed. The original and local rendering environments differ, and tick/text/layout differences remain; no single cause is asserted. The new candidate execution verifies exact saved CSV values and panel generation, not complete PNG/SVG or manuscript-PDF identity.

## Remaining boundaries

Tables 1–8 and 10 and the nine curve panels can use the existing public archives. The author approved the scalar extension and cell reference needed for all-14-table current-reference checking, with planned repository location `data/saved_display_v15/`. Upload and public availability remain unverified at this preparation checkpoint. This authorization/documentation update preserves all six previously checked payloads and does not rerun scientific validation. Full fresh all-comparison training and full upstream molecular reconstruction remain unvalidated.

The private review package contains the candidate execution receipts, bounded guard receipts, source identities, per-file numerical inventory and separate patch/data archives. No models, checkpoint arrays, private manuscripts or reviewer letters are part of the proposed new publication.
