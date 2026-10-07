# Saved-display acquisition and input boundary

The existing [v2026.09.16 release](https://github.com/brite0703/latent-mobius-attention/releases/tag/v2026.09.16) supplies the retained inputs below. Preserve the historical asset names and verify SHA-256 before use. The entry reads ZIP members directly; downloading or extracting full datasets is not an entry action.

| Existing asset | Bytes | SHA-256 | Coverage |
| --- | ---: | --- | --- |
| [09_reviewer_records.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/09_reviewer_records.zip) | 15790995 | `ebc26d627af851479148e8df88e07faeaf2dea9fefad306724de2fb84dd43d17` | curve reference CSV, manifest and historical PDF |
| [14_scientific_evidence.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/14_scientific_evidence.zip) | 577710629 | `1c780fd4462d65ec9dd62c4b7fa3533f41642744b7428f19befb2a3507ecb902` | saved aggregates, routing records, curve selection locks and candidate histories |

~~~bash
sha256sum "$ARCHIVE09" "$ARCHIVE14"
~~~

Public09 curve members used are `curves/curve_points.csv` and `curves/curve_manifest.json`. The former is identical to the retained reference used by the bounded validation. Public14 contains three `evaluation.json` files, three `selection_lock.json` files and three `final_audit.json` files under:

~~~text
units/retained_synthetic/frozen/revision_2026/reviewer_completion_2026_09_08/first_cubic/
units/retained_synthetic/frozen/revision_2026/reviewer_completion_2026_09_08/synthetic_parity/
units/retained_synthetic/frozen/revision_2026/reviewer_completion_2026_09_08/hierarchical_sequence/
~~~

The curve operation reads the 1,560 JSON candidate records bound by those selection locks. It does not load their model/checkpoint arrays. The selected histories comprise 780 runs and 76,392 plotted points. Public14 aggregate members for Tables 4–8 and 10 are:

| Member | SHA-256 |
| --- | --- |
| `units/receptor_replay/bundle/frozen/revision_2026/reviewer_completion_2026_09_08/receptor_context/sequence_summary.json` | `56ae5f496042f350f867541880a9e28bcc09c71bdf0ba628d03533380e3d4659` |
| `units/receptor_replay/bundle/frozen/revision_2026/reviewer_completion_2026_09_08/receptor_context/pocket_reconstruction/matched_study/summary.json` | `c12b3e461bc99b056625442d005b13b0baff2fcc61edb42502d0d60b0ca9c10d` |
| `units/core_cardinality_molecular/combined_results_summary.json` | `18a2206a3e79fe6b5047e73e750e227e89f5eb81d6f8f59203ccb628bd525d02` |
| `units/components_width/summary.json` | `fba320ff52029d63fb01ca0f2b36a2e9d37e4ef6b1b759c69eed12ba2cb5a0dd` |

Routing additionally reads `units/components_width/routing_diagnostics.json`. Synthetic aggregates and routing are bound to the verified full archive identity above; the entry separately checks the four aggregate-member hashes and all locked curve candidate identities. Use the pinned archives rather than an arbitrary replacement ZIP.

## Approved current scalar/reference package

The author approved publication of exactly five scalar CSV files, `reference_cells.json` and `manifest.json` in [the existing repository](https://github.com/brite0703/latent-mobius-attention), at the planned relative directory `data/saved_display_v15/`. Upload and public availability have not yet been verified at this preparation checkpoint; use a checkout containing these files after publisher verification. Six payload files total 57,663 bytes; `manifest.json` adds the schema, each payload's exact byte length and SHA-256, and approval/provenance metadata. Its `publication_authorized` field is `true`, which records authorization rather than proof of public upload. The entry requires all six manifest-bound files when the extension is supplied. No new release tag or upstream-data licence is implied.

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `table_9.csv` | 48 | 19422 | `0295a368a6ed1a1e19f3482f38ee054f36bc7dde179cb2ad4bf8fa33c99226d0` |
| `table_11.csv` | 4 | 497 | `9fe37d70fc1bb5cf2a1aa6ea84ad533a7fa053922995c1f92588340aa10573bc` |
| `table_12.csv` | 14 | 1112 | `a839c6719ace18a9866647c602a9b3ef29546179b31a6e89fbdb16438e89c214` |
| `table_13.csv` | 20 | 4078 | `9dbb7b31c9bc2652865d40ea0f56368ea9cadf83c471c11b4370320f3fa6cccf` |
| `table_14.csv` | 7 | 848 | `f705d5ff4d4a12b73e4f04351d6ef5b27a63384c74e8ab7a639f5b30524d664f` |
| `reference_cells.json` | 173 table rows | 31706 | `56e2bab349f5ffe0a7672e972e54b0ce1e1b858b9383f0877e1fbd5ae3d6c8fc` |

The CSV field schemas, in order, are:

~~~text
table_9.csv: campaign,setting,head,metric,seeds,mean,sample_sd,parameters,forward_batch,forward_median_ms,train_batch,train_step_median_ms,maximum_incremental_train_MiB,profile_seed,scope
table_11.csv: n,id,mean_difference,paired_sample_sd,interval_low,interval_high,clip1_successes,clip10_successes
table_12.csv: family,scope,m,t_Holm_below_005,sign_flip_Holm_below_005
table_13.csv: id,label,context,contrast,metric,family,n,mean_difference,t_reference_interval95,t_p_raw,family_t_p_Holm,family_sign_flip_p_Holm,global_t_p_Holm
table_14.csv: panel,seed,warm_initial_accuracy_percent,warm_initial_ce,random_final_ce,warm_final_ce,warm_minus_random_final_ce,initializer,precision,seed100_percent,seed101_percent,seed102_percent,successes,denominator
reference_cells.json: basis,tables; each table contains label,page,cells
~~~

Table 9 retains a projection of existing saved measurements, including their scope fields; it does not represent new profiling. Historical public archives contain underlying cost records, but not this exact current 48-row projection. Tables 11–14 preserve the current saved clipping, multiplicity and native-N20 views without executing their producing experiments. Table 13 contains 20 displayed contrasts and Table 12 family counts; this small package does not redistribute the full 692-contrast analysis ledger. The cell reference supplies 14 tables, 173 rows and 1,043 cells for display comparison, not raw experimental verification.

The approved payload excludes raw PDB structures/sequences, sample or complex identifiers, predictions, model weights, checkpoints, images, reviewer correspondence, manuscript prose/captions, private filesystem paths and account identifiers. Table 9 drops the `profile_source` and `metric_source` path fields. Historical model keys remain stable identifiers.

The author approved this exact seven-file numerical package without adding MIT, CC or other new licence terms. No additional blanket licence is granted by this payload; applicable existing notices remain. The pinned [LP-PDBBind software notice](https://github.com/THGLab/LP-PDBBind/blob/cc363f80a6a696d562f290a87aa20173e60f6c52/LICENSE.txt) grants educational, research and not-for-profit software/documentation use subject to notice preservation; it does not establish redistribution rights over all PDB-derived materials. No upstream dataset or vendor code is newly included here.
