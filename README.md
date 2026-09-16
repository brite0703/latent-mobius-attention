# Latent Möbius Attention: Bounded-Order Interaction Heads

Jih-Jeng Huang and Chin-Yi Chen

Scientific sources and retained experimental records for the manuscript. The versioned release provides the exact four scientific archives prepared for the 16 September 2026 revision. The manuscript and response letter are submitted separately to the journal.

## Download

Use [release v2026.09.16](https://github.com/brite0703/latent-mobius-attention/releases/tag/v2026.09.16). Expand its **Assets** list to download the following files.

| File | Size (decimal MB) | Contents |
| --- | ---: | --- |
| [09_reviewer_records.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/09_reviewer_records.zip) | 15.79 | Per-seed tables, measured costs, learning curves, routing arrays and diagnostic recalculations. |
| [12_training_inputs.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/12_training_inputs.zip) | 3.39 | Inputs and programs for the three-case CPU training example. |
| [13_training_completed.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/13_training_completed.zip) | 4.87 | Completed reference outputs for that training example. |
| [14_scientific_evidence.zip](https://github.com/brite0703/latent-mobius-attention/releases/download/v2026.09.16/14_scientific_evidence.zip) | 577.71 | Experiment sources, retained tensors, candidate and selection records, checkpoints, predictions, costs and provenance across thirteen scientific units. |

The automatically generated GitHub **Source code** downloads contain this repository's guide files, not the experimental archives. Download the named ZIP assets above. Files 12 and 13 are convenient separate copies of the corresponding units also contained in file 14; they are not additional experiments.

## Check and use the records

Compare the downloaded archives with [SHA256SUMS](SHA256SUMS). The hashes bind this release to the reviewed scientific files; the archives have not been rewritten for GitHub.

Extract file 14 to a short directory, for example `C:\LMA-evidence` on Windows. Some preserved paths are long. In the extracted directory, `python -I -B verify_files.py` checks the supplied files against the archive's manifest using only the Python standard library (Python 3.9 or later). This is a file-integrity check, not a model evaluation.

Read `EVIDENCE_GUIDE.md` and `PUBLIC_REPLAY.md` in file 14 before running an experiment. Source programs and their applicable dependency specifications are included inside the corresponding scientific units. Keep newly generated outputs outside the extracted archive.

## Execution scope

The synthetic collection contains 1,560 candidates and 780 validation choices. Its archived records-mode check regenerates inputs, reconstructs selections and recalculates metrics from saved predictions. The archived complete synthetic and receptor inference runs concern 780 and 130 retained predictors, respectively. These checks do not constitute fresh fitting or independent statistical replication.

Files 12 and 13 provide a separate CPU training example with six fits and three choices: order-two LMA on parity of length ten, the cubic target and a hierarchy of depth three. This example does not retrain all comparison methods.

The collection does not provide a single validated end-to-end procedure for fresh training of every comparison or reconstruction of every upstream molecular input. Original capture paths, historical protocols and execution receipts remain in the scientific units. Dated notes about earlier packaging and deposition status describe those earlier captures, not the current hosting status. No scientific result was changed or experiment rerun to prepare this release.

## Provenance and reuse

Retain the upstream notices included in the archives. In particular, the LP-PDBBind notice is specific to its original software and documentation and is not an unrestricted grant for all upstream annotations. Its pinned source is [THGLab/LP-PDBBind at cc363f80a6a696d562f290a87aa20173e60f6c52](https://github.com/THGLab/LP-PDBBind/blob/cc363f80a6a696d562f290a87aa20173e60f6c52/LICENSE.txt). The archive records the separate provenance of public PDB coordinate sources and derived tensors.

No additional blanket licence is granted for the collection. Existing upstream terms remain applicable.

## Version and availability

Cite the versioned release used in an analysis rather than an unversioned repository snapshot:

J.-J. Huang and C.-Y. Chen, *Scientific evidence for Latent Möbius Attention: Bounded-Order Interaction Heads*, version v2026.09.16, GitHub, 2026. https://github.com/brite0703/latent-mobius-attention/releases/tag/v2026.09.16

The released evidence will remain available after manuscript acceptance. If storage is migrated, preserve this version and its checksums and provide a link to the replacement archive.
