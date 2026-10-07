# Validation of the public entry update

Completed on 7 October 2026 against baseline commit 50472812494054d3c8ea03efe256f3b3b2e47bdf. The 317 baseline file bytes were matched to the pinned public Git tree. Validation used a clean copy containing those public files plus the exact thirteen-file candidate; no private evidence, data, checkpoints or aggregate payloads were present.

- The source checker passed SHA256/byte checks for 308 registered files and AST parsing for their 303 Python files. AST parsing also passed for all 305 repository Python files.
- All 19 entry tests passed with no skips. Public plans used only the standard library and needed no external input metadata. Tests created no files or changed source bytes inside the checkout.
- All four N20 entries passed guarded definition imports: fp32, warm-fp64, random-fp64 and native. All 11 model/helper mappings use existing public code, with explicit repository-relative paths, byte sizes and SHA256 identities.
- Definition imports used an explicit AST view omitting SIZES=head_sizes(), SIZES=sizes() and set_seed(42) at the three recorded source locations. Captured source bytes were unchanged. The checks imported definitions without calculating model sizes, constructing models or executing original scientific main.
- Negative checks rejected changed sources, broken origin bindings, unsafe relative paths and previously imported scientific modules. Guards blocked attempted scientific main, input-audit calls, checkpoint/array reads, model construction, forward/backward and CUDA initialization before those operations could execute.
- Fixed-report preflight rejected missing evidence, unfinished profile status and incomplete stage metadata. Tests of passing metadata gates used synthetic schema fixtures only; no actual retained evidence or status receipt was published or validated.
- Table plans read no external JSON. Preflight tests checked safe paths, manifest identities and structure using synthetic empty-list fixtures with no retained aggregate values. Missing/changed inputs and unsafe output locations were rejected before formatter execution. No table export was performed.

The existing environment used Python 3.13.5, NumPy 2.3.4, PyTorch distribution 2.9.0, scikit-learn 1.7.2 and Matplotlib 3.10.8. This was not a fresh dependency installation. CUDA remained uninitialized. No scientific main, checkpoint/tensor/array load, model construction, forward/backward, training, inference, profiling or numerical recalculation was performed.

These checks validate public source identity, code layout, plans, bounded metadata gates and a guarded definition view. Full original scientific execution in a clean public layout, retained-report reconciliation, native N20 fits/recovery, table-export execution or full TeX equivalence, fresh all-comparison training and upstream molecular reconstruction remain unvalidated. Original scientific inputs, guards, protocols and environment constraints still apply.

The historical release and its receipts retain their original dates and scope. Current aggregates and native N20 input/result payloads were not deposited into that release. Original source captures and release assets were unchanged; the new public helper supplies code-layout plumbing and does not replace the original scientific input audit.
