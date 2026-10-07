# Commands and input boundaries

~~~bash
python3 -I -B tools/check_sources.py
python3 -I -B tools/n20_entry.py fp32
python3 -I -B tools/n20_entry.py warm-fp64 --preflight
python3 -I -B tools/n20_entry.py random-fp64 --import-only
python3 -I -B tools/n20_entry.py native --import-only
python3 -I -B tools/fixed_report_entry.py selection
python3 -I -B tools/fixed_report_entry.py complete --preflight --evidence-root "$EVIDENCE_ROOT"
python3 -I -B tools/table_entry.py
python3 -I -B tools/table_entry.py --preflight --metadata-root "$METADATA_ROOT"
python3 -I -B tests/test_public_entrypoints.py
~~~

Run these commands from a source checkout. EVIDENCE_ROOT and METADATA_ROOT must name existing external directories separate from the checkout. No default private location is inferred. Default plans read public source identities but do not inspect external inputs, import scientific packages, create outputs or run scientific programs.

The new registry is configuration/public_entrypoints.json. Its 11 model/helper mappings identify existing public code by repository-relative source, logical import layout, bytes and SHA256. N20 preflight checks those identities and import-dependency availability. Import-only stages verified public code temporarily and imports an AST definition view. It omits only SIZES=head_sizes(), SIZES=sizes() and set_seed(42) at their explicitly mapped source locations. Original source bytes remain unchanged. Scientific main, checkpoint/array loads, model construction, forward/backward and CUDA initialization are blocked. This mode provides no data audit, predictions, parameter-count recomputation or training.

Fixed-report preflight checks external status metadata, selection-lock presence and context schema. Selection remains blocked during unfinished profiling; complete requires status=complete and train/evaluate/profile stages. Missing evidence fails closed. A passing metadata gate does not validate candidate/checkpoint/vector bindings or reproduce a report. The original reporter and its scientific guards are unchanged and are never executed by this entry.

Table plans work without retained_summary_manifest.json or presentation_spec.json. Preflight requires both files plus the six external JSON summaries named by the manifest; it checks byte identities, safe relative paths and structure without formatting or aggregate arithmetic. Future formatting requires an additional explicit flag:

~~~bash
python3 -I -B tools/table_entry.py --export-retained-tables --metadata-root "$METADATA_ROOT" --output-root "$NEW_TABLE_OUTPUT"
~~~

NEW_TABLE_OUTPUT must be a new directory separate from the checkout and metadata directory. The wrapper binds only the unchanged formatter's input/output layout; the formatter retains its identity, row-selection and output guards. No table export was executed during entry validation. The supplied summaries and presentation specification are external user inputs, and neither their numerical correctness nor full TeX equivalence is established by preflight.

The earlier tools/command_plan.py remains unchanged. Its configuration/entrypoints.json describes historical retained-record commands requiring complete extracted historical 14.

Historically documented synthetic records command, requiring original inputs/dependencies:

~~~bash
python -I -B units/retained_synthetic/public_replay.py records --output "$NEW_RECORD_OUTPUT"
~~~

Records mode recalculates saved-record quantities and was not run for this publication. Historical inference receipts are separate from records-mode validation. The receptor parser accepts only `records` or `replay`, with no `--output` option. It writes to `units/receptor_replay/bundle/replay_reports` and forces CPU execution with one thread. Use a fresh extracted working copy and inspect the original bundle protocol and unchanged guards before any future execution. Its records mode loads retained tensors; it is outside the source-only checks performed here. Inspect original training.py for the six-fit example; no new training command was executed or validated.

The original native N20 scientific programs still require omitted inputs, locks, execution protocols and environment constraints. Guarded definition imports and external metadata gates validate the limited public entry scope only.
