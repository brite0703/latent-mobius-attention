# Commands and input boundaries

~~~bash
python3 -I -B tools/check_sources.py
python3 -I -B tools/command_plan.py --help
python3 -I -B tools/command_plan.py synthetic-records --evidence-root /absolute/extracted14 --output /absolute/new-record-output
~~~

The plan utility prints only. Its registry is configuration/entrypoints.json; scientific templates refer to complete extracted historical 14, not this source-only tree.

Historically documented synthetic records command, requiring original inputs/dependencies:

~~~bash
python -I -B units/retained_synthetic/public_replay.py records --output /absolute/new-record-output
~~~

Records mode recalculates saved-record quantities and was not run for this publication. Historical inference receipts are separate from records-mode validation. The receptor parser accepts only `records` or `replay`, with no `--output` option. It writes to `units/receptor_replay/bundle/replay_reports` and forces CPU execution with one thread. Use a fresh extracted working copy and inspect the original bundle protocol and unchanged guards before any future execution. Its records mode loads retained tensors; it is outside the source-only checks performed here. Inspect original training.py for the six-fit example; no new training command was executed or validated.

The native N20 and table formatter captures remain source-only: their omitted frozen/private inputs and clean-layout runtime are not validated. Old private relocation adapters are not silently relabeled as a complete public execution path.
