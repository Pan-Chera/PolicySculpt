---
name: embodied
description: Diagnose robot-policy trajectories, compare frozen policies in paired simulation, or audit generated, returned, and applied action chunks. Use for embodied control and robotics evaluation.
argument-hint: "[trace diagnosis, paired evaluation, or action-chunk audit]"
---

# Embodied policy diagnostics

Use the active request and `$ARGUMENTS` to choose the smallest useful check. The bundled Python tool reads native robot artifacts and evaluates supplied policies. A task can use this skill with an unedited policy, a trained checkpoint, or a weight-edited candidate.

The tool is at `${CLAUDE_SKILL_DIR}/tools/embodied.py`. See [command examples and adapter protocol](tools/README.md). Run the tool where the native logs, checkpoints, and runtime already reside. Use the task's native Python environment for visual inference and simulation. Keep server-resident artifacts on the server.

- **Locate a failure.** For two existing traces, run `diagnose` with the format matching the logs. Choose thresholds in the signals' native units. Check matched phase occurrences, first observed divergence, and event timing. Keep network outputs, projected commands, applied commands, and sensor readback distinct. Use same-state checks when a claim requires attributing a difference to the policy computation.
- **Compare frozen policies.** Use `evaluate` with a manifest. Recorded Goblet and Cube2 results can be imported with `recorded-manifest`. For new simulation, use the task's native evaluator and freeze models, initial states, evaluation rules, and relevant code and assets. The bundled Cube2 adapter provides an example. A resumable run reuses verified per-arm receipts. Score valid pairs and report repair, regression, technical failure, and safety stops according to the frozen rules.
- **Audit action chunks.** Use `audit-chunks` for a recorded execution trace, `visual-act` for the bundled adapter's fixed-observation comparison, or `visual-act-sim` for a bounded native simulation probe. Distinguish full predicted chunks, prefixes returned by inference, and commands applied by the controller. A bounded probe provides execution-chain evidence. Full task outcomes require a complete evaluation.

Follow the requested execution scope and the task's native controls. Use read-only diagnosis when the task supplies logs alone. Hardware collection and operation belong to the project's existing operator workflow.
