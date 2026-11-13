# Standalone embodied policy tools

Use these commands to compare robot traces, evaluate or resume paired simulations, and audit action chunks. They accept frozen policies from any editing or training workflow. Trace, receipt, and chunk tools use the Python 3.10+ standard library. Run native simulation and visual inference in their task-specific environments.

## Diagnose trajectories

```bash
python3 embodied.py diagnose BASELINE.jsonl.gz CANDIDATE.jsonl.gz \
  --format canonical --threshold object_xyz_m=0.001
```

Use `--format goblet` or `--format cube` for the supported native logs. Other tasks can use canonical JSONL with `tick`, `phase`, `signals`, and `events` fields.

```json
{"tick":240,"phase":"handoff","signals":{"object_xyz_m":[0.18,0.02,0.29]},"events":{"contact":false}}
```

The tool aligns phase occurrences and relative ticks. It reports unmatched spans, the first signal difference above the threshold, and the first occurrence of each event. Set thresholds in the signal's native units. Preserve the distinction between policy outputs, projected commands, applied commands, and sensor readings. For example, Goblet's `projected_joint_reference` is a projected target and `measured_joint_rad` is a measured state. An absent event field reduces coverage. An explicit `false` records that the event was not observed. Use a same-state comparison to attribute a closed-loop difference to the policy.

## Run or resume paired evaluations

With access to the original RSI experiment archives, import recorded Goblet or Cube2 evaluations:

```bash
python3 embodied.py recorded-manifest goblet /path/to/rsi-xarm7-g2 > goblet.json
python3 embodied.py evaluate goblet.json results/goblet --limit-arms 7
python3 embodied.py evaluate goblet.json results/goblet

python3 embodied.py recorded-manifest cube /path/to/rsi-xarm7-g2 > cube.json
python3 embodied.py evaluate cube.json results/cube
```

Recorded mode imports existing receipts. For Goblet it verifies model identity, seed, direction, and native initial state. For Cube2 it uses the frozen move sequence, models, native receipts, and acceptance rules.

With the Cube2 simulator and archive installed, run one frozen sample using the native MuJoCo evaluator:

```bash
python3 embodied.py cube-simulation-manifest /path/to/rsi-xarm7-g2 6 \
  --runtime /path/to/rsi-xarm7-g2/.environments/black-bowl/bin/python > cube-live.json
python3 embodied.py evaluate cube-live.json results/cube-live --limit-arms 1
python3 embodied.py evaluate cube-live.json results/cube-live
```

For other simulators, create a manifest with `mode: simulation`, two `models`, `evaluator` file references, a `command` with `argv`, `cwd`, `timeout_s`, and optional `environment`, plus samples with an `id` and native `initial` value. File references contain an absolute `path` and a `sha256`. Bind the evaluator dependencies, including relevant runtime, adapter code, configuration, and assets.

The command reads one JSON request from stdin. It contains `case_id`, `role`, the model reference, and the native initial state and its hash. Stdout returns one JSON receipt with the request hash, the hash of the model actually loaded, the hash of the initial state passed to the native evaluator, and a `result`. The initial-state hash identifies the input to reset. To verify the resulting state, also include the simulator's readback. A valid result uses `status: valid` and a boolean `success`. Record technical failures as `technical_invalid` and external safety stops as `safety_stop`. A task-defined simulator collision can count as a failure under the original rule and carry its own flag.

Each policy side gets its own receipt. A resumed evaluation skips verified receipts and counts only valid pairs in success, repair, and regression totals. Changes to the bound inputs or receipts prevent reuse of the output directory. Results include `summary.json`. Use the project's native operator workflow for hardware execution.

## Audit action chunks and visual ACT

```bash
python3 embodied.py audit-chunks TRACE.jsonl

/path/to/native/python embodied.py visual-act JOB_DIR JOB_DIR/demos/validation_00_0.npz \
  --runtime /path/to/native/python

/path/to/sim/python embodied.py visual-act-sim JOB_DIR \
  --actor-runtime /path/to/native/python --groups 3
```

Record generated, sent, and applied actions as separate events, using `generated: {id, actions}`, `sent: {id, index, action}`, and `applied: {id, index, action}`. Multiple events can share a tick. The tool reports replanning intervals, visible prefixes, stale chunks, skipped indices, and action value differences. Interpret clipping and interpolation against the native controller.

`visual-act` reconstructs the full action chunk on fixed observations using the archived Cube2 ACT model, paired camera images, and joint history. It compares the native inference subprocess's returned prefix. `visual-act-sim` additionally records the interpolated and applied command at each physics substep, using the task's EGL configuration and native Python environments. The default probe covers three replans and 24 physics substeps. Use a complete frozen evaluation for task success claims.

## Verification

The archived tool checks covered the 28 Goblet and 30 Cube2 recorded pairs, an actual Cube2 case 06 simulation resumed on the candidate side, and fixed-input and simulated ACT probes. These task-specific inputs live outside the GitHub checkout.

Run the bundled tests from this directory with `python3 -m unittest discover -s . -v`.
