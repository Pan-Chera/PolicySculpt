# Hardware data collection by a human operator

Use this workflow when existing logs lack the fields or new interactions needed for a decision. By default, the agent identifies the gap and prepares the collection program. A human operator controls the device, starts and stops motion, resets the task, and labels outcomes. The agent analyzes returned logs offline. Follow explicit user arrangements and the actual interfaces when roles differ.

## Derive the collection plan from the question

State what existing logs can answer and what new evidence would change the decision. For each added field, specify source, sampling point, timestamp, units and frame, missing-value handling, and the hypothesis it helps distinguish. Reuse existing telemetry and the native policy runner. Analyze complete logs before collecting more.

To distinguish a bad proposed action from one that was never applied, collect same-cycle inputs, raw outputs, accepted commands, available driver application records, and sensor readback. Synchronized video can help inspect contact and human intervention. Record synchronization error and mark executed commands as unavailable when the driver does not report them.

Collect successes and naturally occurring failures under comparable conditions, with necessary repeats. Avoid inducing hazardous failures. Targeted enrichment is suitable for diagnosis. Report its composition. Before confirmation, freeze the candidate, order, conditions, and success criteria. Use `assets/collection-plan.template.json` for episode, time, and storage limits and stopping rules. Confirm motion authorization separately.

Use `scripts/check_experiment.py --collection-plan <plan.json>` when a record consistency check helps. Set `collector_mode` to `passive-recording` or `explicit-motion-runner`. Verify actual SDK behavior and the hardware stop separately.

Set a minimum useful deliverable, review point, and safe stop. Treat the budget as a ceiling. Collect additional data only for remaining gaps. If priorities change, have the operator stop admitting new episodes and retain acquired data with an incomplete marker where appropriate.

## Implement against the real interface

Verify the device model, SDK version, controller entry point, inference code, and logging interface. When an interface is unknown, list the gaps and provide a testable adapter skeleton. Record the collector version and dependencies. Identify its mode explicitly.

- **Passive recording.** Subscribe to telemetry produced by the existing operator or runner. Check whether connecting or constructing an SDK object has motion side effects. Keep enable, homing, gripper, and joint motion calls outside this path.
- **Collection with a motion runner.** Use the verified native runner and state what moves and when. Require an explicit motion entry point. Keep help, config checks, and simulated tests motion-free. Leave retries, resets, candidate switching, and deployment under explicit operator control.

When reusing a collection framework, verify its version, device integration, local data paths, and upload behavior. Prefer the site's existing runner and human execution entry point. Inspect upstream `replay` or `record --policy` before choosing a passive recorder. [LeRobot collection integration](open-source-tools.md#lerobot-collection)

Preserve the original controller, control rate, limits, and hardware protections. Keep synchronous large-image writes or compression off the real-time callback. Use bounded buffering, a separate writer, and counters for dropped frames and delay. Keep original sequence numbers. Decide how disk, writer, or buffer failures are signaled to the existing controller. The verified stop mechanism and operator retain responsibility for motion.

Give every episode a unique ID. Record session and case groups, actual model, runtime, controller and calibration, operator-selected conditions, reset, and start and end events. Start logging before reset when feasible and keep the necessary prefix. Record prior-state provenance, RNG, action history, chunk caches, and human intervention. Preserve raw sensor data and actual encoded policy inputs. Reference large files by timestamp or content hash.

Leave readable prefixes and incomplete markers for interruptions, emergency stops, and abnormal exits. Ask the operator to label success, failure, or uncertainty, interventions, and confidence in the cause. Keep those fields in the audit. Classify outcomes with human rescue separately.

## Minimum operator package

1. Collection purpose, motion status, compatible hardware and software, and frozen model version.
2. Runnable program, configuration, dependencies, and output location.
3. A fully hardware-free synthetic self-test, plus a separate device or field check verified to have no motion side effects where the interface permits it. State any read-only limitations.
4. A short run sheet covering config check, recording, normal operation and reset, outcome annotation, and final summary. Include normal shutdown, abnormal handling, and the native hardware stop. Explain the practical differences between closing the terminal, Ctrl-C, and the emergency stop.
5. Planned conditions and order, bounded limits and stopping rules, and the directory, manifest, or summary to return. Follow existing data export authorization.

Use synthetic inputs or existing logs to check sequence, timing, fields, state snapshots, model identity, episode boundaries, annotations, missing data, interruptions, storage limits, and output integrity. For first hardware use, have the operator verify a short segment before collecting the full planned batch. Physical motion, sensor latency, and contact safety need hardware evidence.

Reuse valid short-segment evidence when the collector, runtime, schema, and relevant calibration stay the same. Repeat checks when a new question or dependency invalidates that evidence. Verify identity, completeness, and interventions on the actual returned data.

## Review returned data

Compare planned and actual identities, conditions, interventions, and interruptions. Check baseline reconstruction on the first segment and identify log gaps. Keep collector versions and their records separate after fixes. Request the smallest useful next collection when evidence is missing.

Treat all diagnosis data as exposed once inspected. For human-run frozen confirmation, check actual independence, pairing, and log completeness. Record whether the operator knew the versions. For candidate closed-loop comparisons, hand the operator a separate frozen plan for baseline and candidate episodes. Historical behavior logs support claims about the policy that generated them.
