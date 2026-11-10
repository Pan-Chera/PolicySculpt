# Hardware confirmation with new executions

Offline checks establish contract conformance, computation, and some output constraints within their coverage. Physical distributions, contact, collision, and task success require physical evidence. Discuss off-policy evaluation only with credible support assumptions, calibration, and independent assessment. Low-coverage continuous-action logs from a deterministic policy generally lack usable importance ratios. Train a world model or critic only as a separately scoped task.

Use a bounded closed-loop probe to learn an unknown response. It can stop adaptively within predetermined bounds and remains development evidence. Use formal confirmation to support a frozen candidate's benefit claim. Propose a limited human-run probe when it can resolve a physical uncertainty. [Research efficiency](research-efficiency.md)

## Run sheet

Within existing authorization, identify the robot, driver, calibration, controller, baseline and candidate hashes, task, workspace, observable initial conditions and tolerances, action units, control rate and chunk length, native interlocks and human stop, success criteria, order, budget, review point, stop conditions, and rollback. Hand the plan to the operator when the interface and authorization are ready. If motion authorization or key conditions are missing, deliver a reviewable plan and continue applicable offline work.

Preserve native limits, collision and force controls, workspace limits, action monitoring, and hardware stop. Apply offline output limits and retention thresholds separately.

## Initial conditions and statistics

Record joints and velocities, object estimates and confidence, gripper and scene state, and relevant temperature, battery, or calibration. Exact bitwise hardware resets are unnecessary. Define strata before the run and randomize the baseline and candidate order within comparable conditions. Repeat across days or operators when relevant. Apply initial-condition rules before seeing outcomes.

Use paired statistics only for genuine pairs. Analyze two independent runs under similar distributions as independent or stratified samples. Repeated runs of one case measure repeatability. Document actual initial states even when seeds match.

Freeze one candidate and the protocol before confirmation. Keep baseline-selected failures in targeted development. Preserve all valid policy failures and regressions. Use the fixed endpoint or a valid prespecified sequential design. Record later research questions separately with their multiplicity limits.

Choose the endpoint for the effect, precision, and decision. Report inconclusive results when precision is limited. Predetermine task-based retention trade-offs and keep absolute safety conditions separate.

## Stops and interpretation

- **Technical invalidation.** Preserve the prefix and cost when clocks, logging, model identity, or the execution chain fail. Fix and rerun within applicable authorization.
- **Safety stop.** Use the original hardware, workspace, or monitoring stop. Preserve the frozen thresholds.
- **Policy failure.** Count valid timeouts, failures to release or hold, and other unmet task criteria.
- **Scientific endpoint.** Archive at the planned sample, a falsified hypothesis, or a failed promotion criterion. Treat deployment as a separate action.
- **User or priority change.** Stop admitting new work and let active episodes or pairs finish at a safe boundary. Record the request, acknowledgment, actual stop, and completed sample. Mark interrupted fixed-endpoint confirmation as incomplete.

Timestamp offline gates, pre-step interlocks, post-episode checks, and retrospective statistics separately. A recorded peak force and a real-time force interlock answer different questions. Report phase acceptance only for phases entered.

Report task success counts, failure stages, interventions and stops, strata, time cost, and effect intervals. Wide intervals or zero discordant pairs give limited evidence about equivalence. Use the frozen task success definition. Separate case repairs, independent gains, and deployment.

<a id="execution-verification"></a>
## Execution verification and reliability

Use the checks relevant to the failure in simulation or on hardware.

- **Wrong model hash binding.** Start with the expected SHA-256 for each arm in the frozen plan. Check manifest binding, the checkpoint resolved during preflight, and the file actually loaded by the worker, including resume paths. Hash the actual file and verify model identity from execution records. Trace mismatches to plan references, load paths, or resume config before rechecking the fix.
- **Loss of native initial-state precision.** Use the frozen native state or a trusted native record and preserve precision, units, and frames. Check round-trip values through actual serialization, loading, and native validation. For existing episodes, inspect the post-reset state before the policy acts. Apply preset tolerances and pairing rules. Mark initial-state identity unverified if only a rounded summary or seed is available.
- **Missing parent-process dependency.** Locate the first error in the scheduler parent or worker and record the interpreter, working directory, dependency versions, and search paths. Test the minimal imports and native libraries used before dispatch in the real parent launch environment. A successful worker import establishes only worker availability. Use a preflight with no simulator steps or device motion.

For execution anomalies, record stage, expected and actual values, evidence, admission, and cost. Log rejections before admission as preflight blocks. For admitted runs with verified invalid conditions, apply the prespecified technical invalidation rule. Preserve records, any completed counterpart, and actual cost. Include only valid pairs in paired statistics and flag unresolved cases. Count unmet criteria on valid runs as policy failures. A baseline success paired with a candidate failure is a regression. Record safety stops separately under the frozen rules.

Before queuing, check for an active worker on the same task. Preserve failure costs and RNG and provenance for resumed work. Match the resource measurement domain to the admission threshold.

Reuse valid frozen snapshots and reconstruction evidence. Check cooperative stop requests before admitting each pair or episode. Match monitoring to episode duration and stop needs.

Fix reversible bugs and verify them offline within existing authorization. If automatic review denies an action, preserve evidence, complete unaffected work, and report the denied action, reason, and missing authorization. Keep data and weight exports within the authorized destinations.
