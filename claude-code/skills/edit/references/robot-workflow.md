# WSP workflow for robot policies

Start with bounded offline exploration and screening. Freeze a candidate and arrange independent closed-loop confirmation when there is a promising edit and a clear verification question. Log acceptance, diagnosis, editing, and screening can be revisited as evidence changes. Continue valid offline work when exact physical resets are unavailable. Support MLPs, recurrent, hierarchical, and action-chunk policies through their actual interfaces.

Humans collect data and run hardware confirmation by default. When logs are incomplete, define the gap and prepare a collector, configuration, bounded plan, and operator instructions. Test it without hardware. The operator runs it and returns logs for offline analysis. State whether the program can send motion commands. Preserve the original safety system.

Work within the user's assets, budget, and authorization. Proceed with authorized reversible edits and offline checks. Treat candidate construction, hardware execution, and deployment as distinct actions with their own scope.

## Read references as needed

- For verification scope, slow progress, or larger samples, read [Research efficiency](research-efficiency.md).
- For a new policy, read [Log contract and analysis modes](data-contract.md).
- For adapter claims, shared parameters, caches, or randomness, read [Adapter acceptance and edit integrity](adapter-and-integrity.md).
- For new hardware logs, read [Human-run collection](human-collection.md).
- For tensor selection, repair design, or negative results, read [Mechanisms and edits](mechanisms.md).
- For deep or recurrent paths and permanent edits, read [Interventions and retention](intervention-workflow.md).
- For optional editing and collection tools, read [Tool integration](open-source-tools.md).
- For hardware confirmation or execution anomalies, read [Hardware confirmation](real-robot-validation.md) and [Execution verification](real-robot-validation.md#execution-verification).
- For historical methods and evidence, read [Method provenance](method-and-lessons.md).

## Workflow

Paths under `scripts/` and `assets/` are relative to the skill root.

### Choose the stage and shortest evidence path

Separate input, normalization, loading, and interface errors from weight edits. When essential state is unobservable, improve sensing or collect data for that hypothesis and continue unaffected work. For a localized computational defect, test a direct edit. When evidence is limited, use the smallest check that can distinguish plausible causes. If evidence supports an expressivity gap, report it and scope any training or architecture extension separately.

Choose feasibility, single-hypothesis repair, or full research. Feasibility describes fit, gaps, and whether to proceed. A repair tests one hypothesis with the necessary controls and delivers a preferred candidate or justified NO_GO and retention results. Full research can expand to competing hypotheses and frozen confirmation. Start a requested repair directly when the entry checks support it.

State whether the round is exploration, screening, a bounded closed-loop probe, or formal confirmation. Record the decision it can change, cost ceiling, review point, and stop condition. Begin with representative windows. Expand retention and independent confirmation when justified. Stop an unpromising route and retain its evidence.

A bounded human-run probe can answer an unresolved physical question when authorization, interface, and safety conditions are in place. Use independent confirmation for claims about gains. On a priority change, stop at a safe boundary and mark any unfinished fixed-endpoint confirmation as incomplete. [Staged verification](research-efficiency.md)

### 1. Freeze the task and evidence scope

Keep a short working record with the objective, baseline and SHA-256, runtime, encoding, normalization and control interface, budget used, current and next steps, exposed data, and missing evidence. Recheck it against the latest user instruction when resuming.

| Available evidence | Supported next step |
|---|---|
| Weights and inference source | Map parameter roles and state a falsifiable computational hypothesis |
| Insufficient hardware logs | Prepare and test a collector offline for a human operator |
| Logs with timing and command provenance | Reconstruct the baseline and test offline interventions |
| Authorized closed-loop testing and a working interface | Obtain new observations under a frozen protocol |
| Offline gains only | Report the candidate, computation, and remaining validation gap |

### 2. Accept logs and reconstruct the baseline

Build model inputs from the allowlist. Keep raw model outputs, safety-accepted commands, driver-applied commands, and sensor readback separate. Record observations before action, frames, units, timing and chunk boundaries, estimation error, synchronization, resets, and interruptions. Label confidence in failure annotations.

Recompute representative frames or necessary prefixes and verify encoding, normalization, recurrence, prior actual commands, scaling, and clipping. Set tolerances for deployment dtype and hardware precision before comparing. Trace mismatches to source, inputs, or the action chain. Record initialization, model pointer, monitoring, and interface fixes separately.

Use `scripts/validate_bundle.py --manifest <dataset.json> --check-logs` when full metadata, log, and grouping checks are useful. Model reconstruction, deployment behavior, and physical outcomes require separate evidence.

Record the applicable adapter evidence for structure, fixed inputs, state or randomness, and candidate reload and retention. Check aliases, wrappers, and affected caches before editing. Inspect the consumers of changed weights. [Adapter acceptance](adapter-and-integrity.md)

### 3. Choose offline comparison modes

Run modes that answer the current question and report them separately. A stateless policy may only need a single step. For edits involving memory, integration, filtering, phase transitions, or chunks, check recurrence and residual effects before promotion.

- **`logged_state_one_step`.** Compare direct computation on the same observation, prior hidden state, history, and phase. Rebuild unknown state from the necessary prefix or mark that comparison unavailable.
- **`fixed_observation_sequence`.** Keep logged observations fixed and let each policy update its own internal state. Record how previous actions enter the policy. These observations were generated by the behavior policy.

Attribute logged states, rewards, and success to the policy that produced them. Offline return estimates require action support, valid dynamics, or a calibrated estimator. A close match to successful logged actions supports a local retention check.

Distinguish effects under shared perturbations from stability across perturbations. Prefer explicit noise or verified separate random streams. Check branch-dependent draw order. Use bounded repeats and report reduced pairing scope when exact control is unavailable. [Randomness protocol](adapter-and-integrity.md#rng-protocol)

### 4. Turn the mechanism into a weight edit

Map only the path needed for the hypothesis, including tensor keys and shapes, input meaning and frames, nonlinearities, output consumers, accumulated state, entry and exit conditions, and sharing. Separate learned parameters from buffers, config, and fixed code. A learned phase duration can be editable. A fixed control period is part of the execution setup.

Connect an observed problem to a computational path, a prediction made before the edit, and a falsifier. Locate the earliest relevant divergence. Compare failures with comparable successes and check whether the same phenomenon occurs in both.

Choose matrix analysis, activation decomposition, ablation, directional projection, analytic construction, or rewiring of existing cross-layer paths. Screen predictable analytic edits directly. For interventions that use hooks, verify the saved checkpoint implements the effect. [Intervention workflow](intervention-workflow.md)

Label the method `analytic_wsp`, `statistical_intervention`, `constraint_repair`, `supervised_fit`, or `gradient_training`. Default to direct edits without backward passes, autograd, an optimizer, or iterative loss fitting. Closed-form fits to action labels still count as supervised fitting. Scope fitting and new training separately with the relevant authorization.

Choose sparse, low-rank, or block edits by functional effect, scope, memory, and observable constraints. Define acceptable retention costs and promotion criteria before screening. Preserve hard safety boundaries. For conditional effects, verify the existing inputs and network can express the needed behavior. Report architectural limitations or scope extensions separately.

### 5. Construct and screen candidates

Build from an isolated copy of the unique baseline. Keep deterministic edit code or a diff and accounting for in-memory variants. Save and reload promoted candidates. Record eligibility as learned weights, actual tensor shapes and dtypes, the realized diff, other parameter or buffer changes, and construction rationale. Hash saved files. Keep failed trials and check for edits erased by quantization or BF16 rounding.

Use controls appropriate to the hypothesis, such as the baseline, sign, meaningful strengths, an equal-norm random direction, or component isolation. Freeze decision criteria before confirmation. If both signs help, keep alternative explanations open.

Check reconstruction, action and gripper bounds, timing and smoothness, gating and saturation, residual state, original-success windows, and data support. State how noise and latency assumptions were obtained. Report contact and collision acceptance from applicable execution evidence.

Screen representative known failures and related successes. If the baseline succeeds everywhere or diverges for a different reason, report the coverage gap, change examples when informative, or stop. Expand retention only for promoted candidates. Report the worst windows, coverage, and selection history. [Screening samples and stopping](research-efficiency.md#screening-sample-selection)

Record revisions to exploratory hypotheses and controls as development. For permanent candidates, fold into the original architecture, remove hooks, save and reload, and recheck relevant offline modes. Use a minimal positive control to isolate technical risks and reuse still-valid checks. [Intervention workflow](intervention-workflow.md)

Start with `assets/experiment.template.json` and `assets/evidence.template.json` as appropriate. Keep known failing initial states, repeated resets, and derived neighbors under the original case group. Reserve unseen initial conditions, complete episodes, sessions, and physical case groups for confirmation.

### 6. Freeze before closed-loop confirmation

Use the smallest design that answers the question within available authorization, verified interfaces, and original safety controls. Match or stratify initial conditions in advance. Randomize pair order or repeat when needed. [Hardware confirmation](real-robot-validation.md)

By default, hand a human the frozen candidate, versioned collector or evaluator, run sheet, and return format for hardware. Verify the actual model, interventions, and completeness on return. Record independence and whether the operator knew the versions.

Keep the candidate and success criteria frozen during confirmation. Retain normal failures and regressions. Report technical and safety stops separately. Follow the prespecified endpoint. Use offline checks to screen computational effects and closed-loop evidence to establish physical gains.

### 7. Report evidence and cost

Separate parameter computation, offline changes on logs, case-level closed-loop repairs, and gains on independent starts. State confirmed findings, hypotheses, unknowns, and any supplied formulas or directions.

Report baseline and candidate identities and diff, data and groups, hypothesis and falsifiers, modes used, coverage and uncertainty, decision or justified NO_GO, and actual calls and costs. Omit inapplicable fields and choose deliverables for the current stage.

For structured records, `scripts/check_experiment.py --plan <plan.json> --record <record.json>` performs read-only consistency checks. Add `--check-artifacts` when evidence files can be verified locally. The check covers provenance fields, grouping, freeze records, and claim types. Scientific validity and motion authorization are separate. Start from `assets/experiment-record.template.json` where useful.

Account for actual baseline, candidate, and component forwards by frame or sequence, including warmup, failures, reconstruction, and replay. Record reservations separately and preserve prior costs when expanding an authorized budget. Honor denied exports and continue work that does not depend on them.

Start a new adapter with a representative environment-free forward and expand based on evidence. The bundled NumPy helper folds a fixed affine transform into one linear layer and checks numerical equivalence on given features. Choose third-party tools as needed and keep installs, external transfers, and device connections within their authorization.
