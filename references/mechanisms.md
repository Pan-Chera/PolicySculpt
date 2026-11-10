# Mechanisms, computational paths, and edits

## Four distinct claims

1. Locate tensors, indices, and scales in source to establish computational semantics.
2. Produce a predicted activation or output on the same state to establish a computational effect.
3. Repair a specific failure in a new physical execution to establish a case outcome.
4. Use independent initial states and controls to assess generalization.

Each claim needs its own evidence. To establish a physical mechanism, use interventions that distinguish directions, components, or time windows. Explicit phase embeddings are often easier to trace than shared deep units. Record externally supplied formulas, indices, or directions.

## Diagnose the computational path

Compare multiple failures and nearby successes. Trace inputs, features, activations, gates, priors and feedback, integrators and filters, decoding, and safety layers. Check whether the path receives the necessary state, is active at the failure, affects other skills, and leaves residual state after exit.

A target translation can produce a different object displacement. Rotation can move a TCP, grasp point, handle, or base point through a lever arm. Treat learned motor weights and a measured Jacobian as separate quantities. Align pre-action, post-action, physical substeps, and frames before interpreting direction. If large angles also occur in successes, seek stronger evidence before using angle as a selector.

## Edit families

Choose matching, internal intervention, folding, and retention checks by task. [Intervention workflow](intervention-workflow.md) · [Tool guide](open-source-tools.md)

- Edit learned action, pose, or phase coefficients analytically.
- Project or subtract a supported output direction using a low-rank change.
- Repurpose existing incoming and outgoing connections to create a local cross-layer route. Verify expressivity, exit, and sharing effects, including the unit's former contribution.
- Edit learned gates, durations, or feedback. Report changes to fixed thresholds, time limits, or controllers separately.

Keep architecture changes, new conditional code, and data fitting separate from weight-only comparisons. The car experiment reconstructed a cross-layer route. The math experiments included directional subtraction, head ablation, and edits to selected columns. For cross-layer routing, analyze old contributions, compensation, required inputs, nonlinear expressivity, and recovery on exit. Low activation alone gives little evidence of spare capacity. For recurrent networks, verify the actual gate order, matrix layout, and hidden-state timing.

## Time-envelope example

Apply this only when the actual policy uses `p(u) = b + Σ w_k T_k(2u−1)(1−u)` with `u ∈ [0, 1]`.

- `Δw0 = d/2, Δw1 = d/2` gives `u(1−u)d` and vanishes at the endpoint.
- `Δb = d, Δw0 = −d` gives `ud`.
- `Δb = d, Δw0 = −1.5d, Δw1 = −0.5d` gives `u²d` and remains active at the endpoint.

Terms with the `(1−u)` factor vanish at `u = 1` regardless of how many are edited. For failures after that point, look for an expressible terminal effect. Verify the actual basis, context flattening, dtype, and units before choosing phase or joint indices. After the prior term vanishes, integrated state or velocity can still affect the resulting action.

## Magnitude, controls, and falsification

Measure parameter changes alongside action and gripper differences, saturation, the effect window, and subsequent state. Large edits can be clipped. Small edits can switch a contact mode. Choose magnitude by the measured computation and task constraints.

Try a few meaningful magnitudes rather than repeatedly increasing an imperceptible edit. Screen analytic candidates directly when appropriate. If the missing evidence is the physical response, use a bounded human-run probe when authorized. Evaluate retention against the task's benefit and cost criteria. [Staged verification](research-efficiency.md)

Choose applicable sign, strength, equal-norm random, time-window, and component controls in advance. Count in-memory candidates and all random-direction trials. If both signs help, retain plausible explanations such as nonspecific perturbation or nonlinear contact changes.

When corrective action labels are missing, use predeclared intermediate predictions, output constraints, and original-success checks to rank hypotheses. Verified successful demonstrations or geometric or kinematic targets can also help when their conditions and errors are known. Physical effects need separate evidence. If these measures cannot distinguish candidates, retain competing hypotheses or report NO_GO.

Record baseline and candidate file and tensor hashes, edit keys, indices and values or deterministic generation code, dtype, parameter eligibility, buffer and source changes, hypotheses and falsifiers, scope, entry, exit and residual state, groups, and outside help. Verify the actual diff at deployment precision.

## Alignment pitfalls

A value of `u = 1` can correspond to many timestamps. Matching the first nearest value can align a late action with the earliest endpoint. Cross-trajectory differences include different states and contact outcomes.

For direct computational effects, match observations, prior memory and history, phase and age, and target. Report cross-trajectory changes separately. Keep physical limits for force, collisions, and devices separate from local output retention. If a preregistered measurement is flawed, preserve the original stopping result and freeze a corrected protocol before a new experiment.
