# Trace the mechanism, intervene, fold the edit, and check retention

Use these modules when a direct coefficient edit is insufficient, when the task requires tracing MLPs, recurrence, or shared paths, or when a temporary intervention works and needs to be saved as weights. Internal interventions can establish computational causality under tested conditions. Physical contact mechanisms need their own evidence.

Choose the least costly module that answers the question. An analytically understood change can go straight to an in-memory variant. Activation directions, bidirectional replacement, and a complete physical explanation are optional where they add no decision-relevant evidence. [Research efficiency](research-efficiency.md)

## 1. Freeze a testable question

Use `assets/experiment.template.json` to identify tensor or module location, input axes and units, state provenance, expected intermediate quantity, first expected change, predicted sign, magnitude or time window, falsifier, alternative explanations, and any external help. Express the interpretation as a forward-pass prediction that can be checked.

For success and failure activation contrasts, verify relevant baseline reconstruction and choose comparable development windows. Group by episode and session. Match relevant task phase, object estimate, joints and velocities, gripper or observed contact, and history semantics. Fix matching rules before examining the contrast and report overlap and unmatched cases. With poor overlap, use a same-state intervention or analytic analysis. Keep outcome labels in diagnostic grouping. Use verified action references when correct action labels are required.

## 2. Find the first computational divergence

Sample only inputs, outputs, gates, hidden state, integrators, prior and feedback terms, and pre- and post-decoding or clipping quantities relevant to the hypothesis. Record coordinates, shapes, call counts, and hook locations. For a repeatedly invoked GRU, identify the episode, tick, layer, and pre- or post-call position. Verify sources, initialization, perception, and action interfaces before attributing a bottleneck to the network.

Use mean differences, SVD or PCA, covariance, saturation rates, and unit contributions to locate paths. Check causality separately. State the weighting or sampling rule so long episodes do not dominate by frame count alone. Control for phase, object location, and similar confounders.

## 3. Intervene at a fixed state

Match prior state and random inputs between the baseline and the intervention. Select controls that distinguish the hypothesis.

- Replace selected activations in a failure window with those from a matched success. When useful, make the reverse replacement too. Preserve unrelated observations and action history.
- Zero a component, replace it with a mean, subtract a direction, or isolate the path. Choose sign, magnitude, equal-norm random direction, component, or time-window controls as warranted.
- Keep state fixed for local steps. Before promotion, check relevant sequence dynamics and residual state when the edit can affect them. Label injections whose activations may be unreachable for the current input.

Check the specified intermediate quantity, earliest divergence, effect window, and computation on original successes. Attribute logged outcomes to the behavior policy. Scope necessity and sufficiency to the tested intervention and computation. Redundant paths may survive a failed ablation or replacement. After removing a hook, verify that outputs return to baseline and that caches, parameters, and buffers retain their original state.

## 4. Write a direction into selected weights

In column-vector convention, a linear layer is `y = W h + b` with outputs in rows and inputs in columns. Transpose explicitly when the framework uses the opposite layout. For a unit output direction `u` with `uᵀu = 1`, edit selected input columns `J` with

`W[:, J] ← W[:, J] − α u (uᵀ W[:, J])`.

This reduces the contribution of those columns along `u`. Other paths, biases, and downstream nonlinearities can still carry related information. Choose columns, direction, strength, layers, and normalization from development evidence. Record selection attempts and derive model-specific indices.

Use the same confounding and retention checks for statistical directions such as LEACE. Label the estimate `statistical_intervention`. Check nonlinear downstream effects and physical outcomes separately.

## 5. Fold the intervention into the original weights

A fixed affine transformation `z′ = P z + c` can be folded into an adjacent linear layer in either of these cases.

| Intervention location | Equivalent parameters using the original `W` and `b` |
|---|---|
| Linear layer input `h` | `W′ = W P` and `b′ = b + W c` |
| Linear layer output `y` | `W′ = P W` and `b′ = P b + c` |

If a bias is required but absent, record the architectural limitation or report adding a bias as a separate change. Fold only across compatible affine operations. ReLU, tanh, LayerNorm, clipping, input-dependent gates, and quantization generally require separate treatment. For a shared activation, folding into one consumer affects that branch. State- or time-dependent `P` or `c` may exceed what a fixed edit in the original architecture can express. Report that limit or test an extension separately.

The helper handles these fixed, real-valued, single-layer transforms. Direction discovery, label fitting, GRU gates, and full checkpoints remain outside its scope. Screen numerical equivalence on development features or in-memory variants. For promoted candidates, remove hooks, write the weights, save and reload in the actual runtime and dtype, and check relevant modes. Reuse valid export-path evidence across variants. [Affine folding helper](open-source-tools.md#affine-helper)

## 6. Freeze a retention panel

Before screening, choose a small panel, scales, tolerances, and promotion or stopping rules appropriate to the decision. Expand it only when a candidate advances.

- Include previously successful episodes with coverage before, during, and after the intervention window and later in the task.
- Include relevant untargeted phases, heads, or control modes, and action boundaries or saturation regions.
- Report maximum and quantile action changes, joints and grippers separately, timing, applicable residual hidden state, and nonfinite values. Keep physical units explicit.
- Report target gains alongside retention costs. Set tolerances for the worst windows before inspecting candidate results.

Estimate directions and covariances on development data. A tuning panel becomes exposed and can serve regression checks. Reserve separate confirmation by session and case, and use new human-run hardware interactions where required. Report coverage gaps when there are too few independent groups.

Measure allowable trade-offs by task. Specify which paths and time windows may change, how much degradation is acceptable, and any separate safety constraints. Scale independent confirmation to the intended claim.

## 7. Choose positive and negative controls

When setting up tools or changing a critical loader or edit step, choose the least costly applicable check and reuse valid evidence.

1. **Known synthetic intervention.** `scripts/test_affine_patch.py` compares a temporary transform with written weights in a deterministic small network and checks bias, transpose, and rounding cases. The result covers the local algebra helper.
2. **Apply a published patch.** Bind the author's baseline, patch revision, and dtype and verify hashes. Budget any simulation replay separately from hardware. Use the rounded model specified for the car patch. [Car artifact manifest](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md) · [Math patch protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
3. **Discover an unknown repair.** Give the agent an independent task without the patch or hints. Record help and total attempts separately from known patch application.
4. **Validate the user's policy.** Freeze the candidate and have a human run new hardware episodes from a separate plan when physical verification is required.

Use NO_OP, hook removal, save and reload, or a reverse direction as appropriate to isolate technical errors. Attribute prior answers or supplied directions explicitly. Synthetic and published checks establish their own scoped evidence.
