# Optional open-source tools and integration

Sources reviewed on November 5, 2026. Choose tools for offline repair of an existing policy. Scope installation, online calls, training, and device connections separately. Verify third-party integration with the actual model at use time. The upstream documentation describes upstream capabilities.

## Choose a starting point

| Need | Tool | Deliverable |
|---|---|---|
| Explore tensors, budget probes, inspect diffs, and roll back | weightspace or an existing equivalent | Frozen baseline, probe code, call ledger, and patch |
| Activation replacement, ablation, or tracing RNN and MLP paths | pyvene or simple native PyTorch hooks | Intervention point, state contract, restoration check, and window-level results |
| Estimate a linear concept direction and retention transform | LEACE or explicit projection | Development statistics, `P` and `c`, coverage, confounding, and retention |
| Fold a fixed affine intervention into a linear layer | Bundled `affine_patch.py` | Single-layer NPZ candidate and numerical check, followed by native adapter verification |
| Repair a region with explicit input and output specifications | APRNN when applicable | Specification, solve, and original-network verification under `constraint_repair` |
| Study low-rank edits with retention objectives | EMMET, ROME, or MEMIT | Method applicability and an audit of gradient use |
| Human-run episodes and multimodal recording | Native runner first, LeRobot if useful | Operator package, local logs, and extended state records |

Record the chosen repository, actual commit or package version, dependencies, device and dtype, license, and local test status. Note unavailable versions. Isolate the analysis environment and preserve the field controller environment. Install only dependencies needed for the selected task.

Common packages include `weightspace` with a backend extra such as `torch`, `pyvene`, and `concept-erasure`. The affine helper needs `numpy`. Check compatibility, pin dependencies, and record installed versions with `importlib.metadata.version(...)` plus the source commit where available. Reuse an existing analysis environment when suitable.

## weightspace for offline probes

The third-party harness provides tensor slicing and SVD, edit tracing, checkpoints, rollback, and budgeted probes. Its code sandbox is an execution convenience. Keep security and hardware controls outside it. [Project and API](https://github.com/leofmarciano/weightspace) · [Upstream README](https://raw.githubusercontent.com/leofmarciano/weightspace/main/README.md)

1. Install the selected version and required backend extras in isolation. Build a driver-free inference model from a trusted local checkpoint and verify the baseline hash.
2. `fix` mutates the model in place. Keep a frozen baseline and an independent editable copy. Isolate buffers and caches and write outputs outside deployment directories.
3. Expose only frozen development inputs through a probe. Reconstruct prior state and call the offline adapter. Return computational errors, hypothesis diagnostics, retention observations, and actual counts. Keep drivers, environment stepping, arbitrary files, and confirmation data outside probe access.
4. Check the actual frame or sequence budget inside the probe and count failed calls. `forward_budget` counts upstream probe units. Keep a separate ledger for frames, components, and warmups when these differ. Return an explicit stop at the budget boundary.
5. If an external LLM is needed, use the specified model and reasoning effort and verify authorization for sending weights or log summaries to that provider. Use already authorized tools when export permission is absent.
6. Run a bounded edit and save the transcript, stop reason, diff, patch, unsuccessful variants, and ledger. Check for off-probe forwards or fitting. Apply a frozen patch to a fresh baseline copy, remove hooks, save and reload, and check actual dtype before any human-run confirmation.

This example shows the integration points. The task supplies the adapter, inputs, ledger, and editor backend.

```python
import copy
import weightspace as wsp

editable = copy.deepcopy(base).eval()

@wsp.probe
def inspect_development():
    # Read frozen development inputs and account for each call inside the adapter.
    return adapter.inspect_fixed_inputs(editable, development_ids, frame_ledger)

result = wsp.fix(
    editable, hypothesis_text, llm=editor_backend,
    context=verified_architecture_and_units,
    probes=[inspect_development], forward_budget=probe_cap,
)
result.save(stage_output_directory)
```

For a new or changed integration, test budgeting, rollback, and outputs with `ScriptedLLM` or a small deterministic control. Reuse valid checks. Use the native human-run collection path for hardware.

## pyvene for static interventions

Use static replacement, collection, or ablation. The library also supports trainable interventions, which are outside the default direct-edit workflow. [Project](https://github.com/stanfordnlp/pyvene) · [Documentation](https://stanfordnlp.github.io/pyvene/) · [Paper](https://arxiv.org/abs/2403.07809)

1. Use a reconstructed inference-only baseline and identify actual modules and tensor axes. Collect a small set of points around the hypothesized path.
2. Verify `IntervenableConfig`, `RepresentationConfig`, `IntervenableModel`, and the installed version's component mapping. Map custom robot components to real modules, hook points, and dimensions. If mapping is excessive, use native static hooks and report that choice. [Configuration source](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/configuration_intervenable_model.py) · [Mapping source](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/intervenable_base.py)
3. Run a NO_OP and collection check for baseline-equivalent output and state. Use inference mode and fixed replacements, zeroing, or directions. Reserve learned rotations and optimizer APIs for separately authorized work.
4. Bind episode, tick, call index, prior state, and source and target activation semantics. Verify the actual modified element with tuple outputs, fused recurrent kernels, or cached chunks. [Intervention workflow](intervention-workflow.md)
5. Remove hooks and caches in `finally` and check baseline recovery and parameter and buffer hashes. Save definitions and results. Fold a promoted edit into weights and verify the reloaded checkpoint separately.

## LEACE for a fixed affine transform from development statistics

LEACE estimates covariance and constructs an affine concept-erasure transform. Evaluate downstream retention and task outcomes separately. [Paper](https://arxiv.org/abs/2306.03819) · [Implementation](https://github.com/EleutherAI/concept-erasure) · [Transform source](https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/leace.py)

1. Freeze development activations `X[N,d]` and defined concept labels `Z[N,k]`. One-hot encode categorical labels. Record episode sampling or weighting, phase and pose matching, confounding, and sample size relative to dimensionality.
2. Use `LeaceEraser.fit(X, Z)` or batch updates with `LeaceFitter.update`. Store hyperparameters, sample IDs, means, `P`, and actual rank or truncation. Classify this as a statistical intervention.
3. Try a temporary intervention on offline activations. Check state norms and retention on separate development slices. Add sham or random directions when useful. Stop if the concept is entangled with required task state or the estimate is unstable.
4. Check the installed source. The column-vector form is `x′ = P x + (I − P)μ`. `eraser.bias` stores the center `μ`, so compute the offset `c` from it.
5. Fold only when the original linear layer meets the affine conditions. Remove hooks and verify the saved candidate in the applicable modes and retention panel. Check recurrent residuals where relevant. Retain unsupported folds as diagnostic interventions. [Folding conditions](intervention-workflow.md)

This conversion assumes prepared development activations. Test PyTorch and LEACE in an isolated environment.

```python
import torch
from concept_erasure import LeaceEraser

with torch.no_grad():
    eraser = LeaceEraser.fit(X_development, Z_development)
    P = eraser.P
    center = eraser.bias
    c = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)
    if center is not None:
        c = center - P @ center
    # Verify the destination is authorized before exporting a float64 NPZ.
    probe_values = eraser(X_diagnostic)
```

<a id="affine-helper"></a>
## Local single-layer affine folding helper

The helper uses NumPy. The standard-library log validators run separately. Export `layer.npz` with `weight[out,in]` and optional `bias[out]`, `transform.npz` with `P[d,d]` and `offset[d]`, and `features.npz` with development inputs `features[N,in]`. Supported layer dtypes are float16, float32, and float64. Use numeric arrays without pickles. Treat a float32 conversion of BF16 or quantized weights as a local experiment.

```bash
python3 scripts/affine_patch.py --layer layer.npz --transform transform.npz \
  --features features.npz --side input --atol 0.00001 --rtol 0.00001 \
  --out candidate_layer.npz --report affine_report.json
python3 scripts/test_affine_patch.py
```

Set tolerances for deployment dtype and accumulation before evaluation. `input` and `output` place the transform on the corresponding side of the layer. The adapter maps the resulting arrays to the native checkpoint and verifies tensor paths, transpose, eligibility, reload, and untargeted state.

The reference check uses controlled float64 accumulation and casts at transform and layer boundaries. `local_affine_check_only` scopes the result to the supplied features. `no_op_after_cast=true` means rounding erased the edit. The tool rejects a required absent bias, nonfinite values, invalid axes, and tolerance failures. Preserve failure reports and existing files. Count two reference evaluations of `N` rows as `2N` layer-frame equivalents, including failed checks.

Import `edit_output_writers(weight, direction, columns, alpha)` for directional subtraction on selected columns. Normalize the direction first. The operation leaves biases and unselected columns outside its scope.

<a id="checkpoint-patch"></a>
## Portable checkpoint patches

Use `scripts/checkpoint_patch.py` to package an existing sparse edit for distribution. The patch binds the complete base file by SHA-256 and checks the complete reconstructed candidate state by a separate SHA-256. One compressed NPZ contains the versioned manifest, changed indices, and replacement bit patterns. The format draws on OpenWSP. [OpenWSP edit format](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md)

```bash
python3 scripts/checkpoint_patch.py create base.npz candidate.npz patch.npz
python3 scripts/checkpoint_patch.py apply base.npz patch.npz reconstructed.npz
```

The tool also accepts flat PyTorch state dicts with `.pt` or `.pth` suffixes. It imports PyTorch for that path and uses `weights_only=True`. Use trusted checkpoints and patches. The scope covers NumPy-compatible NPZ arrays and plain contiguous PyTorch tensors without shared storage. Edits replace floating-point values with matching keys, shapes, and dtypes. Use the native adapter for nested checkpoints or tied weights. Repacking the base requires creating a new patch even if its tensor values match. The tool writes new files and verifies reconstructed tensor bits and state metadata. Verify model behavior separately through the native inference adapter.

## APRNN and EMMET as optional methods

APRNN can apply when input regions and output specifications are explicit and the relevant activations and operators are supported. Verify solver dependencies, derive actual task constraints and retention specifications, run a bounded solve in an isolated copy, and verify the original runtime and out-of-region regression. Report the guarantee, actual error, and coverage. A ReLU proxy's guarantee covers that proxy. Report this route as `constraint_repair` and distinguish network specifications from physical closed-loop behavior. [APRNN project](https://github.com/95616ARG/APRNN) · [Paper](https://arxiv.org/abs/2304.03496)

EMMET offers low-rank and constrained editing ideas for achieving a target while preserving existing mappings. Verify the target source, matrix axes, and retention statistics. The default implementation computes a target using Adam and `loss.backward()`, so account for that gradient work when comparing it. Report adapted analytic constraints and supervised or gradient costs separately. Map any addressing scheme to the actual task state. [Unified editing framework](https://github.com/scalable-model-editing/unified-model-editing) · [Paper](https://arxiv.org/abs/2403.14236) · [Target computation](https://github.com/scalable-model-editing/unified-model-editing/blob/main/emmet/compute_z.py)

Closed-form regression to action labels is `supervised_fit`. Meta-trained adapters, weight generators, world models, and critics are separate research routes.

<a id="lerobot-collection"></a>
## LeRobot for human-run data collection

Prefer the existing field runner and extend its logging when possible. Verify device support and integrate custom bimanual robots with their actual robot or teleoperation interfaces. [Project](https://github.com/huggingface/lerobot) · [Collection documentation](https://huggingface.co/docs/lerobot/il_robots)

1. Develop configuration, recorder extensions, and converters in isolation. Test on synthetic or existing logs. State whether the entry point sends motion. [Human-run collection](human-collection.md)
2. Have the operator verify hardware and calibration and collect a short segment. Provide commands for the pinned version with a local `--dataset.root`, task and episode limits, and `--dataset.push_to_hub=False`. Inspect other upload, telemetry, and background paths for that version. Leave motion and reset under operator control.
3. Record raw model output, accepted and applied commands, prior hidden state, previous actions, chunk cache, model and runtime hashes, synchronization, and interventions. Check whether standard state, action, and video fields cover reconstruction and record missing fields.
4. Preserve original episode IDs, timestamps, bad or interrupted records, and dropped-frame gaps when converting to WSP JSONL and manifests. Identify teleoperation commands by their actual provenance.
5. Verify baseline reconstruction from the short segment before the operator completes the frozen collection plan. Use returned development logs for offline repair. Hand off a separate frozen candidate for new confirmation. Scope uploads, training, and field checkpoint replacement separately.
