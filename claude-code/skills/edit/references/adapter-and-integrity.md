# Adapter acceptance, edit integrity, and randomness

Choose the scope that supports the current analysis and claim. A0–A3 label the evidence already available. Check shared parameters, caches, and randomness only where they affect the current computation, and reuse evidence while its dependencies remain valid.

## Adapter evidence states

Bind the evidence to the baseline, runtime, encoding and normalization, schema, dtype, device, and analysis scope. Establish a new result when any relevant identity changes.

| State | Minimum evidence | Supported claim | Further evidence needed for |
|---|---|---|---|
| A0 Structure and parameters are readable | Weights or source, computational path, and known gaps | Hypotheses, analytic predictions, and adaptation guidance | Reproduced baseline behavior or improved candidate actions |
| A1 Fixed inputs are comparable | Baseline outputs reproduced within a stated tolerance on relevant inputs, prior state, and random inputs as needed. Record error and scope. | Computational differences under the tested conditions | Free-running state, full trajectories, or physical gains |
| A2 Required state and randomness are testable | Relevant resets, hidden state, history, action chunks, or phases can be reconstructed. Record the random protocol and evidence, or mark it inapplicable. | Sequence computations or output distributions within scope | Physical consequences of candidate actions on old logs |
| A3 Candidate checkpoint is verified | Save and reload without hooks, audit the edit, check applicable state and retention | The saved candidate implements the tested computational effect | Untested states, independent success rate, or hardware safety and deployment |

Mark state or randomness as inapplicable where appropriate. An A1 result covers the fixed conditions tested. For stochastic reproduction at the distribution level, report the distribution and its scope. Local comparisons can proceed with reconstructed state even when sequence dynamics have yet to be checked.

Record the evidence ID, identities, actual inputs or windows, tolerance and worst error, state and random semantics, evidence references, and gaps. Repeat affected checks when the candidate, runtime, or reconstruction logic changes. Physical and independent performance claims require separate execution evidence.

Human-provided closed-loop results can support the corresponding case description. The adapter evidence level follows the adapter checks already performed. Claims about weight attribution, mechanism, and generalization each require their own evidence.

## Minimum edit integrity checks

1. **Parameter identity and shared storage.** Check whether keys refer to the same `Parameter`, repeated module references, or different views of shared storage. List the targets, alias groups, shape, stride, offset, and actual consumers. Determine which elements overlap and which computations use them.
2. **Effective weights and wrappers.** Check normalization, parametrization, quantization or packing, low-rank components, shared embeddings or heads, and compiled copies. State whether the edit targets the original or effective weight and when it is recomputed. Property overrides can disappear on the next forward pass.
3. **Caches and derived state.** Identify affected KV caches, RNN hidden states, filters and integrators, action chunks, precomputed embeddings and projections, and compiled or parametrized state. Rebuild or invalidate affected state. For a single-step intervention using the baseline state, report the conditioning and reachability limits.
4. **Save and reload.** Check values, sharing and alias topology, wrapper settings, and behavior after loading into a fresh instance. A change in file bytes alone gives no evidence of a tensor value change. Independent serialization or per-key clones can change sharing even when values match.

For PyTorch, check the installed version first. `named_parameters(remove_duplicate=False)` and `named_buffers(remove_duplicate=False)` expose duplicate names when supported. Freeze a genuinely isolated baseline because `state_dict()` returns references. Inspect storage, stride, and offset where needed. Compare named topology across processes. [PyTorch Module API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html) · [Serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)

Use `is_parametrized` and the relevant caching context to check effective and original weights before editing a parametrized module. These API notes guide integration. The adapter performs the actual audit. [PyTorch parametrization source](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrize.py)

Report the number of unique parameter values changed, affected state keys, and shared consumers. If sharing, wrappers, and caches are absent, record the source or object inspection that establishes it. List changes to sharing, added biases, or wrappers as separate modifications.

<a id="rng-protocol"></a>
## Randomness protocol

Separate the effect under the same random input from stability across different random inputs. The former is a controlled computational comparison. The latter measures repeatability across sampled perturbations.

- Identify random sources that affect the target path, such as Python, NumPy, and Torch generators, action sampling, diffusion noise, workers, dropout, and external sensor perturbations. Record version, device, call location, and semantics.
- Where possible, inject the same explicit random input into both policies. Record its hash, shape, dtype, and meaning at the episode, tick, sampling step, or component level. For sampled actions, share the underlying uniform or Gaussian draws and let each policy produce its own action.
- If injection is unavailable, restore separate generator states and check corresponding call counts, order, and purpose. Diverging branches can change the draw sequence even with the same seed. Check shape and semantics before reusing noise.
- For uncontrolled environment disturbances or nondeterministic kernels, use a bounded number of repeats and report uncertainty and whether pairing was exact.
- Check pairing on a short segment before selecting any independent draws needed for stability. Count all repeated forwards. Include noise or generator state and execution mode in cache keys.

Identical seeds can yield different results across PyTorch versions, platforms, and devices. Scope exact pairing claims to conditions you verified. Distribution-level comparisons can still proceed. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

## Checkable records and tool scope

`scripts/check_experiment.py` checks the plan and optional result record for minimum fields, candidate baseline, development and confirmation groups, budget and stopping rules, observed checkpoint hash, and claim level. Cite evidence files in the record and verify hashes when accessible. The adapter remains responsible for reconstruction and for checking sharing, wrappers, and caches.

```bash
python3 scripts/check_experiment.py --plan plan.json
python3 scripts/check_experiment.py --plan plan.json --record record.json --check-artifacts
python3 scripts/check_experiment.py --collection-plan collection-plan.json
```

Use the experiment template to record `task_mode`, `stage_purpose`, the question, budget, and review or stopping rule. Start the record from `assets/experiment-record.template.json` and omit or mark fields that do not apply. Resolve relative `artifacts` paths against the record file. Refer to evidence by artifact ID. If remote files are inaccessible locally, omit the artifact check and record the verification gap. `record_consistency_only` reports structural consistency. Scientific and safety validation require their own evidence.

In `data_usage`, record `group_id`, available session and case identifiers, `used_for`, and `exposed_before_freeze`. Development data can serve both construction and selection. Separate confirmation groups from construction, selection, diagnosis, and any pre-freeze exposure. Claim kinds include `structural_hypothesis`, `fixed_input_effect`, `sampled_output_description`, `sequence_effect`, `saved_candidate_effect`, `closed_loop_description`, and `independent_performance`. For stochastic computations, specify `same_perturbation` or `across_perturbations`. Describe an uncontrolled single sample as an observed output.

Record both the frozen and observed candidate hashes for confirmation. If a fixed-endpoint run stops early, report the observed interactions under `closed_loop_description` and mark confirmation incomplete. Report actual spending and any genuine budget revision. The collection-plan check covers roles, passive or motion modes, bounded limits, and stopping or self-test descriptions. Hardware behavior and safety require separate verification.
