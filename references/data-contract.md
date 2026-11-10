# Log contract and offline execution

## Minimum handoff

Obtain a trusted checkpoint, frozen inference code, input encoding and normalization, the actual action interface, and log provenance. Look for existing assets before requesting missing items. Prefer safe weight formats and restricted loading. Unrestricted pickle loading can execute code, so reserve it for verified trusted sources.

`assets/dataset.template.json` is an example of the log contract. A template with `template_only=true` is ineligible. Fill in actual metadata, file hashes, and logs before setting it to false.

| Metadata | Purpose |
|---|---|
| Model, runtime, controller, and calibration identities | Separate model issues from code, device, and calibration changes |
| Observation and action schemas, units, frames, clocks, and frequencies | Distinguish radians from degrees, absolute from relative commands, TCP from grasp frames, and timing |
| Episode, session, and physical case group | Prevent leakage across repeated resets, nearby segments, and collection batches |
| Log hash and `real_robot`, `simulation`, or `synthetic` source | Bind the evidence and identify the execution setting |
| Observation allowlist, required fields, missing values, and estimation uncertainty | Make absent observations explicit |
| Proposed, accepted, and executed commands with provenance, plus sensor readback | Distinguish model output from actual execution |
| Prior-state provenance and timing, reset, RNG, and action chunks | Reconstruct recurrent, stochastic, and chunked policy computations |
| Outcome, termination, timeout, and annotation confidence | Attribute observed outcomes to the behavior policy |

Keep confidence and calibration for estimated poses. If required object poses are missing, narrow the analysis or decline full replay. Use deployment-time tactile inputs when available and keep simulator contact ground truth in the audit. Record send, acceptance, application, and observation times separately. A driver acknowledgment confirms receipt, while tracking accuracy requires readback.

## JSONL rows

Each row contains `episode_id`, `tick`, monotonic `timestamp_s`, `observation`, and `commands`, with optional `internal_state` and `audit`. Restrict observations to allowlisted fields. Outcomes, teacher outputs, and contact evidence can be recorded under `audit` and kept outside policy inputs. Use `null` for missing commands and record why.

Record `proposed`, `accepted`, and `executed` as `{value, provenance}` and keep sensor readback separate. Align each observation with the command issued that cycle. The next observation reflects the earlier command. Preserve gaps after filtering. For action chunks, record generation time, chunk index, caching and aggregation, and the sub-action actually sent. Count policy calls at generation time.

## Two analysis modes

**`logged_state_one_step`.** Compare both policies on the same `o_t, h_pre, actual_history, phase/age`. Convert post-state to pre-state using the original implementation or reconstruct it from the prefix. Injecting baseline state into the candidate is a local computational intervention. Report whether that state might be unreachable under the edited policy.

**`fixed_observation_sequence`.** Let each model update its own hidden state and phase from reset or a verified reconstruction, with physical observations held fixed. If the original phase schedule is frozen to isolate an action layer, report this conditioning. State whether previous actions come from executed commands or candidate outputs. Candidate actions paired with fixed observations usually serve a diagnostic purpose.

Control and record RNG and configuration for stochastic policies, diffusion sampling, and dropout. If recovery is unavailable, report variation across repeated forwards. Keep deployment dtype, decoding, and clipping consistent. Training mode and BatchNorm updates can confound comparisons. Count any extra forwards performed by diagnostic hooks.

Use explicit noise, verified random streams, or bounded repeats. Separate effects under matched perturbations from stability across perturbations. Scope claims to the reconstruction actually verified, and audit sharing, views, wrappers, and caches relevant to the target computation. [Randomness protocol](adapter-and-integrity.md#rng-protocol) · [Adapter acceptance](adapter-and-integrity.md)

## Implement the model adapter

Reuse the original inference encoding and decoding. For tasks that require state, expose `reset(metadata)`, `snapshot_pre_state()`, and `step(observation, state, rng) -> (raw_output, decoded_command, next_state, diagnostics)`. Include actual dependencies such as hidden state, filters and integrators, previous actions, phases, and chunk caches. Isolate runtime instances and deep-copy mutable state to keep baseline and candidate caches separate.

Reconstruct the baseline in a process without the environment or hardware driver. Record per-frame errors and the first point outside tolerance. Then add the candidate. In single-step mode inject the same prior state into both isolated instances. In sequence mode reset each model and document its history semantics. Report separately which observations, internal states, and behavior schedules are fixed or evolving. Export only allowlisted inputs. Audit data is available to reporting and kept outside the inference function. The allowlist covers top-level fields. The adapter checks nested structure, shapes, dtypes, and ranges against the actual schema.

Report the per-frame differences, phase, saturation, and exit diagnostics, episode aggregates, and worst windows needed for the current question. Preserve the distinction between pre-action and post-action values. Before promoting a candidate, record the actual diff at deployment precision, relevant baseline reproduction, the effect window, and applicable retention results. If inference depends on a robot driver, expose a driver-free entry point. This skill ships a contract checker. Implement the model adapter using its native inference code.

During initial in-memory screening, retain representative windows and diffs. Expand applicable retention checks and save and reload promoted candidates. Reuse baseline forwards when model, runtime, input, state, dtype, and randomness match. Recompute when a relevant dependency changes. [Research efficiency](research-efficiency.md)

The bundled affine helper checks a local transformation for a given `P` and `c`. The adapter still owns model reconstruction, isolated state, and the chosen comparison modes. Reload promoted candidates without hooks and check recurrence and exit behavior when state matters. Skip sequence checks for static networks when they add no evidence. [Affine folding helper](open-source-tools.md#affine-helper)

## Contract check

```bash
python3 scripts/validate_bundle.py --manifest dataset.json --check-logs --evidence evidence.json
```

The script uses the standard library to inspect JSON, JSONL, and hashes. It checks identity formats, timing, the allowlist, finite values, command provenance fields, and split conflicts. `metadata_valid_only` describes metadata conformance. Model behavior and claimed provenance require their own evidence.

Use strict contract checks where full log acceptance is needed. When logs are incomplete, state which comparisons remain valid and continue those local or static analyses. Preserve actual hashes, commands, and prior state and label any unsupported reconstruction.

Keep episodes in the same session or physical case in the same split. Adding labels to exposed episodes leaves them exposed. If groups leave too little data, report development or regression evidence. Hardware adaptation requires real logs to verify timing, the command chain, calibration, and missing sensor observations.
