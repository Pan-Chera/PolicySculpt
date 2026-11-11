# Research efficiency and staged verification

Use this guide to decide which check is worth running next. Minimize the cost of evidence that can change the decision while preserving reproducibility, accounting, and applicable safety conditions.

## Choose one main question per round

Choose a task mode by the user's objective. `feasibility` reports fit, gaps, and whether to proceed. `single_hypothesis` tests one main hypothesis and delivers the necessary controls, one preferred candidate or a justified NO_GO, and retention results. `full_research` can expand to multiple hypotheses and frozen confirmation as evidence warrants. Any mode can use the relevant stages below. Apply read-only consistency checks to existing records.

| Stage | Question | Useful minimum | When to stop or expand |
|---|---|---|---|
| Offline exploration | Can the path produce the predicted computational effect? | A representative window or frame, any required prefix, and a relevant success | Change the hypothesis if falsified. Expand when the effect is clear. |
| Candidate screening | Is more expensive verification worthwhile? | A small failure-focused development set and compact retention panel | Advance if the preset criteria are met. Otherwise report a negative or inconclusive result. |
| Bounded closed-loop probe | What physical response is missing from offline evidence? | Human-run new episodes with limited conditions and controls | Update or reject the hypothesis. Use the observations as development evidence. |
| Formal independent confirmation | What are the frozen candidate's benefits and costs? | Design to the target effect, precision, risk, and cost | Use a fixed endpoint or a valid prespecified sequential rule. |

Choose sample sizes and controls for the question. If a small sample cannot estimate an overall effect, report the interval or uncertainty. Inconclusive evidence can complete a round. Put a larger confirmation on the critical path when the result can change adoption or the next decision.

## Limit the marginal cost of each batch

Record the question, predicted observation, decision it can change, cheapest applicable check, cost ceiling, review point, and stop or promotion criteria. Aim for a useful first result before building an entire toolchain. Within existing authorization, log routine adjustments and proceed.

At the review point, compare the value of new information with its cost. Shift to a more informative mechanism when more samples would only narrow the interval around an unpromising candidate. Treat budgets as ceilings. Freeze a candidate before opening a full independent test.

Exploration can change direction or stop after seeing results. Preserve trials, selection decisions, and costs as development evidence. Use fixed candidates and criteria for formal confirmation. Sequential testing requires a prespecified design and matching inference. If a fixed-endpoint run stops for priority, technical, or safety reasons, mark it incomplete and report the sample actually obtained.

<a id="screening-sample-selection"></a>
## Screening samples and stopping

- Define the target failure, earliest relevant observable divergence, and successful cases for retention. Start with logs from known failing initial states and comparable successes. If a closed-loop screen is authorized and those initial states can be restored reliably, include them in paired baseline and candidate runs. Nearby perturbations can make the baseline succeed, so verify coverage from the observed baseline outcome and divergence.
- If the baseline succeeds everywhere, the panel measures retention under the tested conditions. If observed failures diverge in a different way, record the coverage gap. Change examples or revise the hypothesis only when another bounded batch can resolve the gap and change a decision. When coverage is adequate and the prediction fails or the candidate misses promotion criteria, stop the route and retain results and costs.
- Group known failing initial states, repeated resets, slices of one trajectory, and nearby perturbations generated for it under the original physical case. Use them for development or regression. Reserve unseen initial states for independent confirmation. Separate complete episodes, sessions, and physical cases across the split. Freeze the candidate and inclusion rules before running. Keep valid pairs with both baseline successes and failures.

## Choose informative edit magnitudes

Choose magnitude by observable computation, the effect window around contact, and actual constraints. Check whether rounding, inactive gates, timing, or clipping hides a change. Once the path is verified, try a few functionally distinct strengths. Increase strength when the effect should become observable within bounds. If side effects dominate, investigate selectivity or change paths.

When the missing evidence is contact response, a bounded human-run probe can be more informative than repeatedly inspecting one log. Preserve the original interlocks, model provenance, hard constraints, and authorization.

Measure retention by task benefit, loss of original capabilities, and uncertainty. Predetermine tolerances that allow the intended correction. Use global action differences, zero-regression criteria, or activation preservation only when justified by the task. Keep collision, force, and device limits separate.

## Reuse evidence and check only what changes

- A stateless policy may only need a single-step comparison. Before promoting a stateful edit, check relevant recurrence and residual state at exit. Skip hooks and SVD when an analytic construction answers the question.
- Use the necessary windows and a small retention panel. Cache verified baseline outputs and activations by model, runtime, input, state reconstruction, dtype, RNG, and mode. Invalidate on a relevant change. Recurrent windows require an authentic prior state or prefix.
- Recompute affected prefixes and hidden state after an edit. Reuse valid baseline caches. Save, reload, audit the full diff, and expand retention checks for promoted candidates. Keep lightweight diffs and accounting for exploratory variants.
- Recheck load and save behavior when a dependency changes or evidence expires. Use a small synthetic or NO_OP check when a positive control is needed. Public car and language model reproductions are optional research.
- Batch reservation and accounting when useful. Record actual use, failed attempts, incomplete prefixes, and reservations. Count real model and component forwards. Keep parsing and reuse of existing outputs separate.
- Continue valid local or static analysis when logs are incomplete. State what remains incomparable. Keep future ground truth outside the policy. Use strict log validation where the full contract applies.
- Prefer native hooks or matrix operations for small tasks. Reuse available tools. Run independent lightweight work in parallel only within authorized resources. Isolate locks, hardware control, confirmation data, and mutable state.

<a id="sample-resume"></a>
## Resume a frozen evaluation by sample

For interruptible evaluations, the standard-library helper `scripts/sample_resume.py` stores complete paired receipts. It follows Inspect AI's use of stable sample IDs to retain completed results across retries. [Inspect AI evaluation logs and retries](https://inspect.aisi.org.uk/eval-logs.html#eval-retries)

- Bind the complete checkpoint identities for the baseline and candidate separately, including all model components actually loaded. Bind evaluator identity to execution and scoring code, settings, runtime, and any random protocol that affects outcomes. Give each stable sample ID a digest of its native input, exact initial state, and required random inputs or stream. Use byte or file hashes for native state.
- Resume the same frozen plan in a different order when samples are independent of execution order. Reuse only committed pairs with two technically valid results, matching identities, and an intact receipt hash. Valid policy failures are results. Keep attempt records for invalid or interrupted pairs and rerun them under the original rules. A changed identity or sample set requires a separate run directory.
- The caller verifies actual identities and result validity. Use a single coordinator to schedule and commit pairs. The task's existing runner owns environment execution, scoring, and safe stopping.

```python
from sample_resume import SampleJournal

# identities maps baseline, candidate, and evaluator to SHA-256 digests.
# sample_hashes maps stable IDs to hashes of the actual inputs.
journal = SampleJournal(run_dir, identities, sample_hashes)
done = journal.completed()
for sample_id in sample_hashes:
    if sample_id in done:
        continue
    baseline_result, candidate_result = evaluate_valid_pair(sample_id)
    journal.commit(sample_id, baseline_result, candidate_result)
```

## Responsive execution and stopping

Humans handle hardware collection and confirmation by default. Verify a cooperative stop entry point such as `stop_after_current_episode` or `stop_after_current_pair` in collectors and simulation schedulers. Check it before starting the next episode or pair. Preserve completed work and list runs that never started. Make run order and pairing boundaries explicit. Leave motion stops to the verified safe mechanism.

Submit long work in reviewable batches. Match monitoring frequency to episode duration and stop requirements. During a frozen run, keep unrelated offline work separate and preserve the executing policy, evaluator, physics, and resource allocations.

When the user changes priorities, identify the actual admission boundary and verify that the stop or redirect has taken effect. If a request is queued and there is no safe interruption interface, state the active status and the actual manual stop operation.

## Deliver a useful result from one round

Report the question answered, data and modes used, findings, largest unresolved issue, actual cost, and next decision. Record omitted checks and why. Reference valid cached evidence by path and hash. A justified NO_GO or bounded exploration with insufficient evidence is a valid result.
