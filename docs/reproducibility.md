# Case evidence and reproducibility

The repository includes the skill, example tools, and the media shown in the README. The exact research checkpoints, frozen cohorts, raw logs, and task-specific runners for the four showcased cases are held in the experiment archives. A GitHub checkout can run the [standalone example](first-edit.md) and bundled tests. Exact replay of the showcased runs requires the corresponding archive and runtime.

| Case | Reported comparison | What an exact replay needs |
| --- | --- | --- |
| Goblet handoff | Independent paired simulation on 28 fresh starts. Stable handoff and placement rose from 18/28 to 25/28, with 8 repairs and 1 regression. The confirmation cohort emphasized one transfer direction. | Frozen baseline and candidate, full reset state and transfer direction for each sample, native simulator and assets, and the original event and success checks. |
| Bimanual bowl stacking | On the 100 fixed regression starts (seeds 14000–14099), successful three-bowl stacks rose from 95/100 to 98/100. The selected candidate repaired 4 original failures, retained 94 original successes, and regressed on seed 14025. Seed 14057 still failed. The set was used to compare candidates. | The frozen MLP and its vision locator, edited checkpoint, matched seed list, native MuJoCo runtime and robot assets, and the original final stack and return-to-home checks. |
| Connect Four | Mirror disagreements went from 7,928 to 0 on 24,000 held-out endgames. A disagreement measures inconsistent actions across a board and its reflection. | Frozen board list and mirror convention, both policy checkpoints, deterministic action selection, and the archived evaluator version. |
| Exposure | Mean highlight clipping went from 12.90% to 2.50% on 65 eligible held-out FiveK photos. | Exact FiveK fold and eligible photo list, pinned Exposure source and checkpoint, native preprocessing, matched dropout masks, and the archived clipping metric. FiveK data is governed by its own terms. |

The robot results are simulation measurements. Demonstration clips illustrate particular runs. The [method notes](../references/method-and-lessons.md) describe the Goblet direction imbalance and earlier confirmation batches.

When reproducing an experiment, record checkpoint and evaluator hashes, data identity, and the runtime. Keep development cases separate from independent confirmation. Freeze the selected candidate and scoring rule before confirmation. For paired evaluation, report repairs and regressions alongside the totals and keep technical invalidations separate from policy failures. See the [evaluation guide](../references/research-efficiency.md) and [robot workflow](../references/robot-workflow.md).
