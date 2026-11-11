# Method provenance and practical lessons

Sources reviewed on November 5, 2026. This skill adapts the research to robot and other policy tasks. The scope of each case follows its recorded evidence.

## Primary sources

- The authors describe direct weight reading and writing under several tool and feedback settings. [Original article](https://rikiyat.github.io/blog/weight-space-programming/)
- The car patch rewires cross-layer connections to create run-up and exit paths. The 51 edit statements change 6,073 parameter values. Its one-shot, tool-free setting is specific to that experiment. [Car protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md) · [Patch code](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/answer.md)
- The math experiment allows limited forward and activation inspection and uses direction, layer, strength, and random controls. Development and final evaluation are separate. A pass can include many sequences. [Math protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) · [Final 50-pass report](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/final_answer.md)
- The Gemma follow-up documents cases where direct editing produced unreliable gains. [Gemma negative results](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md)
- Offline RL has its own data support assumptions. This workflow starts with direct edits. [Offline RL tutorial and review](https://arxiv.org/abs/2005.01643)

For strict reproduction, freeze hashes of actual source files and the deployment dtype. Record the available commit. A moving main branch can change. Separate author interpretations, published results, local reproduction, and inference.

## What this skill adapts

The original research spans multiple tasks and feedback settings. This skill groups cross-layer routing from the car experiment and directional and component edits from the math experiment into optional intervention modules. The time-envelope formulas apply to the particular policy basis described in the robot reference. [Intervention workflow](intervention-workflow.md) · [Optional tools](open-source-tools.md)

Evaluate application of an author's patch, independent repair discovery, development on a user's policy, and independent hardware gains as separate evidence. Use task-appropriate budgets, controls, and human hardware execution. Choose tools by the current question and apply any no-tools constraints only when relevant.

Choose modules for the current question. Earlier candidates' confirmation checks are reusable where applicable. Record actual cost, review points, and changes of direction. Report any interrupted fixed-endpoint confirmation as incomplete. [Research efficiency](research-efficiency.md)

The tool guide links primary sources for pyvene, LEACE, weightspace, APRNN, EMMET, and LeRobot. Record `source_reviewed`, `installed`, `synthetic_tested`, `policy_adapter_tested`, and `human_robot_confirmed` according to the evidence available. The bundled NumPy tests establish only the behavior of that helper.

## Robot simulation experience

The index preserves paths, hashes, and verification scope. Some entries refer to private historical archives that require separate access. Six original local report paths in L1–L7 were unavailable when checked again on November 9, 2026. Treat those reports as historical evidence. [Experience evidence index](experience-evidence.json)

**Goblet receiver approach and timing** (`goblet_receiver_entry_timing`). A gradient-free analytic edit reconstructs the same historical candidate from its frozen baseline. It changes six scalars, moving the receiver approach by −1.35 mm and scaling five learned stage durations to 72%. Fixed observations verify the computation. Targeted closed-loop screening went from 2/6 to 6/6. Paired confirmation on new initial states went from 18/28 to 25/28, with eight repairs and one regression. The preregistered 1→0 direction went from 14/22 to 21/22 (one-sided exact McNemar p = 0.01953). The 0→1 direction stayed at 4/6. The direction-balanced net difference was about +15.9 percentage points, with a conservative 95% interval from −36.0 to +60.9 points. Two earlier confirmation batches missed their own thresholds with the same candidate. The result applies to the frozen candidate and tested simulation conditions.

- **L1, report checked.** Three scalar edits moved the internal target as predicted. The physical object landed in the opposite direction.
- **L2, report checked.** A mid-trajectory pulse ended before the fault. Check the basis envelope and timing before enlarging the same block.
- **L3, report checked.** A late change repaired repeated resets of one case for both signs. This supports a local effect and leaves the sign and Jacobian interpretation unresolved.
- **L4, report checked.** Eight exposed development cases went from 4/8 to 5/8. Both policies achieved 10/12 on independent bidirectional initial states. Report these results separately and retain the uncertainty.
- **L5, report checked.** An earliest-row match at `u = 1` caused a guard stop. Separate same-state computational effects from cross-rollout differences. Preserve the original stop and preregister the corrected check.
- **L6a, report checked.** The plan pointed to an old baseline while the probe came from the correct one. The smoke test blocked before inference. Verify model pointers and manifest provenance. **L6b, methodological reminder.** Initialization, collisions, execution limits, and evaluator behavior can also cause failures. Record implementation fixes separately.
- **L7, report checked.** Audit forwards consume budget. Passive parsing and reuse of existing outputs have separate accounting. Avoid replaying complete histories on every metadata check.

Choose scales, stage indices, CPU affinity, sample sizes, and budgets for the actual policy, hardware, task, and authorization.
