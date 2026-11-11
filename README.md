# PolicySculpt

**Improve trained robot policies by editing their weights directly.**

PolicySculpt is a Codex skill that fixes a specific failure in an existing policy. In paired simulation tests, direct weight edits improved bimanual handoff and 2×2 cube solving with **zero gradient training steps**.

Tsinghua University IEI Lab · [Lab website](https://maureenzou.github.io/lab.html)

[![Codex Skill](https://img.shields.io/badge/Codex-Skill-3069b3.svg)](#quick-start) [![Paired simulation evaluation](https://img.shields.io/badge/Evaluation-Paired%20simulation-5e5aa7.svg)](#robotics-results)

## Robotics Results

| Simulated task | Paired test | Original → Edited | Parameters changed |
| --- | --- | ---: | ---: |
| Bimanual goblet handoff | Successful handoff and stable placement on 28 new starts | **18/28 → 25/28** | **6** of 90,281 |
| Bimanual 2×2 cube solving | Full solves on the same 30 scrambles | **21/30 → 25/30** | **5,474** of 2,084,480 |

### Bimanual Goblet Handoff

We edited six parameters that govern the receiving arm’s approach and phase timing. In the same-start example below, the original policy leaves the goblet tilted **12°**, while the edited policy sets it down upright. [Watch MP4](assets/showcase/goblet-handoff.mp4)

![Simulated goblet placement from the same initial state. Original policy on the left, edited policy on the right.](assets/showcase/goblet-handoff.gif)

### Bimanual 2×2 Cube Solving

We shifted learned trajectory anchors for the holding gripper away from the rotating layer. In this close-up, cross-layer contact fell from **41/200 to 0/200 control steps**. Red marks contact, and green marks clearance. [Second close-up](assets/showcase/cube-case-28.gif)

![Simulated bottom-layer turn. The original gripper contacts the rotating layer on the left. The edited gripper clears it on the right.](assets/showcase/cube-case-06.gif)

The replay kept all **21** original solves. On a separate set of 10 new scrambles, split evenly between ordinary and high-risk move sequences, full solves rose from **4/10 to 10/10**.

<details>
<summary>Two other uses of the same workflow</summary>

- **Connect Four.** Editing the action head removed **7,928 mirror disagreements** across 24,000 held-out endgames, bringing the count to **zero**. The edit changed **1,798** of 134,735 parameters. [Board comparison](assets/showcase/connect-four.png)
- **Photo retouching.** Editing the exposure regressor reduced mean highlight clipping from **12.90% to 2.50%** across 65 eligible held-out FiveK photos. The edit changed **129** of 6,123,680 parameters. [Photo comparison](assets/showcase/exposure-rocket.jpg)

</details>

## How It Works

1. Trace a repeatable failure to the computation that drives it.
2. Edit the relevant weights, screen candidates, then save and reload one checkpoint.
3. Run the original and edited policies on the same test cases and check that previously successful behavior still passes.

Use PolicySculpt when a trained policy already has the skill you need and you can isolate a specific failure. The policy adapter runs native inference. The task evaluator measures outcomes in your simulator or on fixed inputs. The approach builds on Takehi et al.’s work on reading and editing model weights directly. [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)

## Quick Start

From your project root, clone the skill. Access to the private repository is required.

```bash
git clone https://github.com/Pan-Chera/PolicySculpt.git \
  .agents/skills/robot-weight-programming
```

Give Codex your checkpoint, native inference command, and an evaluator with representative failures and successes.

```text
Use $robot-weight-programming to fix this repeatable failure.
Edit the relevant weights, save and reload a candidate, and compare it
with the original on held-out cases. Check that the original successes still pass.
```

[Skill instructions](SKILL.md) · [Robotics setup](references/robot-workflow.md) · [Policy adapters](references/adapter-and-integrity.md) · [Evaluation and resumption](references/research-efficiency.md)

## Authors and Citation

**Guanxing Qu** and **Xueyan Zou**, Tsinghua University IEI Lab. [Citation metadata](CITATION.cff)

```bibtex
@misc{qu2026policysculpt,
  author       = {Qu, Guanxing and Zou, Xueyan},
  title        = {{PolicySculpt (PS): Direct Weight Editing for Trained Policies}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Pan-Chera/PolicySculpt}
}
```

Photo-retouching references and demo credit. [Exposure paper](https://arxiv.org/abs/1709.09602) · [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) · [Falcon 9 photo by SpaceX](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.rocket)
