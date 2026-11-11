# PolicySculpt for Claude Code

A Claude Code plugin from Tsinghua University IEI Lab. It exposes two independently callable skills.

| Command | Use |
| --- | --- |
| `/policysculpt:edit` | Diagnose a behavior in a trained policy, construct and save a direct weight edit, and verify it against the baseline |
| `/policysculpt:embodied` | Diagnose robot traces, run or resume paired simulations, and audit generated and applied action chunks |

## Load for a session

From the project where you want Claude Code to work, pass the absolute path of this directory:

```bash
claude --plugin-dir /absolute/path/to/claude-code
```

For example, enter one of these in that session:

```text
/policysculpt:edit Reproduce the baseline on the provided logs and propose a direct edit for the release failure.
/policysculpt:embodied Compare the two recorded runs and locate the first contact divergence.
```

The editing skill bundles the English PolicySculpt instructions, references, templates, and checkpoint utilities. The embodied skill bundles the standalone Python tools. Run tools that need server-only logs, weights, or simulation on the machine that already hosts them.

Use the project's Python environment for bundled scripts. The affine and checkpoint utilities use NumPy, and PyTorch checkpoint patches require a runtime with `Tensor.untyped_storage()`. The standalone trace, receipt, and chunk tools use the Python 3.10+ standard library. Native simulation and visual ACT probes use their task-specific environments.

## Validate the package

```bash
claude plugin validate --strict /absolute/path/to/claude-code
```

English editing instructions and support files were taken from a pinned PolicySculpt revision. The Claude entry point uses the plugin's namespaced commands and skill-relative paths. [Upstream revision](https://github.com/Pan-Chera/PolicySculpt/tree/0bd4af48f5dca9bfd97912d1194a56c76a45f2b1) · [Claude Code plugin instructions](https://code.claude.com/docs/en/plugins/create)
