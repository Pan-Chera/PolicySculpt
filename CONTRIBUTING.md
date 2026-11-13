# Contributing

Use [GitHub Issues](https://github.com/Pan-Chera/PolicySculpt/issues) for bugs and proposals. For a bug, include the PolicySculpt commit, host and version (Codex or Claude Code), Python and model runtime, the exact command, and the expected and observed behavior. A small synthetic input or sanitized trace helps maintainers reproduce the problem. Keep private checkpoints, data, credentials, and robot logs out of issues and pull requests.

For changes to the editing workflow, describe the behavior the change supports and how you verified it. For a new research case, identify the baseline, frozen candidate, evaluation rule, development and confirmation splits, and any regressions. See [case evidence](docs/reproducibility.md) for examples of the reporting scope.

## Run the checks

Use Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
python -m unittest discover -s scripts
python -m unittest discover -s claude-code/skills/edit/scripts
python -m unittest discover -s claude-code/skills/embodied/tools
```

The PyTorch checkpoint test runs when PyTorch is installed. Run it with the target project's PyTorch runtime when changing that format. For Claude Code packaging changes, also run `claude plugin validate --strict "$PWD/claude-code"` with the Claude Code CLI installed.

## Maintain both packages

The repository root is the Codex skill. `claude-code/skills/edit/` carries the editing skill for Claude Code. Sync shared changes to `assets/`, `references/`, and `scripts/` into that package, and keep each `SKILL.md` entry point appropriate to its host. The standalone robot tools are bundled in `claude-code/skills/embodied/tools/`.

Before tagging a release, validate both host entry points from a fresh checkout, run the [standalone example](docs/first-edit.md), and check local links and published demo assets. Update the Claude plugin manifest version when releasing a changed plugin. Record the tested runtime and changes in the GitHub release notes.
