# Security

For a suspected vulnerability, use **Report a vulnerability** in the repository's Security tab if private reporting is enabled. Otherwise, contact a repository maintainer privately. Include the affected commit, impact, prerequisites, and a minimal reproduction. Share confidential logs or model files only through a private channel arranged with the maintainer.

When running PolicySculpt, treat model files, adapters, and evaluator commands as inputs with different trust requirements. Use trusted checkpoints and run project-supplied adapters and evaluators in the intended runtime with only the access they need. Some checkpoint formats can execute code during loading. The NumPy example uses `allow_pickle=False`. See [adapter and integrity checks](references/adapter-and-integrity.md) for model loading, hashing, and reload verification.

For physical robot experiments, follow the project's operating procedures and the [real-robot validation guide](references/real-robot-validation.md). Keep simulation evidence and hardware validation separate.
