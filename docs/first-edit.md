# First weight edit

Run a small example from the repository root. It uses a two-action NumPy policy, the bundled affine editing tool, and four fixed inputs. It needs no simulator, agent account, pretrained checkpoint, or downloaded dataset.

The policy expects an observation with coordinates in training order. Here, an adapter presents the two features in reverse order, so some inputs pick the wrong action. This example folds the known input permutation into the existing weights. If you control the observation encoder, the permutation could also be corrected there. The example illustrates a direct weight edit for a fixed interface.

## Generate a baseline and edit it

Use Python 3.10 or later with NumPy installed. Create a fresh directory for the artifacts because the editing tool protects existing outputs.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
export PS_EXAMPLE="$(mktemp -d)"

python - <<'PY'
import os
from pathlib import Path

import numpy as np

work = Path(os.environ["PS_EXAMPLE"])
np.savez(work / "base.npz", weight=np.eye(2, dtype=np.float32))
np.savez(work / "transform.npz",
         P=np.array([[0, 1], [1, 0]], dtype=np.float32),
         offset=np.zeros(2, dtype=np.float32))
np.savez(work / "dev.npz", features=np.array([[0, 2], [2, 0]], dtype=np.float32))
PY

python scripts/affine_patch.py \
  --layer "$PS_EXAMPLE/base.npz" \
  --transform "$PS_EXAMPLE/transform.npz" \
  --features "$PS_EXAMPLE/dev.npz" --side input --atol 0 --rtol 0 \
  --out "$PS_EXAMPLE/candidate.npz" --report "$PS_EXAMPLE/edit.json"
```

The report should say `local_affine_check_only` and record four changed weight entries. This confirms the fold on the supplied development features.

## Reload and evaluate on separate inputs

```bash
python - <<'PY'
import os
from pathlib import Path

import numpy as np

work = Path(os.environ["PS_EXAMPLE"])
expected_order = np.array([[3, 1], [1, 3], [4, 4], [5, 5]], dtype=np.float32)
observed = expected_order[:, ::-1]
expected = expected_order.argmax(axis=1)

def evaluate(checkpoint):
    with np.load(work / checkpoint, allow_pickle=False) as data:
        return (observed @ data["weight"].T).argmax(axis=1) == expected

base = evaluate("base.npz")
edited = evaluate("candidate.npz")
repairs = (~base & edited).sum()
regressions = (base & ~edited).sum()
print(f"Correct decisions: {base.sum()}/4 -> {edited.sum()}/4")
print(f"Repairs: {repairs}; regressions: {regressions}")
assert (base.sum(), edited.sum(), repairs, regressions) == (2, 4, 2, 0)
PY
```

The equal-feature cases protect the existing tie behavior. This example measures fixed-input decisions. For a real policy, use its native inference and serialization, provide a separate evaluation set, and run paired rollouts when the claim depends on future states. See the [adapter guide](../references/adapter-and-integrity.md) and [evaluation guide](../references/research-efficiency.md).
