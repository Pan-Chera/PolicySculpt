"""Single-case bridge to the existing Cube2 native simulator and acceptance logic."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from adapters import cube_result
from pairs import identity, verified
from traces import read


def main():
    source = Path(sys.argv[1]).resolve()
    request = json.load(sys.stdin)
    if request["mode"] != "simulation":
        raise ValueError("Cube2 bridge requires simulation mode")
    native_cases = read(source.parent / "cohort.json")["cases"]
    native = next(row for row in native_cases if str(row["id"]) == request["case_id"])
    if identity(native) != request["initial_sha256"] or native != request["initial"]:
        raise ValueError("native Cube2 cohort no longer matches requested initial case")

    sys.path.insert(0, str(source))
    import evaluate

    if Path(evaluate.__file__).resolve() != source / "evaluate.py":
        raise ValueError("wrong Cube2 evaluator loaded")
    models = request["model"]
    evaluate.BASE = evaluate.Policy.load(verified(models["checkpoint"]))
    evaluate.D2 = evaluate.D2Model.load(verified(models["d2"]))
    with TemporaryDirectory(prefix="embodied-cube-") as work:
        evaluate.DEST = Path(work)
        evaluate.solve_case(native)
        receipt = read(evaluate.DEST / f"case_{native['id']:02d}.json")
    print(json.dumps({"request_sha256": identity(request), "native_initial_sha256": identity(native),
                      "loaded_models": {name: ref["sha256"] for name, ref in models.items()},
                      "result": {**cube_result(receipt), "native_receipt": receipt}}, allow_nan=False))


if __name__ == "__main__":
    main()
