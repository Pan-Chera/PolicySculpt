"""Simulator-neutral, resumable evaluation with immutable per-arm receipts."""
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROLES = ("base", "candidate")
STATUSES = {"valid", "technical_invalid", "safety_stop"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def verified(ref):
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"bound file changed: {path}")
    return path


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def check_result(result):
    status = result["status"]
    if status not in STATUSES:
        raise ValueError("unknown evaluation status")
    if status == "valid":
        if type(result.get("success")) is not bool:
            raise ValueError("valid receipt requires boolean success")
    elif result.get("success") is not None:
        raise ValueError("technical and safety stops cannot be scored as task outcomes")
    return result


def evaluate_arm(plan, sample, role):
    if plan["mode"] == "recorded":
        from adapters import recorded
        return recorded(plan["adapter"], sample, role)

    request = {
        "mode": "simulation", "case_id": sample["id"],
        "initial": sample["initial"], "initial_sha256": identity(sample["initial"]),
        "role": role, "model": plan["models"][role],
        "evaluator_sha256": identity(plan["evaluator"]),
    }
    cmd = plan["command"]
    env = {**os.environ, **cmd.get("environment", {})}
    output = subprocess.run(cmd["argv"], cwd=cmd["cwd"], input=canonical(request) + b"\n",
                            capture_output=True, env=env, timeout=cmd["timeout_s"])
    if output.returncode:
        raise ValueError(f"native evaluator failed for {sample['id']}/{role}: "
                         + output.stderr.decode(errors="replace")[-3000:])
    response = json.loads(output.stdout)
    if response["request_sha256"] != identity(request):
        raise ValueError("native evaluator returned a receipt for a different request")
    result = check_result(response["result"])
    if result["status"] == "valid":
        expected = {name: ref["sha256"] for name, ref in request["model"].items()}
        if response["loaded_models"] != expected or response["native_initial_sha256"] != request["initial_sha256"]:
            raise ValueError("native evaluator reported a model or reset mismatch")
    return result


def run(plan, destination, limit=None):
    if plan["mode"] not in {"recorded", "simulation"}:
        raise ValueError("expected recorded or simulation mode")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    samples = plan["cases"]
    if not samples or len({row["id"] for row in samples}) != len(samples):
        raise ValueError("expected a nonempty cohort with unique case IDs")
    if set(plan["models"]) != set(ROLES):
        raise ValueError("require base and candidate models")
    for group in [*plan["models"].values(), plan["evaluator"]]:
        if not group:
            raise ValueError("model and evaluator bindings cannot be empty")
        for ref in group.values():
            verified(ref)
    if plan["mode"] == "recorded":
        for sample in samples:
            for role in ROLES:
                verified(sample["receipts"][role])
    else:
        command = plan["command"]
        paths = {str(Path(ref["path"]).resolve()) for ref in plan["evaluator"].values()}
        executable = Path(command["argv"][0])
        if not executable.is_absolute() or str(executable.resolve()) not in paths:
            raise ValueError("bind the native command executable in the evaluator manifest")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(plan, root, limit)


def run_locked(plan, root, limit):
    bound = root / "plan.json"
    if bound.exists():
        if canonical(json.loads(bound.read_text())) != canonical(plan):
            raise ValueError("frozen model, evaluator, receipt or initial state changed")
    else:
        atomic_json(bound, plan)
    plan_hash = identity(plan)
    new = reused = 0
    records = {}
    for sample in plan["cases"]:
        key = sample["id"]
        pair = records[key] = {}
        for role in ROLES:
            binding = {"plan": plan_hash, "sample": key, "initial": identity(sample["initial"]), "role": role}
            path = root / f"{identity([key, role])}.json"
            if path.exists():
                receipt = json.loads(path.read_text())
                body = receipt["body"]
                if receipt["sha256"] != identity(body) or body["binding"] != binding:
                    raise ValueError(f"inconsistent receipt for {key}/{role}")
                pair[role] = check_result(body["result"])
                reused += 1
            elif limit is None or new < limit:
                result = check_result(evaluate_arm(plan, sample, role))
                body = {"binding": binding, "result": result}
                atomic_json(path, {"body": body, "sha256": identity(body)})
                pair[role] = result
                new += 1

    counts = Counter({key: 0 for key in ("base_success", "candidate_success", "valid_pairs",
                                          "incomplete", "technical_invalid", "safety_stop",
                                          "simulated_collision_guard_pairs")})
    repairs, regressions = [], []
    for sample_id, pair in records.items():
        if len(pair) < 2:
            counts["incomplete"] += 1
            continue
        a, b = (pair[role] for role in ROLES)
        counts["simulated_collision_guard_pairs"] += any(
            row.get("simulated_collision_guard", False) for row in (a, b))
        statuses = {a["status"], b["status"]}
        if statuses != {"valid"}:
            for status in statuses - {"valid"}:
                counts[status] += 1
            continue
        counts["valid_pairs"] += 1
        counts["base_success"] += a["success"]
        counts["candidate_success"] += b["success"]
        if not a["success"] and b["success"]:
            repairs.append(sample_id)
        elif a["success"] and not b["success"]:
            regressions.append(sample_id)
    summary = {"mode": plan["mode"], "plan_sha256": plan_hash, "new_arms": new, "reused_arms": reused,
               "counts": dict(counts), "repairs": repairs, "regressions": regressions, "results": records}
    atomic_json(root / "summary.json", summary)
    return {key: value for key, value in summary.items() if key != "results"}
