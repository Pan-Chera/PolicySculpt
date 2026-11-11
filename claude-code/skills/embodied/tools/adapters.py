"""Adapters for archived Goblet and Cube2 native evaluator receipts."""
from pathlib import Path

from pairs import reference, verified
from traces import read


def adapter_sources():
    directory = Path(__file__).resolve().parent
    return {f"tool:{name}": reference(directory / name) for name in ("adapters.py", "traces.py", "pairs.py")}


def recorded(adapter, sample, role):
    ref = sample["receipts"][role]
    row = read(verified(ref))
    initial = sample["initial"]
    if adapter == "goblet":
        if (row["seed"], row["giver"], row["actual_initial_pose"]) != (
            initial["seed"], initial["giver"], initial["pose"]
        ) or row["model_sha256"] != sample["native_models"][role]:
            raise ValueError(f"Goblet runtime receipt identity mismatch for {sample['id']}/{role}")
        if row["status"] not in ("success", "task_failure", "failure", "completed", "verified"):
            raise ValueError(f"unrecognized Goblet status: {row['status']}")
        return {"status": "valid", "success": row["success"], "failure": row["first_missing_event"],
                "ticks": row["steps"], "source": ref}
    if adapter == "cube":
        if row["id"] != initial["id"] or row["solution"] != initial["solution"]:
            raise ValueError(f"Cube2 case or solver sequence mismatch for {sample['id']}/{role}")
        return {**cube_result(row), "source": ref}
    raise ValueError(f"unknown recorded adapter: {adapter}")


def cube_result(row):
    failure = row["failure_reason"]
    status = "technical_invalid" if failure == "rollout_error" else "valid"
    return {"status": status, "success": row["completed"] if status == "valid" else None,
            "failure": failure, "simulated_collision_guard": failure == "robot_collision_guard",
            "attempted_moves": len(row["attempts"])}


def goblet_manifest(root):
    evidence = root / "rsi_state/real_rsi_v1/skill_evidence/goblet_wsp_receiver_entry_timing_20261109"
    confirm = evidence / "confirm"
    report = read(confirm / "COMPARISON.json")
    offline = read(evidence / "evidence/OFFLINE.json")
    refs = {role: offline[role]["sha256"] for role in ("base", "candidate")}
    models = {}
    for role in refs:
        path = evidence / "checkpoints" / Path(offline[role]["path"]).name
        models[role] = {"checkpoint": {"path": str(path), "sha256": refs[role]}}
    cases = [{
        "id": row["case_id"],
        "initial": {"seed": row["seed"], "giver": row["giver"], "pose": row["checkpoint"]["pose"]},
        "native_models": refs,
        "receipts": {"base": row["checkpoint"]["receipt"], "candidate": row["candidate"]["receipt"]},
    } for row in report["pairs"]]
    evaluator = {name: reference(confirm / name) for name in ("PLAN.json", "COMPARISON.json")}
    evaluator.update(adapter_sources())
    return {"mode": "recorded", "adapter": "goblet", "models": models, "evaluator": evaluator, "cases": cases}


def cube_manifest(root):
    evidence = root / "rsi_state/real_rsi_v1/skill_evidence"
    replay = evidence / "cube2_wsp_original30_20261110"
    binding = read(replay / "RUN_BINDING.json")
    cohort_path = verified(binding["cohort"])
    original = cohort_path.parent
    candidate = Path(binding["candidate"]["path"])
    hashes = binding["other_model_hashes"]
    models = {
        "base": {"checkpoint": {"path": str(original/"base_model.npz"), "sha256": hashes["base"]}},
        "candidate": {"checkpoint": binding["candidate"]},
    }
    for role in models:
        path = (original if role == "base" else candidate.parent) / "d2_model.npz"
        models[role]["d2"] = {"path": str(path), "sha256": hashes["d2"]}
    cohort = {case["id"]: case for case in read(cohort_path)["cases"]}
    cases = []
    for row in read(replay / "PAIRS.json"):
        receipt = dict(row["wsp_log"])
        receipt["path"] = str(replay / receipt["path"])
        cases.append({"id": str(row["id"]), "initial": cohort[row["id"]],
                      "receipts": {"base": row["baseline_log"], "candidate": receipt}})
    evaluator = {name: reference(replay / name) for name in ("RUN_BINDING.json", "PAIRS.json")}
    evaluator["cohort"] = binding["cohort"]
    evaluator.update({name: {"path": str(original/"source"/name), "sha256": sha}
                      for name, sha in binding["source_hashes"].items()})
    evaluator.update(adapter_sources())
    return {"mode": "recorded", "adapter": "cube", "models": models, "evaluator": evaluator, "cases": cases}


def cube_simulation_manifest(root, runtime, case_id):
    plan = cube_manifest(root)
    source = Path(plan["evaluator"]["evaluate.py"]["path"]).parent
    case = next((case for case in plan["cases"] if case["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown Cube2 case: {case_id}")
    bridge = Path(__file__).with_name("cube_native.py").resolve()
    plan["evaluator"].update(runtime=reference(runtime), bridge=reference(bridge))
    plan["evaluator"].update({f"asset:{path.relative_to(source.parent)}": reference(path)
                              for folder in ("meshes", "assets")
                              for path in sorted((source.parent/folder).rglob("*")) if path.is_file()})
    return {"mode": "simulation", "models": plan["models"], "evaluator": plan["evaluator"],
            "cases": [{"id": case["id"], "initial": case["initial"]}],
            "command": {"argv": [str(runtime), "-B", str(bridge), str(source)],
                        "cwd": str(root), "timeout_s": 240,
                        "environment": {"MUJOCO_GL": "disable", "CUDA_VISIBLE_DEVICES": "",
                                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                        "MKL_NUM_THREADS": "1"}}}
