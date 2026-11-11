#!/usr/bin/env python3
"""Read-only consistency checks for a scoped experiment plan and optional record.

Checks declarations/file hashes, not model behavior, statement truth, statistical
validity, execution authorization, or safety. No imports of a model or driver.
"""
import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True

from validate_bundle import Invalid, digest, hash_field, nonempty, parse, require


LEVELS = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
MODES = {"feasibility", "single_hypothesis", "full_research"}
PHASES = {"exploration", "screening", "closed_loop_probe", "confirmation"}
CLAIMS = {"structural_hypothesis": 0, "fixed_input_effect": 1, "sequence_effect": 2,
          "sampled_output_description": 1, "saved_candidate_effect": 3,
          "closed_loop_description": 0, "independent_performance": 0}


def object_file(path):
    obj = parse(Path(path).read_text(encoding="utf-8"))
    require(isinstance(obj, dict), "JSON root must be an object")
    require(type(obj.get("schema_version")) is int and obj["schema_version"] == 1, "bad schema_version")
    require(obj.get("template_only") is False, "template is not an experiment record")
    return obj


def strings(value, name, allow_empty=False, unique=True):
    require(isinstance(value, list) and (allow_empty or value), f"{name}: require list")
    require(all(isinstance(v, str) and v.strip() for v in value), f"{name}: invalid string")
    require(not unique or len(set(value)) == len(value), f"{name}: duplicate value")
    return value


def number(value, name, integer=False):
    require(type(value) in ((int,) if integer else (int, float)) and value >= 0,
            f"{name}: require finite nonnegative {'integer' if integer else 'number'}")


def check_data(rows):
    require(isinstance(rows, list), "data_usage: require list")
    seen = {}
    confirmation = False
    for row in rows:
        require(isinstance(row, dict), "data_usage row must be object")
        nonempty(row, ["group_id"])
        uses = set(strings(row.get("used_for"), "used_for"))
        require(uses <= {"diagnosis", "construction", "selection", "regression", "confirmation"}, "unknown data use")
        exposed = row.get("exposed_before_freeze")
        require(type(exposed) is bool, "declare exposed_before_freeze")
        is_confirmation = "confirmation" in uses
        require(not (is_confirmation and (exposed or uses & {"diagnosis", "construction", "selection"})),
                "confirmation data exposed/used for construction or selection")
        for key in ("group_id", "session_id", "physical_case_group"):
            if key in row:
                nonempty(row, [key])
                group = (key, row[key])
                previous = seen.get(group, [])
                for was_confirmation, was_exposed, old_uses in previous:
                    require(not ((is_confirmation and (was_exposed or old_uses & {"diagnosis", "construction", "selection"}))
                                 or (was_confirmation and (exposed or uses & {"diagnosis", "construction", "selection"}))),
                            f"data group leakage: {group}")
                seen.setdefault(group, []).append((is_confirmation, exposed, uses))
        confirmation |= is_confirmation
    return confirmation


def check(plan_path, record_path=None, check_artifacts=False):
    plan = object_file(plan_path)
    nonempty(plan, ["stage_id", "decision_question", "review_point"])
    require(plan.get("task_mode") in MODES, "invalid task_mode")
    require(plan.get("stage_purpose") in PHASES, "invalid stage_purpose")
    require(plan.get("entry_route") in {"implementation_fix", "missing_observation", "weight_edit",
            "capability_gap", "insufficient_evidence"}, "invalid entry_route")
    strings(plan.get("stop_conditions"), "stop_conditions")
    budget = plan.get("budget")
    require(isinstance(budget, dict), "missing budget")
    for key in ("max_forward_frame_equivalents", "max_candidates", "hardware_episodes"):
        number(budget.get(key), key, True)
    number(budget.get("max_analysis_seconds"), "max_analysis_seconds")
    require(budget["max_analysis_seconds"] > 0, "analysis time limit must be positive")
    if plan.get("base_sha256") is not None:
        hash_field(plan, "base_sha256")
    else:
        require(plan["task_mode"] == "feasibility", "editing plan needs designated base")
    result = dict(status="record_consistency_only", plan_sha256=digest(plan_path),
                  model_executed=False, physical_effectiveness_verified=False,
                  declarations_verified=False, file_integrity_checked=False,
                  execution_authorized_by_checker=False, warnings=[])
    if record_path is None:
        result["warnings"].append("plan only; no execution/candidate/claim checks performed")
        return result
    record = object_file(record_path)
    hash_field(record, "plan_sha256")
    require(record["plan_sha256"] == digest(plan_path), "record plan hash mismatch")
    artifacts = record.get("artifacts", [])
    require(isinstance(artifacts, list), "artifacts must be list")
    ids = set()
    for artifact in artifacts:
        require(isinstance(artifact, dict), "artifact must be object")
        nonempty(artifact, ["id", "path"])
        require(artifact["id"] not in ids, "duplicate artifact id")
        ids.add(artifact["id"])
        hash_field(artifact, "sha256")
        if check_artifacts:
            path = Path(artifact["path"])
            if not path.is_absolute():
                path = Path(record_path).parent / path
            require(path.is_file() and digest(path) == artifact["sha256"], "artifact missing/hash mismatch")

    def refs(value, name, required=True):
        values = strings(value, name, allow_empty=not required)
        require(set(values) <= ids, f"{name}: unknown artifact reference")

    candidates = record.get("candidates", [])
    require(isinstance(candidates, list), "candidates must be list")
    checkpoint_hashes, candidate_ids = set(), set()
    for candidate in candidates:
        require(isinstance(candidate, dict), "candidate must be object")
        nonempty(candidate, ["id"])
        require(candidate["id"] not in candidate_ids, "duplicate candidate id")
        candidate_ids.add(candidate["id"])
        hash_field(candidate, "base_sha256")
        require(candidate["base_sha256"] == plan.get("base_sha256"), "candidate designated base mismatch")
        hash_field(candidate, "edit_program_sha256")
        require(candidate.get("representation") in {"in_memory", "checkpoint"}, "unknown candidate representation")
        if candidate.get("candidate_sha256") is not None:
            hash_field(candidate, "candidate_sha256")
            if candidate["representation"] == "checkpoint":
                checkpoint_hashes.add(candidate["candidate_sha256"])
        else:
            require(candidate["representation"] == "in_memory", "checkpoint candidate needs hash")
    require(len(candidates) <= budget["max_candidates"], "recorded candidates exceed declared cap")
    usage = record.get("actual_usage", {})
    require(isinstance(usage, dict), "actual_usage must be object")
    for key, cap in (("forward_frame_equivalents", "max_forward_frame_equivalents"),
                     ("analysis_seconds", "max_analysis_seconds"), ("hardware_episodes", "hardware_episodes")):
        if key in usage:
            number(usage[key], f"actual_usage.{key}", integer=key != "analysis_seconds")
            require(usage[key] <= budget[cap], f"actual usage exceeds {cap}; retain actual charges and amend honestly")
    has_confirmation = check_data(record.get("data_usage", []))
    adapter = record.get("adapter_assurance")
    require(isinstance(adapter, dict) and adapter.get("level") in LEVELS, "invalid adapter assurance")
    nonempty(adapter, ["scope_id"])
    level = LEVELS[adapter["level"]]
    refs(adapter.get("evidence_refs", []), "adapter evidence", required=level > 0)
    if level > 0:
        for key in ("base_sha256", "runtime_source_sha256"):
            hash_field(adapter, key)
            require(adapter[key] == plan.get(key), f"adapter {key} scope mismatch")
        for key in ("state_required", "state_verified", "sequence_required", "sequence_verified", "stochastic",
                    "pairing_verified", "save_reload_verified", "preservation_checked"):
            require(type(adapter.get(key)) is bool, f"adapter.{key}: require boolean")
        require(not adapter["state_required"] or adapter["state_verified"], "required pre-state not verified")
        require(record.get("data_usage"), "computational evidence needs data scope")
    rng_mode = adapter.get("rng_mode", "unverified")
    if level >= 2 and adapter.get("stochastic"):
        require(rng_mode in {"explicit_inputs", "restored_streams_verified", "repeated_sampling"}, "A2 randomness protocol incomplete")
    if adapter.get("stochastic") and adapter.get("pairing_verified"):
        require(rng_mode in {"explicit_inputs", "restored_streams_verified"}, "same seed/repeated samples do not verify pairing")
        refs(adapter.get("pairing_evidence_refs", []), "randomness pairing evidence")
    if level >= 3:
        require(adapter.get("save_reload_verified") and adapter.get("preservation_checked"), "A3 save/reload/preservation missing")
        require(not adapter.get("sequence_required") or adapter.get("sequence_verified"), "A3 required sequence check missing")
        hash_field(adapter, "candidate_sha256")
        require(adapter["candidate_sha256"] in checkpoint_hashes, "A3 candidate scope mismatch: saved checkpoint required")
        integrity = record.get("integrity", {})
        require(isinstance(integrity, dict), "integrity must be object")
        for key in ("aliases_views", "parametrizations", "dependent_caches", "reload_topology"):
            require(integrity.get(key) in {"checked", "not_applicable"}, f"A3 integrity {key} unresolved")
    frozen = record.get("confirmation", {})
    require(isinstance(frozen, dict) and type(frozen.get("started", False)) is bool, "bad confirmation record")
    if frozen.get("started"):
        hash_field(frozen, "frozen_candidate_sha256")
        require(frozen["frozen_candidate_sha256"] in checkpoint_hashes, "frozen saved candidate missing")
        observed = strings(frozen.get("observed_candidate_sha256"), "observed confirmation hashes", unique=False)
        require(all(h == frozen["frozen_candidate_sha256"] for h in observed), "candidate changed after confirmation started")
    claims = record.get("claims", [])
    require(isinstance(claims, list), "claims must be list")
    for claim in claims:
        require(isinstance(claim, dict) and claim.get("kind") in CLAIMS, "unknown claim kind")
        kind = claim["kind"]
        require(level >= CLAIMS[kind], "claim exceeds adapter assurance level")
        refs(claim.get("evidence_refs", []), "claim evidence")
        if kind == "sequence_effect":
            require(adapter.get("sequence_verified"), "sequence effect not verified")
        if adapter.get("stochastic") and kind in {"fixed_input_effect", "sequence_effect", "saved_candidate_effect"}:
            comparison = claim.get("noise_comparison")
            require(comparison in {"same_perturbation", "across_perturbations"}, "declare stochastic comparison, not a bare seed")
            if comparison == "across_perturbations":
                require(level >= 2 and rng_mode in {"explicit_inputs", "restored_streams_verified", "repeated_sampling"},
                        "across perturbations needs declared sampling protocol")
        if claim.get("noise_comparison") == "same_perturbation" and adapter.get("stochastic"):
            require(adapter.get("pairing_verified"), "same perturbation claim without verified pairing")
        if kind in {"closed_loop_description", "independent_performance"}:
            execution = record.get("closed_loop", {})
            require(isinstance(execution, dict), "closed_loop must be object")
            require(execution.get("source") in {"real_robot", "simulation"}, "actual closed-loop source missing")
            number(execution.get("completed_episodes"), "completed_episodes", True)
            require(execution["completed_episodes"] > 0, "no completed closed-loop episodes")
            refs(execution.get("evidence_refs", []), "execution evidence")
            if execution["source"] == "real_robot":
                require(execution.get("human_executed") is True, "real robot result needs human execution record")
            if kind == "independent_performance":
                require(frozen.get("started") and frozen.get("inference_design_completed") is True
                        and has_confirmation, "independent performance lacks completed frozen independent design")
    result.update(record_sha256=digest(record_path), adapter_level=adapter["level"],
                  candidates=len(candidates), claims=len(claims), file_integrity_checked=check_artifacts)
    result["warnings"].append("consistency only: adapter, alias/cache/RNG and execution statements remain declarations")
    if not check_artifacts:
        result["warnings"].append("artifact contents not checked; only declared identities/references checked")
    return result


def check_collection(path):
    obj = object_file(path)
    require(obj.get("execution_owner") == "human-operator", "collection contract expects human operator")
    require(obj.get("collector_mode") in {"passive-recording", "explicit-motion-runner"}, "declare actual collector mode")
    nonempty(obj, ["purpose", "question_and_missing_evidence", "motion_behavior", "robot_and_sdk", "review_point",
                   "stop_admission_boundary_and_interface", "offline_selftest_command",
                   "human_start_and_normal_finish", "interruption_and_original_hardware_stop"])
    budget = obj.get("budget")
    require(isinstance(budget, dict), "missing collection budget")
    for key in ("max_episodes", "max_wall_seconds", "max_storage_bytes"):
        number(budget.get(key), key, integer=key != "max_wall_seconds")
        require(budget[key] > 0, f"{key}: positive finite cap required")
    strings(obj.get("stop_conditions"), "collection stop_conditions")
    return dict(status="record_consistency_only", collection_plan_sha256=digest(path),
                model_executed=False, physical_effectiveness_verified=False, declarations_verified=False,
                execution_authorized_by_checker=False,
                warnings=["collection declarations only; no SDK, motion, data quality or safety validation performed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--plan")
    target.add_argument("--collection-plan")
    parser.add_argument("--record")
    parser.add_argument("--check-artifacts", action="store_true")
    args = parser.parse_args()
    try:
        if args.collection_plan:
            require(args.record is None and not args.check_artifacts, "record/artifact options apply to --plan only")
            result = check_collection(args.collection_plan)
        else:
            result = check(args.plan, args.record, args.check_artifacts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Invalid, ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps(dict(status="inconsistent", error=str(exc), model_executed=False,
                              execution_authorized_by_checker=False), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
