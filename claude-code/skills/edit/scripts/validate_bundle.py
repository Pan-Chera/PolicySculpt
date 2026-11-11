#!/usr/bin/env python3
"""Read-only JSON evidence/log contract checks; no model or hardware execution.

This validates declarations and file integrity, not provenance truth, policy
compatibility, robot safety, or physical effectiveness. Python standard library.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def bad_constant(value):
    raise Invalid(f"nonfinite JSON constant: {value}")


def parse(text):
    value = json.loads(text, object_pairs_hook=strict_pairs, parse_constant=bad_constant)
    finite(value)
    return value


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite numeric value")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_field(obj, key):
    value = obj.get(key)
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
            f"{key}: require lowercase SHA256, not a placeholder")
    require(value != "0" * 64, f"{key}: zero placeholder hash")


def nonempty(obj, keys):
    for key in keys:
        require(isinstance(obj.get(key), str) and bool(obj[key].strip()), f"missing {key}")


def numeric_vector(value, context):
    require(isinstance(value, list) and len(value) > 0, f"{context}: require action vector")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value),
            f"{context}: nonnumeric action vector")
    finite(value)


def check_logs(root, episode, fields, required):
    path = Path(episode["log_path"])
    if not path.is_absolute():
        path = root / path
    require(path.is_file(), f"log missing: {path}")
    require(digest(path) == episode["log_sha256"], f"log hash mismatch: {path}")
    last_tick = last_time = None
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            require(bool(line.strip()), f"blank JSONL row: {path}:{line_no}")
            row = parse(line)
            require(isinstance(row, dict), "JSONL row must be an object")
            require(row.get("episode_id") == episode["episode_id"], "row episode mismatch")
            tick, stamp = row.get("tick"), row.get("timestamp_s")
            require(isinstance(tick, int) and not isinstance(tick, bool) and tick >= 0,
                    "invalid tick")
            require(isinstance(stamp, (int, float)) and not isinstance(stamp, bool),
                    "invalid timestamp_s")
            require(last_tick is None or tick == last_tick + 1,
                    "tick gap/reversal: split/reconstruct explicitly, do not silently bridge")
            require(last_time is None or stamp > last_time, "non-monotonic timestamp")
            obs = row.get("observation")
            require(isinstance(obs, dict), "missing observation object")
            require(set(obs) <= fields, "observation contains non-whitelisted fields")
            require(required <= set(obs), "required policy observation missing")
            require(all(obs[k] is not None for k in required), "required observation is null")
            commands = row.get("commands")
            require(isinstance(commands, dict), "missing commands object")
            for name, available in episode["command_availability"].items():
                require(name in commands, f"commands.{name} must be explicit, null if unknown")
                command = commands[name]
                if available:
                    require(isinstance(command, dict), f"commands.{name} missing")
                    nonempty(command, ["provenance"])
                    numeric_vector(command.get("value"), f"commands.{name}")
                else:
                    require(command is None, f"commands.{name}: conflicts with declared absence")
            last_tick, last_time = tick, stamp
            count += 1
    require(count > 0, f"empty episode: {path}")
    return count


def validate_manifest(path, check_log_files=False):
    path = Path(path)
    obj = parse(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), "manifest must be an object")
    require(type(obj.get("schema_version")) is int and obj["schema_version"] == 1,
            "unsupported schema_version")
    require(obj.get("template_only") is False, "template is not an admitted dataset")
    nonempty(obj, ["dataset_id", "robot_id", "calibration_id", "observation_schema_id", "action_schema_id"])
    require(obj.get("source") in {"real_robot", "simulation", "synthetic"}, "invalid source")
    for key in ["policy_model_sha256", "runtime_source_sha256", "controller_sha256"]:
        hash_field(obj, key)
    control = obj.get("control")
    require(isinstance(control, dict), "missing control contract")
    rate = control.get("rate_hz")
    require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0,
            "invalid control rate")
    require(control.get("observation_timing") == "pre_action", "normalize timing before admission")
    nonempty(control, ["action_semantics"])
    fields, required = control.get("policy_fields"), control.get("required_policy_fields")
    require(isinstance(fields, list) and len(fields) > 0 and all(isinstance(k, str) and k for k in fields),
            "invalid policy_fields")
    require(len(fields) == len(set(fields)), "duplicate policy fields")
    require(isinstance(required, list) and required and all(isinstance(k, str) for k in required),
            "invalid required_policy_fields")
    require(len(required) == len(set(required)), "duplicate required policy fields")
    require(set(required) <= set(fields), "required field is not whitelisted")
    require(not set(fields) & {"quality", "success", "outcome", "teacher_action", "contact_truth"},
            "audit/teacher/truth field cannot be declared a policy input")
    for key in ["units", "frames"]:
        require(isinstance(control.get(key), dict) and control[key], f"missing {key} declaration")
    episodes = obj.get("episodes")
    require(isinstance(episodes, list) and episodes, "no episodes")
    ids, groups = set(), {}
    rows, limitations = 0, []
    for ep in episodes:
        require(isinstance(ep, dict), "episode must be an object")
        nonempty(ep, ["episode_id", "session_id", "physical_case_group", "log_path"])
        require(ep["episode_id"] not in ids, "duplicate episode_id")
        ids.add(ep["episode_id"])
        require(ep.get("split") in {"development", "offline_confirmation", "regression"}, "invalid split")
        require(isinstance(ep.get("exposed"), bool), "exposure must be explicit")
        require(not (ep["split"] == "offline_confirmation" and ep["exposed"]),
                "exposed episode cannot be unseen confirmation")
        for key in ["session_id", "physical_case_group"]:
            group = (key, ep[key])
            require(group not in groups or groups[group] == ep["split"], f"split leakage: {group}")
            groups[group] = ep["split"]
        hash_field(ep, "log_sha256")
        availability = ep.get("command_availability")
        require(isinstance(availability, dict) and set(availability) == {"proposed", "accepted", "executed"},
                "declare availability of proposed/accepted/executed separately")
        require(all(isinstance(v, bool) for v in availability.values()), "availability must be boolean")
        if not availability["executed"]:
            limitations.append(f"{ep['episode_id']}: executed command unavailable; action lineage incomplete")
        if check_log_files:
            rows += check_logs(path.parent, ep, set(fields), set(required))
    return obj, {"manifest_sha256": digest(path), "episodes": len(episodes),
                 "log_files_checked": check_log_files, "rows_checked": rows,
                 "limitations": limitations}


def check_evidence(path, manifest_path, manifest):
    obj = parse(Path(path).read_text(encoding="utf-8"))
    require(isinstance(obj, dict) and type(obj.get("schema_version")) is int
            and obj["schema_version"] == 1, "bad evidence schema")
    require(obj.get("template_only") is False, "evidence template is unfilled")
    require(obj.get("mode") in {"logged_state_one_step", "fixed_observation_sequence", "closed_loop_observation"},
            "invalid evidence mode")
    for key in ["base_sha256", "candidate_sha256", "dataset_manifest_sha256"]:
        hash_field(obj, key)
    require(obj["base_sha256"] == manifest["policy_model_sha256"], "evidence base identity mismatch")
    require(obj["dataset_manifest_sha256"] == digest(manifest_path), "evidence dataset identity mismatch")
    require(obj.get("source") == manifest["source"], "evidence source mismatch")
    require(isinstance(obj.get("metrics"), dict), "missing metrics")
    if obj["mode"] != "closed_loop_observation":
        nonempty(obj, ["state_conditioning"])
        require(obj.get("performance_claim") == "none", "offline evidence cannot claim physical performance")
        def check_metric_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    k = key.lower()
                    require(not (k in {"success_rate", "return", "rescues", "regressions"}
                                 or k.endswith("_success_rate") or k.endswith("_return")),
                            f"offline metric implies unobserved candidate outcomes: {key}")
                    check_metric_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_metric_keys(child)
        check_metric_keys(obj["metrics"])
    else:
        hash_field(obj, "execution_report_sha256")
        execution_ids = obj.get("independent_execution_ids")
        require(isinstance(execution_ids, list) and execution_ids
                and all(isinstance(v, str) and v.strip() for v in execution_ids)
                and len(execution_ids) == len(set(execution_ids)),
                "closed-loop evidence needs actual execution identities")
    return {"mode": obj["mode"], "evidence_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-logs", action="store_true")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    try:
        manifest, result = validate_manifest(args.manifest, args.check_logs)
        if args.evidence:
            result["evidence"] = check_evidence(args.evidence, args.manifest, manifest)
        result.update(status="metadata_valid_only", physical_effectiveness_verified=False,
                      model_executed=False, source_identity_contents_verified=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Invalid, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
