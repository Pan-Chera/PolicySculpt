"""Read robot traces and compare observed channels within matching phase occurrences."""
import gzip
import json
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def episode(path, kind):
    if kind == "goblet":
        rows = []
        for item in jsonl(path):
            pre, post, quality = (item[key] for key in ("before", "after", "quality"))
            air = quality["air"]
            projected = item.get("projected_joint_references")
            raw = item["action"]
            rows.append({
                "tick": pre["tick"], "phase": str(item["mode"]),
                "events": {
                    "air_handover_verified": quality["air_handover_verified"],
                    "placement_success": quality["success"],
                    "air_event": air["event"],
                },
                "signals": {
                    "policy_joint_rad": raw[:7] + raw[8:15],
                    "policy_closure": [raw[7], raw[15]],
                    "projected_joint_reference": projected,
                    "measured_joint_rad": [q for arm in post["arms"] for q in arm["joint_q_rad"][:7]],
                    "object_xyz_m": post["object"]["pose"]["position_m"],
                    "table_contact_N": quality["table_contact_N"],
                },
            })
        return rows
    if kind == "cube":
        trace = read(path)
        return [{
            "tick": item["tick"], "phase": "turn:" + trace["row"]["move"],
            "events": {"wrong_layer_contact": bool(item["wrong"])},
            "signals": {
                "recorded_action": item.get("action"),
                "holding_tool_axis_mm": item["tool_axis_mm"]["hold"],
                "object_xyz_m": item["pose"],
            },
        } for item in trace["turn"]]
    return list(jsonl(path))


def phases(rows):
    counts = defaultdict(int)
    result = {}
    if not rows:
        raise ValueError("empty trace")
    if any(type(row["tick"]) is not int for row in rows) or any(
        right["tick"] <= left["tick"] for left, right in zip(rows, rows[1:])
    ):
        raise ValueError("trace ticks must be strictly increasing integers")
    for phase, group in groupby(rows, lambda row: row["phase"]):
        occurrence = counts[phase]
        counts[phase] += 1
        segment = list(group)
        start = segment[0]["tick"]
        result[(phase, occurrence)] = {row["tick"]-start: row for row in segment}
    return result


def numbers(value):
    if isinstance(value, list):
        return [x for item in value for x in numbers(item)]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("signal values must be finite numbers")
    return [value]


def difference(left, right):
    a, b = numbers(left), numbers(right)
    if len(a) != len(b) or not a:
        raise ValueError("incompatible signal dimensions")
    index = max(range(len(a)), key=lambda i: abs(a[i]-b[i]))
    return abs(a[index]-b[index]), index, a[index], b[index]


def diagnose(base, candidate, thresholds):
    a, b = phases(base), phases(candidate)
    if any(not math.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError("thresholds must be finite and nonnegative")
    first = {}
    aligned = 0
    coverage = defaultdict(int)
    for phase, left in a.items():
        if phase not in b:
            continue
        right = b[phase]
        l0, r0 = left[0]["tick"], right[0]["tick"]
        if l0 != r0 and "phase_start" not in first:
            first["phase_start"] = {"phase": list(phase), "base_tick": l0, "candidate_tick": r0}
        for elapsed in sorted(left.keys() & right.keys()):
            x, y = left[elapsed], right[elapsed]
            aligned += 1
            window = {"phase": list(phase), "offset": elapsed, "base_tick": x["tick"], "candidate_tick": y["tick"]}
            for name, limit in thresholds.items():
                u, v = (row["signals"].get(name) for row in (x, y))
                if u is None or v is None:
                    continue
                coverage[name] += 1
                change, index, before, after = difference(u, v)
                if change > limit and name not in first:
                    first[name] = {**window, "max_abs_delta": change, "threshold": limit,
                                   "component": index, "base": before, "candidate": after}
            for name in sorted(x["events"].keys() & y["events"].keys()):
                u, v = (row["events"][name] for row in (x, y))
                if u != v and "event:" + name not in first:
                    first["event:" + name] = {**window, "base": u, "candidate": v}
    events = {}
    for role, rows in (("base", base), ("candidate", candidate)):
        found = {}
        for row in rows:
            for name, value in row["events"].items():
                if value:
                    item = found.setdefault(name, {"first_tick": row["tick"], "observed_ticks": 0})
                    item["observed_ticks"] += 1
        events[role] = found
    return {
        "alignment": "phase occurrence and elapsed ticks",
        "interpretation": "observed divergence; closed-loop traces alone do not establish causality",
        "aligned_ticks": aligned,
        "unaligned_ticks": {"base": len(base)-aligned, "candidate": len(candidate)-aligned},
        "unmatched_phases": {"base": [key for key in a if key not in b],
                             "candidate": [key for key in b if key not in a]},
        "signal_coverage": {name: coverage[name] for name in thresholds},
        "first_divergence": first,
        "events": events,
    }
