"""Audit generated chunks against returned prefixes or recorded applied actions."""
import math

from traces import difference, numbers


def audit_chunks(rows, tolerance=1e-6):
    if not rows or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("require a nonempty chunk trace and finite nonnegative tolerance")
    chunks = {}
    current = None
    previous_tick = None
    replans = []
    mismatches = []
    reports = []
    for row in rows:
        tick = row["tick"]
        if type(tick) is not int or (previous_tick is not None and tick <= previous_tick):
            raise ValueError("chunk trace ticks must strictly increase")
        previous_tick = tick
        if "generated" in row:
            item = row["generated"]
            name = item["id"]
            actions = item["actions"]
            if name in chunks or not actions or any(not numbers(action) for action in actions):
                raise ValueError("expected unique chunk IDs and nonempty actions")
            current = name
            chunks[name] = {"actions": actions, "generated": len(actions), "sent": 0, "applied": 0}
            replans.append(tick)
            reports.append({"chunk": name, "tick": tick, "generated": len(actions)})
        for channel in ("sent", "applied"):
            if channel not in row:
                continue
            item = row[channel]
            name, index = item["id"], item["index"]
            if name not in chunks or type(index) is not int or not 0 <= index < chunks[name]["generated"]:
                mismatches.append({"tick": tick, "channel": channel, "reason": "unknown_chunk_or_index"})
                continue
            chunk = chunks[name]
            if name != current:
                mismatches.append({"tick": tick, "channel": channel, "reason": "stale_chunk"})
            if index != chunk[channel]:
                mismatches.append({"tick": tick, "channel": channel, "reason": "nonprefix_execution",
                                   "expected_index": chunk[channel], "observed_index": index})
            amount, _, _, _ = difference(chunk["actions"][index], item["action"])
            if amount > tolerance:
                mismatches.append({"tick": tick, "channel": channel, "reason": "action_changed",
                                   "max_abs_delta": amount})
            chunk[channel] += 1
    for item in reports:
        chunk = chunks[item["chunk"]]
        item.update(sent=chunk["sent"], applied=chunk["applied"])
    return {"chunks": reports, "replan_intervals_ticks": [b-a for a, b in zip(replans, replans[1:])],
            "physical_execution_observed": any("applied" in row for row in rows), "mismatches": mismatches}
