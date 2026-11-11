import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from chunks import audit_chunks
from pairs import reference, run
from traces import diagnose


class ToolTests(unittest.TestCase):
    def test_phase_alignment_and_missing_signal(self):
        def frame(tick, phase, contact, signal):
            return {"tick": tick, "phase": phase, "events": {"contact": contact}, "signals": signal}

        base = [frame(2, "a", False, {}), frame(6, "b", False, {"joint": [1, 0]}),
                frame(7, "b", True, {})]
        candidate = [frame(2, "a", False, {}), frame(4, "b", False, {"joint": [1, .2]}),
                     frame(5, "b", False, {})]
        report = diagnose(base, candidate, {"joint": .1, "applied": .1})
        first = report["first_divergence"]
        self.assertEqual((first["joint"]["base_tick"], first["joint"]["candidate_tick"]), (6, 4))
        self.assertIs(first["event:contact"]["candidate"], False)
        self.assertEqual(report["signal_coverage"]["applied"], 0)

    def test_chunk_prefix_and_stale_output(self):
        report = audit_chunks([
            {"tick": 0, "generated": {"id": "a", "actions": [[0], [1]]},
             "sent": {"id": "a", "index": 0, "action": [0]}},
            {"tick": 1, "generated": {"id": "b", "actions": [[2], [3]]},
             "applied": {"id": "a", "index": 1, "action": [1.2]}},
        ])
        self.assertEqual(report["chunks"][0]["sent"], 1)
        self.assertEqual({x["reason"] for x in report["mismatches"]},
                         {"stale_chunk", "nonprefix_execution", "action_changed"})

    def test_simulation_receipt_resume_and_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = root / "evaluate.py"
            command.write_text("""
import json, hashlib, sys
request = json.load(sys.stdin)
encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
print(json.dumps(dict(request_sha256=hashlib.sha256(encoded).hexdigest(),
                      native_initial_sha256=request["initial_sha256"],
                      loaded_models={k: v["sha256"] for k,v in request["model"].items()},
                      result=dict(status="valid", success=request["role"] == "candidate"))))
""")
            base, candidate = root / "base", root / "candidate"
            base.write_text("old")
            candidate.write_text("new")
            plan = {"mode": "simulation", "models": {"base": {"policy": reference(base)},
                                                     "candidate": {"policy": reference(candidate)}},
                    "evaluator": {"runtime": reference(sys.executable), "bridge": reference(command)},
                    "command": {"argv": [sys.executable, str(command)], "cwd": temp, "timeout_s": 10},
                    "cases": [{"id": "a", "initial": {"pose": [1.000000001, 2, 3]}}]}
            destination = root / "run"
            first = run(plan, destination, 1)
            self.assertEqual(first["counts"]["incomplete"], 1)
            result = run(plan, destination)
            self.assertEqual((result["new_arms"], result["reused_arms"], result["repairs"]), (1, 1, ["a"]))
            changed = copy.deepcopy(plan)
            changed["cases"][0]["initial"]["pose"][0] = 1.000000002
            with self.assertRaisesRegex(ValueError, "initial state changed"):
                run(changed, destination)
            replacement = root / "replacement"
            replacement.write_text("replacement")
            for key, role in (("models", "base"), ("models", "candidate"), ("evaluator", "bridge")):
                changed = copy.deepcopy(plan)
                if key == "models":
                    changed[key][role]["policy"] = reference(replacement)
                else:
                    changed[key][role] = reference(replacement)
                with self.assertRaisesRegex(ValueError, "evaluator.*changed"):
                    run(changed, destination)
            receipt = next(p for p in destination.glob("*.json") if len(p.stem) == 64)
            body = json.loads(receipt.read_text())
            body["body"]["result"]["success"] = not body["body"]["result"]["success"]
            receipt.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "inconsistent receipt"):
                run(plan, destination)
            candidate.write_text("changed")
            with self.assertRaisesRegex(ValueError, "bound file changed"):
                run(plan, destination)


if __name__ == "__main__":
    unittest.main()
