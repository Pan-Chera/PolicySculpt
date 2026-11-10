#!/usr/bin/env python3
"""Synthetic contract/CLI tests. No policy, simulator, driver, or real logs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import validate_bundle as v


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.manifest = json.loads((assets / "dataset.template.json").read_text())
        self.manifest.update(template_only=False, source="synthetic", dataset_id="test-data",
                             policy_model_sha256="a" * 64, runtime_source_sha256="b" * 64,
                             controller_sha256="c" * 64)
        self.episode = self.manifest["episodes"][0]
        self.rows = [dict(episode_id=self.episode["episode_id"], tick=tick,
                          timestamp_s=tick / 30,
                          observation=dict(joint_q_rad=[0.1, 0.2], gripper_m=0.03),
                          commands=dict(proposed=dict(value=[0.11, 0.21], provenance="synthetic-actor"),
                                        accepted=dict(value=[0.11, 0.21], provenance="synthetic-controller"),
                                        executed=None)) for tick in range(3)]
        self.path = self.root / "dataset.json"
        self.evidence_path = self.root / "evidence.json"
        self.evidence = json.loads((assets / "evidence.template.json").read_text())
        self.evidence.update(template_only=False, source="synthetic", base_sha256="a" * 64,
                             candidate_sha256="d" * 64)
        self.save()

    def save(self):
        log_path = self.root / self.episode["log_path"]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.episode["log_sha256"] = v.digest(log_path)
        self.path.write_text(json.dumps(self.manifest))
        self.evidence["dataset_manifest_sha256"] = v.digest(self.path)
        self.evidence_path.write_text(json.dumps(self.evidence))

    def validate(self):
        return v.validate_manifest(self.path, True)

    def assert_invalid(self, fragment, action=None):
        with self.assertRaisesRegex(v.Invalid, fragment):
            (action or self.validate)()

    def test_valid_log_missing_executed_is_explicit_limitation(self):
        _, result = self.validate()
        self.assertEqual(result["rows_checked"], 3)
        self.assertEqual(result["episodes"], 1)
        self.assertIn("executed command unavailable", result["limitations"][0])

    def test_unfilled_template_and_boolean_schema_rejected(self):
        for change, fragment in [({"template_only": True}, "template"),
                                 ({"schema_version": True}, "schema_version")]:
            with self.subTest(change=change):
                original = dict(self.manifest)
                self.manifest.update(change)
                self.save()
                self.assert_invalid(fragment)
                self.manifest = original

    def test_hash_mismatch(self):
        self.episode["log_sha256"] = "e" * 64
        self.path.write_text(json.dumps(self.manifest))
        self.assert_invalid("hash mismatch")

    def test_future_audit_field_not_policy_input(self):
        self.rows[0]["observation"]["success"] = True
        self.save()
        self.assert_invalid("non-whitelisted")

    def test_required_input_missing(self):
        del self.rows[0]["observation"]["joint_q_rad"]
        self.save()
        self.assert_invalid("required policy observation missing")

    def test_no_fabricated_executed_command(self):
        self.rows[0]["commands"]["executed"] = copy.deepcopy(self.rows[0]["commands"]["accepted"])
        self.save()
        self.assert_invalid("conflicts with declared absence")

    def test_action_provenance_required(self):
        self.rows[0]["commands"]["accepted"]["provenance"] = ""
        self.save()
        self.assert_invalid("provenance")

    def test_tick_gap_and_timestamp_reversal(self):
        self.rows[1]["tick"] = 9
        self.save()
        self.assert_invalid("tick gap/reversal")
        self.rows[1]["tick"] = 1
        self.rows[1]["timestamp_s"] = -1
        self.save()
        self.assert_invalid("non-monotonic")

    def test_nonfinite_and_duplicate_json_keys(self):
        for raw in ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x":1,"x":2}']:
            with self.subTest(raw=raw), self.assertRaises(v.Invalid):
                v.parse(raw)

    def test_session_and_case_leakage(self):
        for field in ["session_id", "physical_case_group"]:
            with self.subTest(field=field):
                extra = copy.deepcopy(self.episode)
                extra.update(episode_id="other", session_id="other-session",
                             physical_case_group="other-case", split="offline_confirmation", exposed=False)
                extra[field] = self.episode[field]
                self.manifest["episodes"] = [self.episode, extra]
                self.save()
                self.assert_invalid("split leakage", lambda: v.validate_manifest(self.path))

    def test_exposed_confirmation_rejected(self):
        self.episode["split"] = "offline_confirmation"
        self.save()
        self.assert_invalid("exposed episode")

    def test_valid_offline_evidence(self):
        self.assertEqual(v.check_evidence(self.evidence_path, self.path, self.manifest)["mode"],
                         "logged_state_one_step")

    def test_offline_success_claim_rejected(self):
        for metrics in [{"candidate_success_rate": 0.9}, {"nested": [{"rescues": 1}]}]:
            with self.subTest(metrics=metrics):
                self.evidence["metrics"] = metrics
                self.save()
                self.assert_invalid("unobserved candidate outcomes",
                                    lambda: v.check_evidence(self.evidence_path, self.path, self.manifest))

    def test_evidence_identity_mismatch(self):
        for change in [{"source": "real_robot"}, {"base_sha256": "e" * 64},
                       {"dataset_manifest_sha256": "e" * 64}]:
            with self.subTest(change=change):
                evidence = dict(self.evidence, **change)
                self.evidence_path.write_text(json.dumps(evidence))
                self.assert_invalid("mismatch",
                                    lambda: v.check_evidence(self.evidence_path, self.path, self.manifest))

    def test_closed_loop_needs_report_and_execution_identity(self):
        self.evidence.update(mode="closed_loop_observation", execution_report_sha256="e" * 64,
                             independent_execution_ids=[])
        self.save()
        self.assert_invalid("actual execution identities",
                            lambda: v.check_evidence(self.evidence_path, self.path, self.manifest))

    def test_cli_reports_metadata_only_and_rejects_bad_input(self):
        command = [sys.executable, str(Path(v.__file__)), "--manifest", str(self.path),
                   "--check-logs", "--evidence", str(self.evidence_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "metadata_valid_only")
        for key in ["physical_effectiveness_verified", "model_executed", "source_identity_contents_verified"]:
            self.assertIs(report[key], False)
        self.rows[0]["commands"]["accepted"]["value"] = [True]
        self.save()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["status"], "rejected")

    def test_documented_envelopes(self):
        for u in [0, 0.1, 0.5, 0.9, 1]:
            t1, d = 2 * u - 1, 0.002
            self.assertAlmostEqual((d / 2 + d / 2 * t1) * (1 - u), u * (1 - u) * d)
            self.assertAlmostEqual(d - d * (1 - u), u * d)
            self.assertAlmostEqual(d + (-1.5 * d - 0.5 * d * t1) * (1 - u), u * u * d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
