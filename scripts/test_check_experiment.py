#!/usr/bin/env python3
"""Synthetic consistency tests; no model, driver, sensor, or external API."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import check_experiment as c


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.plan = json.loads((assets / "experiment.template.json").read_text())
        self.plan.update(template_only=False, stage_id="synthetic-stage", task_mode="single_hypothesis",
                         entry_route="weight_edit", base_sha256="a" * 64, runtime_source_sha256="b" * 64)
        self.plan["budget"] = dict(max_forward_frame_equivalents=100, max_candidates=2,
                                   max_analysis_seconds=60, hardware_episodes=2)
        self.plan_path, self.record_path = self.root / "plan.json", self.root / "record.json"
        evidence = self.root / "evidence.txt"
        evidence.write_text("synthetic report, not a true behavioral verification\n")
        self.record = json.loads((assets / "experiment-record.template.json").read_text())
        self.record.update(template_only=False, artifacts=[dict(id="e", path=evidence.name,
                           sha256=c.digest(evidence))])
        self.save()

    def save(self):
        self.plan_path.write_text(json.dumps(self.plan))
        self.record["plan_sha256"] = c.digest(self.plan_path)
        self.record_path.write_text(json.dumps(self.record))

    def run_check(self, artifacts=False):
        self.save()
        return c.check(self.plan_path, self.record_path, artifacts)

    def reject(self, message):
        with self.assertRaisesRegex(c.Invalid, message):
            self.run_check()

    def candidate(self, memory=False):
        self.record["candidates"] = [dict(id="candidate-1", representation="in_memory" if memory else "checkpoint",
            base_sha256="a" * 64, edit_program_sha256="c" * 64,
            candidate_sha256=None if memory else "d" * 64)]

    def assurance(self, level):
        a = self.record["adapter_assurance"]
        a.update(level=level, scope_id="known-test-scope", base_sha256="a" * 64,
                 runtime_source_sha256="b" * 64, evidence_refs=["e"], rng_mode="deterministic")
        self.record["data_usage"] = [dict(group_id="dev", session_id="dev-session", used_for=["construction", "selection"], exposed_before_freeze=True)]
        return a

    def claim(self, kind, **kwargs):
        self.record["claims"] = [dict(kind=kind, evidence_refs=["e"], **kwargs)]

    def confirmation(self, complete=True, real=False):
        self.candidate()
        self.record["confirmation"].update(started=True, frozen_candidate_sha256="d" * 64,
            observed_candidate_sha256=["d" * 64, "d" * 64], inference_design_completed=complete)
        self.record["data_usage"] = [dict(group_id="confirm", used_for=["confirmation"], exposed_before_freeze=False)]
        self.record["closed_loop"].update(source="real_robot" if real else "simulation", completed_episodes=2,
                                         human_executed=real, evidence_refs=["e"])

    def test_feasibility_can_finish_at_structure_without_candidate_or_base(self):
        self.plan.update(task_mode="feasibility", base_sha256=None, entry_route="insufficient_evidence")
        self.claim("structural_hypothesis")
        self.assertEqual(self.run_check()["adapter_level"], "A0")

    def test_plan_only_is_not_execution_authorization(self):
        self.save()
        result = c.check(self.plan_path)
        self.assertFalse(result["execution_authorized_by_checker"])
        self.assertFalse(result["declarations_verified"])

    def test_editing_needs_designated_base(self):
        self.plan["base_sha256"] = None
        self.reject("designated base")

    def test_budget_and_stop_fields_reject_negative_bool_or_missing(self):
        for change in [-1, True, None]:
            with self.subTest(change=change):
                self.plan["budget"]["max_candidates"] = change
                self.reject("max_candidates")
        self.plan["budget"]["max_candidates"] = 2
        self.plan["stop_conditions"] = []
        self.reject("stop_conditions")

    def test_actual_usage_is_retained_as_inconsistency_not_hidden(self):
        self.record["actual_usage"]["forward_frame_equivalents"] = 101
        self.reject("actual usage exceeds")
        self.assertEqual(self.record["actual_usage"]["forward_frame_equivalents"], 101)

    def test_record_bound_to_plan_hash(self):
        self.save()
        self.plan_path.write_text(json.dumps(dict(self.plan, stage_id="changed")))
        with self.assertRaisesRegex(c.Invalid, "plan hash mismatch"):
            c.check(self.plan_path, self.record_path)

    def test_candidate_wrong_base_rejected(self):
        self.candidate()
        self.record["candidates"][0]["base_sha256"] = "e" * 64
        self.reject("base mismatch")

    def test_memory_variant_does_not_need_full_checkpoint(self):
        self.candidate(memory=True)
        self.assertEqual(self.run_check()["candidates"], 1)

    def test_construction_and_selection_can_share_development_data(self):
        self.assurance("A1")
        self.claim("fixed_input_effect")
        self.assertEqual(self.run_check()["claims"], 1)

    def test_exposed_confirmation_rejected(self):
        self.record["data_usage"] = [dict(group_id="g", used_for=["confirmation"], exposed_before_freeze=True)]
        self.reject("confirmation data exposed")

    def test_same_session_or_case_cannot_be_relabelled_independent(self):
        for field in ["session_id", "physical_case_group"]:
            with self.subTest(field=field):
                self.record["data_usage"] = [dict(group_id="dev", used_for=["selection"], exposed_before_freeze=True,
                                                   **{field: "shared"}),
                                             dict(group_id="confirm", used_for=["confirmation"], exposed_before_freeze=False,
                                                   **{field: "shared"})]
                self.reject("data group leakage")

    def test_structure_cannot_claim_observed_computational_effect(self):
        self.claim("fixed_input_effect")
        self.reject("exceeds adapter")

    def test_adapter_evidence_bound_to_runtime_scope(self):
        a = self.assurance("A1")
        a["runtime_source_sha256"] = "f" * 64
        self.reject("scope mismatch")

    def test_required_state_not_known_cannot_claim_parity(self):
        a = self.assurance("A1")
        a.update(state_required=True, state_verified=False)
        self.reject("pre-state not verified")

    def test_sequence_claim_needs_sequence_evidence(self):
        a = self.assurance("A1")
        self.claim("sequence_effect")
        self.reject("exceeds adapter")
        a.update(level="A2")
        self.reject("sequence effect not verified")
        a.update(sequence_verified=True)
        self.assertEqual(self.run_check()["adapter_level"], "A2")

    def test_shared_seed_does_not_verify_pairing(self):
        a = self.assurance("A1")
        a.update(stochastic=True, rng_mode="shared_seed_only", pairing_verified=True, pairing_evidence_refs=["e"])
        self.reject("do not verify pairing")

    def test_explicit_noise_pairing_requires_its_own_evidence(self):
        a = self.assurance("A2")
        a.update(stochastic=True, rng_mode="explicit_inputs", pairing_verified=True)
        self.claim("fixed_input_effect", noise_comparison="same_perturbation")
        self.reject("pairing evidence")
        a["pairing_evidence_refs"] = ["e"]
        self.assertEqual(self.run_check()["claims"], 1)

    def test_repeated_sampling_allowed_without_fake_pairing(self):
        a = self.assurance("A2")
        a.update(stochastic=True, rng_mode="repeated_sampling", pairing_verified=False)
        self.claim("fixed_input_effect", noise_comparison="across_perturbations")
        self.assertEqual(self.run_check()["claims"], 1)
        self.record["claims"][0]["noise_comparison"] = "same_perturbation"
        self.reject("same perturbation")

    def test_stochastic_effect_requires_comparison_but_raw_sample_description_does_not(self):
        a = self.assurance("A1")
        a.update(stochastic=True, rng_mode="shared_seed_only")
        self.claim("fixed_input_effect")
        self.reject("declare stochastic comparison")
        self.claim("sampled_output_description")
        self.assertEqual(self.run_check()["claims"], 1)

    def test_saved_candidate_requires_reload_integrity_and_preservation(self):
        a = self.assurance("A3")
        self.candidate()
        a.update(candidate_sha256="d" * 64)
        self.claim("saved_candidate_effect")
        self.reject("save/reload/preservation")
        a.update(save_reload_verified=True, preservation_checked=True)
        self.reject("integrity")
        self.record["integrity"] = {k: "not_applicable" for k in ["aliases_views", "parametrizations", "dependent_caches", "reload_topology"]}
        self.assertEqual(self.run_check()["adapter_level"], "A3")
        a["sequence_required"] = True
        self.reject("required sequence check")

    def test_confirmation_detects_changed_hash_but_allows_repeated_same_hash(self):
        self.confirmation()
        self.assertEqual(self.run_check()["status"], "record_consistency_only")
        self.record["confirmation"]["observed_candidate_sha256"].append("e" * 64)
        self.reject("changed after confirmation")

    def test_confirmation_cannot_freeze_only_an_unsaved_memory_variant(self):
        self.confirmation()
        self.record["candidates"][0]["representation"] = "in_memory"
        self.reject("frozen saved candidate missing")

    def test_early_stop_allows_description_but_not_completed_confirmation_claim(self):
        self.confirmation(complete=False)
        self.claim("closed_loop_description")
        self.assertEqual(self.run_check()["claims"], 1)
        self.claim("independent_performance")
        self.reject("completed frozen independent design")

    def test_real_robot_description_needs_human_execution_record(self):
        self.confirmation(real=True)
        self.claim("closed_loop_description")
        self.assertEqual(self.run_check()["claims"], 1)
        self.record["closed_loop"]["human_executed"] = False
        self.reject("human execution")

    def test_real_execution_does_not_inflate_adapter_level(self):
        self.confirmation()
        self.claim("independent_performance")
        result = self.run_check()
        self.assertEqual(result["adapter_level"], "A0")
        self.assertFalse(result["physical_effectiveness_verified"])

    def test_artifact_content_checks_are_optional_and_explicit(self):
        self.save()
        (self.root/'evidence.txt').write_text("changed")
        self.assertFalse(c.check(self.plan_path, self.record_path)["file_integrity_checked"])
        with self.assertRaisesRegex(c.Invalid, "hash mismatch"):
            c.check(self.plan_path, self.record_path, True)

    def test_cli_read_only_output_and_error_exit(self):
        self.save()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        command = [sys.executable, '-B', str(Path(c.__file__)), '--plan', str(self.plan_path),
                   '--record', str(self.record_path), '--check-artifacts']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = json.loads(result.stdout)
        self.assertFalse(output['model_executed'])
        self.assertFalse(output['physical_effectiveness_verified'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.record['plan_sha256'] = 'f' * 64
        self.record_path.write_text(json.dumps(self.record))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['status'], 'inconsistent')

    def test_collection_plan_human_owner_budget_and_stop_fields(self):
        assets = Path(__file__).resolve().parents[1] / "assets"
        plan = json.loads((assets / "collection-plan.template.json").read_text())
        plan.update(template_only=False, collector_mode="passive-recording",
                    budget=dict(max_episodes=2, max_wall_seconds=60, max_storage_bytes=100000),
                    stop_conditions=["operator-request", "storage-cap"])
        path = self.root / "collection.json"
        path.write_text(json.dumps(plan))
        result = c.check_collection(path)
        self.assertFalse(result["execution_authorized_by_checker"])
        for change, message in [({"execution_owner": "agent"}, "human operator"),
                                ({"collector_mode": "unknown"}, "collector mode"),
                                ({"stop_conditions": []}, "stop_conditions"),
                                ({"budget": dict(max_episodes=0, max_wall_seconds=60, max_storage_bytes=1000)}, "positive finite cap")]:
            with self.subTest(change=change):
                path.write_text(json.dumps(dict(plan, **change)))
                with self.assertRaisesRegex(c.Invalid, message):
                    c.check_collection(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
