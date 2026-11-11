"""Tests for identity-bound, atomic paired evaluation receipts."""

import json
from pathlib import Path
import tempfile
import unittest

from sample_resume import SampleJournal, fingerprint


class SampleJournalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run"
        self.roles = {name: fingerprint(name) for name in ("baseline", "candidate", "evaluator")}
        self.samples = {name: fingerprint(name) for name in ("a", "b")}

    def test_resume_in_different_order_and_ignore_uncommitted_file(self):
        journal = SampleJournal(self.path, self.roles, self.samples)
        journal.commit("a", {"success": False}, {"success": True})
        (self.path / ".pending-cutoff").write_text('{"pair":')
        restored = SampleJournal(self.path, self.roles, dict(reversed(tuple(self.samples.items()))))
        self.assertEqual(restored.completed(), {"a": {"baseline": {"success": False},
                                                      "candidate": {"success": True}}})
        restored.commit("b", {"success": True}, {"success": True})
        self.assertEqual(set(restored.completed()), {"a", "b"})

    def test_all_identity_components_are_binding(self):
        journal = SampleJournal(self.path, self.roles, self.samples)
        journal.commit("a", {}, {})
        for name in self.roles:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity mismatch"):
                SampleJournal(self.path, dict(self.roles, **{name: fingerprint("changed")}), self.samples)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            SampleJournal(self.path, self.roles, dict(self.samples, a=fingerprint("new input")))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            SampleJournal(self.path, self.roles, {"a": self.samples["a"]})
        self.assertEqual(set(journal.completed()), {"a"})

    def test_single_arm_and_duplicate_commit_do_not_replace_pair(self):
        journal = SampleJournal(self.path, self.roles, self.samples)
        with self.assertRaisesRegex(ValueError, "both result"):
            journal.commit("a", {}, None)
        self.assertFalse(journal.completed())
        journal.commit("a", {"value": 0}, {"value": 1})
        with self.assertRaises(FileExistsError):
            journal.commit("a", {"value": 4}, {"value": 8})
        self.assertEqual(journal.completed()["a"]["candidate"], {"value": 1})

    def test_modified_receipt_is_rejected(self):
        journal = SampleJournal(self.path, self.roles, self.samples)
        journal.commit("a", {"value": 0}, {"value": 1})
        path = journal.pair_path("a")
        record = json.loads(path.read_text())
        record["pair"]["candidate"]["value"] = 2
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "inconsistent paired receipt"):
            journal.completed()


if __name__ == "__main__":
    unittest.main()
