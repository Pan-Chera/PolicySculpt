"""Regression checks for sparse checkpoint transport without model execution."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from checkpoint_patch import Checkpoint, apply, create


class CheckpointPatchTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def npz_pair(self):
        base, candidate, patch = (self.root / name for name in ("base.npz", "candidate.npz", "edit.npz"))
        original = np.asfortranarray(np.arange(12, dtype=">f4").reshape(3, 4))
        edited = original.copy(order="F")
        edited[0, 0] = -0.0
        edited[2, 1] = -5.25
        np.savez_compressed(base, anchors=original, scalar=np.float32(1), names=np.array(["D", "D2"]))
        np.savez_compressed(candidate, anchors=edited, scalar=np.float32(2), names=np.array(["D", "D2"]))
        return base, candidate, patch

    def test_roundtrip_preserves_bits_dtype_and_untouched_metadata(self):
        base, candidate, patch = self.npz_pair()
        self.assertEqual([x["changed"] for x in create(base, candidate, patch)["edits"]], [2, 1])
        output = self.root / "recovered.npz"
        apply(base, patch, output)
        restored, expected = Checkpoint(output), Checkpoint(candidate)
        self.assertEqual(restored.state_hash(), expected.state_hash())
        self.assertTrue(np.signbit(restored.values["anchors"][0, 0]))
        self.assertEqual(restored.values["scalar"].shape, ())
        self.assertEqual(restored.values["names"].tolist(), ["D", "D2"])

    def test_same_state_in_different_file_is_rejected(self):
        base, candidate, patch = self.npz_pair()
        create(base, candidate, patch)
        wrong = self.root / "repacked.npz"
        np.savez(wrong, **Checkpoint(base).values)
        self.assertEqual(Checkpoint(wrong).state_hash(), Checkpoint(base).state_hash())
        output = self.root / "output.npz"
        with self.assertRaisesRegex(ValueError, "base checkpoint SHA-256 mismatch"):
            apply(wrong, patch, output)
        self.assertFalse(output.exists())

    def test_tampering_and_missing_base_binding_are_rejected(self):
        base, candidate, patch = self.npz_pair()
        create(base, candidate, patch)
        output = self.root / "output.npz"
        with np.load(patch, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays["value_0"][0, 1] ^= 1
        np.savez_compressed(patch, **arrays)
        with self.assertRaisesRegex(ValueError, "reconstructed state SHA-256 mismatch"):
            apply(base, patch, output)
        self.assertFalse(output.exists())
        manifest = json.loads(arrays["manifest"].tobytes())
        manifest["base_sha256"] = None
        arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
        np.savez_compressed(patch, **arrays)
        with self.assertRaisesRegex(ValueError, "invalid base checkpoint SHA-256"):
            apply(base, patch, output)
        self.assertFalse(output.exists())

    def test_never_overwrites_existing_output(self):
        base, candidate, patch = self.npz_pair()
        create(base, candidate, patch)
        output = self.root / "output.npz"
        output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            apply(base, patch, output)
        self.assertEqual(output.read_bytes(), b"existing")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_bfloat16_and_shared_storage(self):
        import torch

        base, candidate, patch, output = (self.root / name for name in
                                          ("base.pt", "candidate.pt", "edit.npz", "out.pt"))
        model = torch.nn.Linear(3, 2).to(torch.bfloat16)
        original = model.state_dict()
        torch.save(original, base)
        with torch.no_grad():
            model.weight[0, 0] = original["weight"][0, 0].float() + 2
        torch.save(model.state_dict(), candidate)
        create(base, candidate, patch)
        apply(base, patch, output)
        self.assertEqual(Checkpoint(output).state_hash(), Checkpoint(candidate).state_hash())
        aliased = self.root / "aliased.pt"
        torch.save({"a": model.weight.detach(), "b": model.weight.detach()}, aliased)
        with self.assertRaisesRegex(ValueError, "shared tensor storage"):
            Checkpoint(aliased)


if __name__ == "__main__":
    unittest.main()
