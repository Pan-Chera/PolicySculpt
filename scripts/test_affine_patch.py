#!/usr/bin/env python3
"""Synthetic neural computation tests; no actual policy/hardware or third-party APIs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import affine_patch as a


class AffineTests(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[1., 2., -1.], [0.5, -0.25, 2.]])
        self.b = np.array([0.2, -0.3])
        self.x = np.array([[1., -1., 0.2], [0., 0.5, -2.], [0.3, 1., -0.7]])

    def check(self, side, p, c, bias=True):
        b = self.b if bias else None
        new_w, new_b = a.fold_affine(self.w, b, p, c, side)
        result = a.compare_intervention(self.w, b, p, c, side, self.x, new_w, new_b, 1e-12, 1e-12)
        self.assertTrue(result["equivalent_on_given_features"], result)
        return new_w, new_b

    def test_input_affine_with_nonsymmetric_projection(self):
        self.check("input", np.array([[1., .5, 0.], [0., 1., 0.], [.1, 0., 1.]]), np.array([.1, -.1, .3]))

    def test_output_affine_uses_original_bias(self):
        self.check("output", np.array([[.8, .3], [-.2, 1.]]), np.array([.1, -.3]))

    def test_biasless_projection_preserves_architecture(self):
        _, b = self.check("input", np.diag([1., 0., 1.]), np.zeros(3), bias=False)
        self.assertIsNone(b)

    def test_biasless_translation_cannot_add_parameter(self):
        with self.assertRaisesRegex(a.Invalid, "new bias"):
            a.fold_affine(self.w, None, np.eye(3), np.ones(3), "input")

    def test_noop_preserves_input_and_reports_noop(self):
        before = self.w.copy()
        nw, nb = self.check("input", np.eye(3), np.zeros(3))
        report = a.compare_intervention(self.w, self.b, np.eye(3), np.zeros(3), "input", self.x, nw, nb, 0., 0.)
        self.assertTrue(report["no_op_after_cast"])
        np.testing.assert_array_equal(self.w, before)

    def test_two_layer_network_equivalence_after_hook_removal(self):
        p, c = np.array([[.9, .1], [-.3, 1.]]), np.array([.2, -.1])
        nw, nb = self.check("output", p, c)
        hook_hidden = np.tanh((self.x @ self.w.T + self.b) @ p.T + c)
        saved_hidden = np.tanh(self.x @ nw.T + nb)
        head = np.array([[.4, -.2]])
        np.testing.assert_allclose(hook_hidden @ head.T, saved_hidden @ head.T, atol=1e-12)

    def test_transform_cannot_commute_through_tanh(self):
        y = self.x @ self.w.T + self.b
        self.assertFalse(np.allclose(np.tanh(2 * y), 2 * np.tanh(y)))

    def test_projection_changes_target_but_preserves_orthogonal_component(self):
        p, c = np.diag([0., 1.]), np.zeros(2)
        nw, nb = self.check("output", p, c)
        edited = self.x @ nw.T + nb
        original = self.x @ self.w.T + self.b
        np.testing.assert_allclose(edited[:, 0], 0.)
        np.testing.assert_allclose(edited[:, 1], original[:, 1])

    def test_selected_writers_do_not_edit_unselected_columns(self):
        result = a.edit_output_writers(self.w, np.array([1., 0.]), [0, 2], 1.)
        np.testing.assert_array_equal(result[:, 1], self.w[:, 1])
        np.testing.assert_array_equal(result[0, [0, 2]], 0.)
        np.testing.assert_array_equal(result[1], self.w[1])

    def test_invalid_writers_rejected(self):
        for direction, columns, alpha in [(np.ones(2), [0], 1.), (np.array([1., 0.]), [0, 0], 1.),
                                           (np.array([1., 0.]), [True], 1.), (np.array([1., 0.]), [0], float('nan'))]:
            with self.subTest(columns=columns, alpha=alpha), self.assertRaises(a.Invalid):
                a.edit_output_writers(self.w, direction, columns, alpha)

    def test_axis_dtype_and_nonfinite_rejected(self):
        for w, b, p, c in [(self.w, self.b, np.eye(2), np.zeros(2)),
                            (self.w, self.b.astype(np.float32), np.eye(3), np.zeros(3)),
                            (self.w.astype(int), None, np.eye(3), np.zeros(3)),
                            (self.w, self.b, np.eye(3), np.array([float('nan'), 0., 0.]))]:
            with self.subTest(shape=w.shape), self.assertRaises(a.Invalid):
                a.fold_affine(w, b, p, c, "input")

    def test_float16_noop_after_rounding(self):
        w, b = self.w.astype(np.float16), self.b.astype(np.float16)
        p, c = np.eye(3) * (1. + 1e-10), np.zeros(3)
        nw, nb = a.fold_affine(w, b, p, c, "input")
        report = a.compare_intervention(w, b, p, c, "input", self.x, nw, nb, 1e-2, 1e-2)
        self.assertTrue(report["no_op_after_cast"])

    def test_overflow_rejected(self):
        with self.assertRaisesRegex(a.Invalid, "overflow"):
            a.fold_affine(self.w.astype(np.float16), self.b.astype(np.float16), np.eye(3) * 1e9,
                          np.zeros(3), "input")

    def test_bad_tolerance_and_feature_axis_rejected(self):
        nw, nb = self.w.copy(), self.b.copy()
        for x, atol in [(self.x, -1.), (self.x, float('inf')), (np.ones((3, 2)), 1e-3)]:
            with self.subTest(atol=atol), self.assertRaises(a.Invalid):
                a.compare_intervention(self.w, self.b, np.eye(3), np.zeros(3), "input", x, nw, nb, atol, 0.)

    def test_cli_save_reload_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/'layer.npz', weight=self.w, bias=self.b)
            np.savez(root/'transform.npz', P=np.diag([1., 0., 1.]), offset=np.zeros(3))
            np.savez(root/'features.npz', features=self.x)
            command = [sys.executable, str(Path(a.__file__)), '--layer', str(root/'layer.npz'),
                       '--transform', str(root/'transform.npz'), '--features', str(root/'features.npz'),
                       '--side', 'input', '--atol', '1e-12', '--rtol', '1e-12', '--out', str(root/'new.npz'),
                       '--report', str(root/'report.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'], 'local_affine_check_only')
            self.assertFalse(report['hardware_executed'])
            self.assertFalse(report['model_executed'])
            with np.load(root/'new.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['weight'][:, 1], 0.)
                self.assertEqual(data['bias'].dtype, self.b.dtype)
            before = (root/'new.npz').read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(before, (root/'new.npz').read_bytes())

    def test_cli_equivalence_failure_leaves_report_without_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/'layer.npz', weight=self.w.astype(np.float16), bias=self.b.astype(np.float16))
            np.savez(root/'transform.npz', P=np.eye(2) * .311, offset=np.array([.0003, -.0007]))
            np.savez(root/'features.npz', features=self.x)
            command = [sys.executable, str(Path(a.__file__)), '--layer', str(root/'layer.npz'),
                       '--transform', str(root/'transform.npz'), '--features', str(root/'features.npz'),
                       '--side', 'output', '--atol', '0', '--rtol', '0', '--out', str(root/'new.npz'),
                       '--report', str(root/'report.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertFalse((root/'new.npz').exists())
            self.assertFalse(json.loads((root/'report.json').read_text())['equivalent_on_given_features'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
