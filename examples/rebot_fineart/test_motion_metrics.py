"""Checks for padded tails and the precise action-chunk replacement boundary."""

import unittest

import numpy as np
from motion_metrics import chunk_motion, replan_motion


class MotionMetricsTest(unittest.TestCase):
    def test_padding_does_not_create_motion(self):
        result = chunk_motion([[2], [5], [999]], [[1], [2], [999]], [1], [True, True, False])
        self.assertEqual(result["predicted"]["first_action_minus_state_per_joint"], [1])
        self.assertEqual(result["predicted"]["max_abs_step_per_joint"], [3])
        self.assertEqual(result["predicted"]["adjacent_pairs"], 1)
        single = chunk_motion([[2]], [[1]], [1], [True])
        self.assertIsNone(single["predicted"]["rms_step_per_joint"])

    def test_boundary_uses_last_executed_action_and_aligned_overlap(self):
        previous = {"predicted": [[0], [1], [2], [3]], "target": [[0], [1], [2], [3]], "valid": [True] * 4}
        current = {
            "predicted": [[8], [9], [999], [999]],
            "target": [[2], [3], [999], [999]],
            "valid": [True, True, False, False],
        }
        result = replan_motion(previous, current, 2)
        self.assertEqual(result["predicted_boundary_delta_per_joint"], [7])
        self.assertEqual(result["reference_boundary_delta_per_joint"], [1])
        self.assertEqual(result["predicted_overlap_rms_per_joint"], [6])
        self.assertEqual(result["reference_overlap_rms_per_joint"], [0])
        self.assertEqual(result["overlap_frames"], 2)

    def test_invalid_anchor_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_motion([[1]], [[1]], [0], [False])
        with self.assertRaises(ValueError):
            chunk_motion([[np.inf]], [[1]], [0], [True])
        row = {"predicted": [[0], [1]], "target": [[0], [1]], "valid": [True, True]}
        for offset in (0, 2):
            with self.assertRaises(ValueError):
                replan_motion(row, row, offset)


if __name__ == "__main__":
    unittest.main()
