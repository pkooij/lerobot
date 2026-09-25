"""Separate reduction overflow from nonfinite gradient elements."""

import json
import math
import unittest

import torch
from gradient_metrics import gradient_summary


class GradientMetricsTest(unittest.TestCase):
    def parameter(self, values):
        parameter = torch.nn.Parameter(torch.zeros(len(values)))
        parameter.grad = torch.tensor(values, dtype=torch.float32)
        return parameter

    def test_finite_norm_without_mutation(self):
        parameter = self.parameter([3, 4])
        result = gradient_summary([("weight", parameter)])
        self.assertEqual(result["norm_float64"], 5)
        self.assertEqual(result["nonfinite_elements"], 0)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))

    def test_float32_reduction_overflow_is_not_nonfinite_gradient(self):
        parameter = self.parameter([1e20, 1e20])
        result = gradient_summary([("weight", parameter)])
        self.assertTrue(math.isfinite(result["norm_float64"]))
        self.assertEqual(result["nonfinite_elements"], 0)
        native_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        self.assertTrue(torch.isinf(native_norm))
        self.assertEqual(torch.count_nonzero(parameter.grad), 0)

    def test_nonfinite_elements_remain_explicit_and_json_safe(self):
        result = gradient_summary([("bad", self.parameter([float("inf"), float("nan")]))])
        self.assertEqual(result["nonfinite_elements"], 2)
        self.assertEqual(result["nonfinite_parameters"][0]["parameter"], "bad")
        self.assertEqual(result["norm_float64"], "nan")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
