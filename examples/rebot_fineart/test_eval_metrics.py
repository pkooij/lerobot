"""Regression checks for temporal coverage and masked metric denominators."""

import unittest

from eval_metrics import component_summary, episode_indices


class EvalMetricsTest(unittest.TestCase):
    def test_samples_cover_each_real_dev_episode(self):
        for length in (1942, 1837, 1177, 786, 822):
            ids = episode_indices(length, 50)
            self.assertEqual(len(set(ids)), 50)
            self.assertLess(ids[0], length * 0.02)
            self.assertGreater(ids[-1], length * 0.98)
            self.assertTrue(all(0 <= index < length for index in ids))
            self.assertTrue(all(a < b for a, b in zip(ids, ids[1:], strict=False)))

    def test_short_episode_is_not_duplicated(self):
        self.assertEqual(episode_indices(3, 50), [0, 1, 2])
        with self.assertRaises(ValueError):
            episode_indices(0, 50)

    def test_inactive_zero_loss_does_not_dilute_component(self):
        rows = [
            {"branch": "action", "losses": {"flow_loss": 2.0, "fast_action_loss": 4.0, "text_loss": 0.0}},
            {"branch": "text", "losses": {"text_loss": 6.0}},
        ]
        result = component_summary(rows)
        self.assertEqual(result["text_loss"], {"frames": 1, "mean_per_frame": 6.0})
        self.assertEqual(result["flow_loss"], {"frames": 1, "mean_per_frame": 2.0})
        self.assertIsNone(component_summary(rows[:1])["text_loss"]["mean_per_frame"])
        with self.assertRaises(ValueError):
            component_summary([{"branch": "text", "losses": {"text_loss": float("nan")}}])


if __name__ == "__main__":
    unittest.main()
