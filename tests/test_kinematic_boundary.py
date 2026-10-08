import unittest
from unittest.mock import patch

import numpy as np

from kinematic_boundary import (
    boundary_metrics,
    kinematic_eventness,
    local_peaks,
    refine_boundaries,
)


class BoundaryCoreTests(unittest.TestCase):
    def test_peak_rule_matches_lasa_strict_max_and_threshold(self):
        probability = np.array([0.9, 0.2, 0.8, 0.8, 0.1, 0.7, 0.0])
        self.assertEqual(local_peaks(probability, threshold=0.7), [0, 5])

    def test_missing_cue_or_qc_and_low_quality_return_baseline(self):
        probability = np.array([1.0, 0.1, 0.1, 0.95, 0.1, 0.1, 0.0])
        baseline = local_peaks(probability, threshold=0.7)
        speed = np.linspace(0, 1, len(probability))
        turn = np.zeros_like(speed)
        for args in [
            (probability, None, turn, 1.0),
            (probability, speed, turn, None),
            (probability, speed, turn, 0.2),
        ]:
            _, peaks, eventness = refine_boundaries(*args, threshold=0.7)
            self.assertEqual(peaks, baseline)
            self.assertTrue(np.array_equal(eventness, np.zeros_like(eventness)))

    def test_only_existing_model_candidates_can_move(self):
        n = 80
        t = np.arange(n)
        speed = np.r_[np.zeros(40), np.ones(40)]
        turn = np.zeros(n)
        probability = 0.01 + 0.95 * np.exp(-0.5 * ((t - 44) / 1.4) ** 2)
        # Strong motion event with no learned peak must not become a boundary.
        turn[65] = 20.0
        baseline = local_peaks(probability, threshold=0.7)
        _, peaks, _ = refine_boundaries(
            probability, speed, turn, 1.0, threshold=0.7,
            snap_radius=8, min_eventness=0.55, smooth_width=1,
        )
        self.assertEqual(len(peaks), len(baseline))
        self.assertTrue(set(peaks).issubset({0, *range(36, 53)}))

    def test_low_quality_fallback_does_not_require_finite_kinematics(self):
        probability = np.array([1.0, 0.0, 0.9, 0.0, 0.0])
        _, peaks, _ = refine_boundaries(
            probability, [np.nan] * 5, [np.nan] * 5, 0.1, threshold=0.7
        )
        self.assertEqual(peaks, local_peaks(probability, threshold=0.7))

    def test_short_sequence_and_frame_zero_are_not_kinematic_targets(self):
        probability = np.array([1.0, 0.0])
        _, peaks, eventness = refine_boundaries(
            probability, [0.0, 1.0], [0.0, 1.0], 1.0, threshold=0.7
        )
        self.assertEqual(peaks, [0])
        self.assertTrue(np.array_equal(eventness, np.zeros(2)))

    def test_two_candidates_snapped_to_one_event_are_deduplicated(self):
        probability = np.array([1.0] + [0.0] * 19 + [0.9, 0.0, 0.0, 0.0, 0.9] + [0.0] * 5)
        synthetic_event = np.zeros_like(probability)
        synthetic_event[22] = 1.0
        with patch("kinematic_boundary.kinematic_eventness", return_value=synthetic_event):
            _, peaks, _ = refine_boundaries(
                probability, np.zeros_like(probability), np.zeros_like(probability), 1.0,
                threshold=0.7, snap_radius=5, min_eventness=0.5,
            )
        self.assertEqual(peaks, [0, 22])

    def test_overestimated_qc_with_wrong_cue_can_worsen_a_boundary(self):
        n = 80
        t = np.arange(n)
        probability = np.full(n, 0.01)
        probability[44] = 0.95
        # Deliberately wrong synthetic cue produces a strong event at frame 38.
        speed = 1.0 / (1.0 + np.exp(-(t - 38.0)))
        turn = np.zeros(n)
        turn[38] = 20.0
        baseline = local_peaks(probability, threshold=0.7)
        _, guided, _ = refine_boundaries(
            probability, speed, turn, 1.0, threshold=0.7,
            snap_radius=10, min_eventness=0.4, smooth_width=1,
        )
        baseline_error = abs(baseline[1] - 44)
        guided_error = abs(guided[1] - 44)
        self.assertGreater(guided_error, baseline_error)

    def test_shape_errors_are_explicit_when_qc_would_enable_cues(self):
        with self.assertRaises(ValueError):
            kinematic_eventness([0, 1, 2], [0, 1])

    def test_sparse_speed_event_is_not_erased_by_percentile_scaling(self):
        speed = np.zeros(100)
        speed[50:] = 1.0
        turn = np.zeros_like(speed)
        turn[50] = 10.0
        eventness = kinematic_eventness(speed, turn, smooth_width=1)
        self.assertGreater(float(eventness[50]), 0.0)

    def test_metrics_match_truth_one_to_one(self):
        scores = boundary_metrics([0, 12, 13], [0, 10], tolerance=2)
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (2, 1, 0))
        self.assertEqual(scores["mean_matched_error_frames"], 1.0)


if __name__ == "__main__":
    unittest.main()
