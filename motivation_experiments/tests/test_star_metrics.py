import math
import unittest

import numpy as np

from star_metrics import (
    aggregate_group_metrics,
    compute_group_metrics,
    deterministic_anchor_partition,
)


class StarMetricTests(unittest.TestCase):
    def test_symmetric_star_has_normalized_angle_two(self):
        # anchor at angle 0; leaves at +/- theta. Each leaf meets tau=cos(theta),
        # while their mutual angular distance is 2*theta.
        theta = math.radians(20.0)
        tau = math.cos(theta)
        keys = np.asarray(
            [
                [1.0, 0.0],
                [math.cos(theta), math.sin(theta)],
                [math.cos(theta), -math.sin(theta)],
            ],
            dtype=np.float32,
        )
        candidates = np.asarray([1, 2], dtype=np.int64)
        assignment = np.asarray([0, 0], dtype=np.int64)
        anchor_cos = np.asarray([tau, tau], dtype=np.float32)
        anchors = np.asarray([0], dtype=np.int64)
        rows = compute_group_metrics(keys, candidates, assignment, anchor_cos, anchors, tau, 1e-7)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].unsafe)
        self.assertAlmostEqual(rows[0].normalized_max_angle, 2.0, places=5)

    def test_safe_group(self):
        tau = math.cos(math.radians(30.0))
        a1 = math.radians(5.0)
        a2 = math.radians(15.0)
        keys = np.asarray(
            [
                [1.0, 0.0],
                [math.cos(a1), math.sin(a1)],
                [math.cos(a2), math.sin(a2)],
            ],
            dtype=np.float32,
        )
        candidates = np.asarray([1, 2], dtype=np.int64)
        assignment = np.asarray([0, 0], dtype=np.int64)
        anchor_cos = np.asarray([math.cos(a1), math.cos(a2)], dtype=np.float32)
        anchors = np.asarray([0], dtype=np.int64)
        rows = compute_group_metrics(keys, candidates, assignment, anchor_cos, anchors, tau, 1e-7)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].unsafe)
        self.assertLess(rows[0].normalized_max_angle, 1.0)

    def test_partition_is_deterministic_and_disjoint(self):
        a1, c1 = deterministic_anchor_partition(100, 0.4, 7)
        a2, c2 = deterministic_anchor_partition(100, 0.4, 7)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        self.assertEqual(len(a1), 40)
        self.assertEqual(len(c1), 60)
        self.assertEqual(len(np.intersect1d(a1, c1)), 0)
        self.assertEqual(len(np.union1d(a1, c1)), 100)

    def test_aggregation(self):
        theta = math.radians(20.0)
        tau = math.cos(theta)
        keys = np.asarray(
            [
                [1.0, 0.0],
                [math.cos(theta), math.sin(theta)],
                [math.cos(theta), -math.sin(theta)],
            ],
            dtype=np.float32,
        )
        rows = compute_group_metrics(
            keys,
            np.asarray([1, 2]),
            np.asarray([0, 0]),
            np.asarray([tau, tau], dtype=np.float32),
            np.asarray([0]),
            tau,
            1e-7,
        )
        agg = aggregate_group_metrics(rows)
        self.assertEqual(agg["eligible_groups"], 1)
        self.assertEqual(agg["unsafe_groups"], 1)
        self.assertAlmostEqual(agg["unsafe_group_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
