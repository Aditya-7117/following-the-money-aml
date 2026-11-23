"""
Tests for src/evaluation/metrics.py.
Every expected number below was worked out by hand.

Run:  python -m unittest discover -s tests/evaluation -t .
"""

import math
import unittest

import pandas as pd

from src.evaluation.metrics import (
    average_precision,
    choose_threshold,
    recall_at_budget,
    threshold_metrics,
)

# Small example used in several tests.
# score: 0.9  0.8  0.4  0.3  0.1
# label:  1    0    1    0    0
LABELS = [1, 0, 1, 0, 0]
SCORES = [0.9, 0.8, 0.4, 0.3, 0.1]


class TestThresholdMetrics(unittest.TestCase):

    def test_worked_example(self):
        # threshold 0.35 flags the top 3 rows: 2 laundering, 1 normal
        m = threshold_metrics(LABELS, SCORES, 0.35)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 0, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 0.8)
        self.assertAlmostEqual(m["alert_fraction"], 3 / 5)

    def test_measures_class_1(self):
        # only 1 laundering row; if class 0 was measured by mistake, tp would be 3
        m = threshold_metrics([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1], 0.5)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["tn"], 3)

    def test_model_that_flags_nothing(self):
        m = threshold_metrics(LABELS, SCORES, 2.0)
        self.assertTrue(math.isnan(m["precision"]))   # n/a, not 0
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)                # 2*TP / (2*TP + FP + FN) = 0
        self.assertEqual(m["alert_fraction"], 0.0)

    def test_model_that_flags_everything(self):
        m = threshold_metrics(LABELS, SCORES, 0.0)
        self.assertAlmostEqual(m["precision"], 2 / 5)  # equals the laundering rate
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["alert_fraction"], 1.0)

    def test_threshold_is_inclusive(self):
        # a score equal to the threshold is flagged
        m = threshold_metrics([1, 0], [0.5, 0.2], 0.5)
        self.assertEqual(m["tp"], 1)


class TestChooseThreshold(unittest.TestCase):

    def test_worked_example(self):
        # F1 at each threshold: 0.9->0.67, 0.8->0.5, 0.4->0.8, 0.3->0.67, 0.1->0.57
        self.assertEqual(choose_threshold(LABELS, SCORES), 0.4)

    def test_tie_keeps_higher_threshold(self):
        # 0.9 gives P=1.0, R=0.5 and 0.6 gives P=0.5, R=1.0: both F1 = 2/3
        self.assertEqual(choose_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6]), 0.9)

    def test_needs_laundering_rows(self):
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [0.3, 0.2])


class TestAveragePrecision(unittest.TestCase):

    def test_worked_example(self):
        # 0.9 is laundering: recall 0.5 at precision 1   -> 0.5 * 1
        # 0.4 is laundering: recall 1.0 at precision 2/3 -> 0.5 * 2/3
        self.assertAlmostEqual(average_precision(LABELS, SCORES), 0.5 + 1 / 3)

    def test_no_laundering_gives_nan(self):
        self.assertTrue(math.isnan(average_precision([0, 0], [0.4, 0.2])))


class TestRecallAtBudget(unittest.TestCase):

    def make_df(self, labels, scores, ids):
        return pd.DataFrame({"transaction_id": ids, "is_laundering": labels, "score": scores})

    def test_worked_example(self):
        # budget 20% of 5 rows = 1 alert, given to the 0.9 row (laundering)
        r = recall_at_budget(self.make_df(LABELS, SCORES, [10, 11, 12, 13, 14]), budget=0.2)
        self.assertEqual(r["alerts_allowed"], 1)
        self.assertEqual(r["caught"], 1)
        self.assertAlmostEqual(r["recall_at_budget"], 0.5)

    def test_budget_rounds_up(self):
        df = self.make_df([0] * 101, [0.5] * 101, range(101))
        self.assertEqual(recall_at_budget(df, 0.01)["alerts_allowed"], 2)   # 1.01 -> 2
        df = self.make_df([0] * 100, [0.5] * 100, range(100))
        self.assertEqual(recall_at_budget(df, 0.07)["alerts_allowed"], 7)   # not 8

    def test_ties_go_to_smallest_id(self):
        # all scores equal, so only the ID decides: ID 1 gets the alert
        df = self.make_df([0, 0, 0, 0, 1], [0.0] * 5, [5, 4, 3, 2, 1])
        self.assertEqual(recall_at_budget(df, 0.2)["caught"], 1)

    def test_labels_do_not_choose_the_alerts(self):
        # same scores and IDs, laundering moved: ID 1 is still alerted, so 0 caught
        df = self.make_df([1, 0, 0, 0, 0], [0.0] * 5, [5, 4, 3, 2, 1])
        self.assertEqual(recall_at_budget(df, 0.2)["caught"], 0)


if __name__ == "__main__":
    unittest.main()
