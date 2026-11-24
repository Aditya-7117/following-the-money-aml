"""Tests for the EDA helpers in src/analysis/ (data loading and statistics).

Run from the project root:
    python3 -m unittest discover -s . -p 'test_*.py'
They use small synthetic files only, so no real dataset is needed.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import load_eda_frame, load_timestamps, SPLIT_ORDER
from src.analysis.stats import chi_square_test, mann_whitney_test, rate_table, wilson_interval


def write_processed(folder, ids=(2, 1, 3, 4), splits=("train", "train", "val", "test_tail"), labels=(0, 1, 0, 1)):
    """Write tiny full_features / full_labels / full_split_assignments CSVs."""
    folder = Path(folder)
    pd.DataFrame({
        "transaction_id": ids,
        "log_amount_paid": [1.0, 2.0, 3.0, 4.0][: len(ids)],
        "payment_currency": ["Euro", "US Dollar", "Euro", "Euro"][: len(ids)],
        "payment_format": ["ACH", "Wire", "ACH", "ACH"][: len(ids)],
        "same_bank": [True, False, True, False][: len(ids)],
    }).to_csv(folder / "full_features.csv", index=False)
    pd.DataFrame({"transaction_id": ids, "is_laundering": labels}).to_csv(folder / "full_labels.csv", index=False)
    pd.DataFrame({"transaction_id": ids, "split": splits}).to_csv(folder / "full_split_assignments.csv", index=False)


class TestLoadEdaFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_returns_train_only(self):
        write_processed(self.dir)
        df = load_eda_frame(processed_dir=self.dir)
        self.assertEqual(set(df["split"]), {"train"})
        self.assertEqual(len(df), 2)

    def test_join_keeps_each_label_with_its_transaction(self):
        write_processed(self.dir)
        df = load_eda_frame(processed_dir=self.dir).set_index("transaction_id")
        self.assertEqual(df.loc[1, "is_laundering"], 1)
        self.assertEqual(df.loc[1, "payment_format"], "Wire")
        self.assertEqual(df.loc[2, "is_laundering"], 0)

    def test_all_splits_and_types(self):
        write_processed(self.dir)
        df = load_eda_frame(splits=SPLIT_ORDER, processed_dir=self.dir)
        self.assertEqual(len(df), 4)
        self.assertEqual(df["is_laundering"].dtype, np.int64)
        self.assertEqual(df["same_bank"].dtype, bool)
        self.assertEqual(list(df["split"].cat.categories), SPLIT_ORDER)

    def test_unknown_split_is_rejected(self):
        write_processed(self.dir)
        with self.assertRaises(ValueError):
            load_eda_frame(splits=("tset",), processed_dir=self.dir)

    def test_misaligned_files_are_rejected(self):
        write_processed(self.dir)
        pd.DataFrame({"transaction_id": [1, 2, 3, 4], "is_laundering": [1, 0, 0, 1]}).to_csv(
            self.dir / "full_labels.csv", index=False
        )
        with self.assertRaises(ValueError):
            load_eda_frame(processed_dir=self.dir)


class TestLoadTimestamps(unittest.TestCase):
    def test_ids_are_one_based_row_numbers_and_times_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            raw.write_text(
                "Timestamp,From Bank,Account,To Bank,Account\n"
                "2022/09/01 00:05,1,A,2,B\n"
                "2022/09/02 13:36,3,C,4,D\n"
            )
            ts = load_timestamps(raw)
        self.assertEqual(list(ts["transaction_id"]), [1, 2])
        self.assertEqual(ts.loc[1, "timestamp"], pd.Timestamp("2022-09-02 13:36"))


class TestWilsonInterval(unittest.TestCase):
    def test_known_value(self):
        low, high = wilson_interval([10], [100])
        self.assertAlmostEqual(low[0], 0.0552, places=4)
        self.assertAlmostEqual(high[0], 0.1744, places=4)

    def test_zero_successes_never_goes_below_zero(self):
        low, high = wilson_interval([0], [1000])
        self.assertGreaterEqual(low[0], 0.0)
        self.assertAlmostEqual(low[0], 0.0, places=12)
        self.assertGreater(high[0], 0.0)


class TestRateTable(unittest.TestCase):
    def test_counts_rates_share_and_lift(self):
        df = pd.DataFrame({
            "fmt": ["ACH"] * 10 + ["Wire"] * 10,
            "is_laundering": [1] * 4 + [0] * 6 + [1] * 1 + [0] * 9,
        })
        table = rate_table(df, "fmt")
        self.assertEqual(table.loc["ACH", "laundering"], 4)
        self.assertAlmostEqual(table.loc["ACH", "rate_%"], 40.0)
        self.assertAlmostEqual(table.loc["ACH", "share_of_laundering_%"], 80.0)
        self.assertAlmostEqual(table.loc["ACH", "lift"], 1.6)  # 40% vs overall 25%
        self.assertEqual(table.index[0], "ACH")  # highest rate first


class TestChiSquare(unittest.TestCase):
    def test_no_association_gives_v_near_zero(self):
        df = pd.DataFrame({"c": ["a", "b"] * 500, "is_laundering": [0, 0, 1, 1] * 250})
        result = chi_square_test(df, "c")
        self.assertAlmostEqual(result["cramers_v"], 0.0, places=3)
        self.assertGreater(result["p_value"], 0.05)

    def test_perfect_association_gives_v_near_one(self):
        df = pd.DataFrame({"c": ["a"] * 500 + ["b"] * 500, "is_laundering": [1] * 500 + [0] * 500})
        result = chi_square_test(df, "c")
        self.assertGreater(result["cramers_v"], 0.99)
        self.assertLess(result["p_value"], 0.001)


class TestMannWhitney(unittest.TestCase):
    def test_higher_laundering_values_give_auc_one(self):
        df = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12], "is_laundering": [0, 0, 0, 1, 1, 1]})
        result = mann_whitney_test(df, "x")
        self.assertAlmostEqual(result["auc"], 1.0)
        self.assertAlmostEqual(result["rank_biserial"], 1.0)

    def test_identical_groups_give_auc_half(self):
        df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3], "is_laundering": [0, 0, 0, 1, 1, 1]})
        self.assertAlmostEqual(mann_whitney_test(df, "x")["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()