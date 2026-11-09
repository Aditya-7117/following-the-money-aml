"""
tests/data/test_split.py
Unit tests for src/data/split.py.

All tests use synthetic boundary dictionaries so they do not depend on
config/data.json being present, except the two config-integration tests
which explicitly load the real config file.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.split import (
    assign_split,
    get_split_boundaries,
    get_feature_columns,
    load_config,
    SPLIT_LABELS,
)

# Canonical boundary dict matching config/data.json provisional values.
# Used in all non-config tests to keep them independent of the config file.
_BOUNDARIES = {
    "train_cutoff": "2022/09/06 13:36",
    "val_cutoff":   "2022/09/08 16:12",
    "tail_cutoff":  "2022/09/11 00:00",
}


class TestAssignSplit(unittest.TestCase):
    """assign_split returns the correct label for every boundary region."""

    # ------------------------------------------------------------------
    # Train region
    # ------------------------------------------------------------------

    def test_dataset_start_is_train(self):
        self.assertEqual(assign_split("2022/09/01 00:00", _BOUNDARIES), "train")

    def test_one_minute_before_train_cutoff_is_train(self):
        self.assertEqual(assign_split("2022/09/06 13:35", _BOUNDARIES), "train")

    def test_exact_train_cutoff_is_val(self):
        """The boundary itself belongs to val, not train."""
        self.assertEqual(assign_split("2022/09/06 13:36", _BOUNDARIES), "val")

    # ------------------------------------------------------------------
    # Val region
    # ------------------------------------------------------------------

    def test_one_minute_after_train_cutoff_is_val(self):
        self.assertEqual(assign_split("2022/09/06 13:37", _BOUNDARIES), "val")

    def test_one_minute_before_val_cutoff_is_val(self):
        self.assertEqual(assign_split("2022/09/08 16:11", _BOUNDARIES), "val")

    def test_exact_val_cutoff_is_test_main(self):
        """The boundary itself belongs to test_main, not val."""
        self.assertEqual(assign_split("2022/09/08 16:12", _BOUNDARIES), "test_main")

    # ------------------------------------------------------------------
    # test_main region
    # ------------------------------------------------------------------

    def test_one_minute_after_val_cutoff_is_test_main(self):
        self.assertEqual(assign_split("2022/09/08 16:13", _BOUNDARIES), "test_main")

    def test_one_minute_before_tail_cutoff_is_test_main(self):
        self.assertEqual(assign_split("2022/09/10 23:59", _BOUNDARIES), "test_main")

    def test_exact_tail_cutoff_is_test_tail(self):
        """The boundary itself belongs to test_tail."""
        self.assertEqual(assign_split("2022/09/11 00:00", _BOUNDARIES), "test_tail")

    # ------------------------------------------------------------------
    # test_tail region
    # ------------------------------------------------------------------

    def test_one_minute_after_tail_cutoff_is_test_tail(self):
        self.assertEqual(assign_split("2022/09/11 00:01", _BOUNDARIES), "test_tail")

    def test_dataset_end_is_test_tail(self):
        self.assertEqual(assign_split("2022/09/18 16:18", _BOUNDARIES), "test_tail")

    # ------------------------------------------------------------------
    # Validity and coverage
    # ------------------------------------------------------------------

    def test_all_returned_labels_are_valid(self):
        """Every call returns one of the four recognised split labels."""
        timestamps = [
            "2022/09/01 00:00",
            "2022/09/06 13:35",
            "2022/09/06 13:36",
            "2022/09/08 16:11",
            "2022/09/08 16:12",
            "2022/09/10 23:59",
            "2022/09/11 00:00",
            "2022/09/18 16:18",
        ]
        valid = set(SPLIT_LABELS)
        for ts in timestamps:
            with self.subTest(ts=ts):
                self.assertIn(assign_split(ts, _BOUNDARIES), valid)

    def test_disjoint_full_coverage(self):
        """Labels assigned to a list spanning all four regions are mutually
        exclusive and each region is covered exactly once."""
        region_samples = {
            "train":     "2022/09/01 00:00",
            "val":       "2022/09/07 00:00",
            "test_main": "2022/09/09 00:00",
            "test_tail": "2022/09/12 00:00",
        }
        assigned = {
            label: assign_split(ts, _BOUNDARIES)
            for label, ts in region_samples.items()
        }
        for expected, got in assigned.items():
            self.assertEqual(expected, got, msg=f"Expected {expected!r}, got {got!r}")
        # All four labels appear exactly once
        self.assertEqual(set(assigned.values()), set(SPLIT_LABELS))

    def test_invalid_timestamp_raises_value_error(self):
        """Short or empty timestamps should raise ValueError."""
        with self.assertRaises(ValueError):
            assign_split("", _BOUNDARIES)
        with self.assertRaises(ValueError):
            assign_split("2022/09/01", _BOUNDARIES)

    def test_test_main_and_test_tail_are_disjoint(self):
        """No single timestamp can be classified as both test_main and test_tail."""
        timestamps = [
            "2022/09/08 16:12",  # exact val_cutoff -> test_main
            "2022/09/10 23:59",  # one minute before tail -> test_main
            "2022/09/11 00:00",  # exact tail_cutoff -> test_tail
            "2022/09/11 00:01",  # after tail_cutoff -> test_tail
        ]
        for ts in timestamps:
            label = assign_split(ts, _BOUNDARIES)
            self.assertIn(label, {"test_main", "test_tail"})
            # Must be one or the other, never both (trivially true for a string)
            self.assertNotEqual(label, "train")
            self.assertNotEqual(label, "val")


class TestLoadConfig(unittest.TestCase):
    """load_config validates structure and raises on bad input."""

    def _write_config(self, tmp_dir: str, cfg: dict) -> Path:
        p = Path(tmp_dir) / "data.json"
        with p.open("w") as f:
            json.dump(cfg, f)
        return p

    def _valid_cfg(self) -> dict:
        return {
            "config_version": "1.0",
            "feature_list_version": "1.0",
            "feature_columns": ["log_amount_paid", "payment_format"],
            "rolling_window_rules": {"semantics": "open"},
            "split_boundaries": {
                "status": "provisional",
                "train_cutoff": "2022/09/06 13:36",
                "val_cutoff":   "2022/09/08 16:12",
                "tail_cutoff":  "2022/09/11 00:00",
            },
        }

    def test_valid_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._write_config(tmp, self._valid_cfg())
            cfg = load_config(cfg_path)
            self.assertIn("feature_columns", cfg)
            self.assertIn("split_boundaries", cfg)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config(Path("/nonexistent/path/data.json"))

    def test_missing_feature_columns_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._valid_cfg()
            del cfg["feature_columns"]
            path = self._write_config(tmp, cfg)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_missing_train_cutoff_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._valid_cfg()
            del cfg["split_boundaries"]["train_cutoff"]
            path = self._write_config(tmp, cfg)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_disordered_boundaries_raise_value_error(self):
        """val_cutoff before train_cutoff must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._valid_cfg()
            cfg["split_boundaries"]["val_cutoff"] = "2022/09/05 00:00"  # < train_cutoff
            path = self._write_config(tmp, cfg)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_empty_feature_columns_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._valid_cfg()
            cfg["feature_columns"] = []
            path = self._write_config(tmp, cfg)
            with self.assertRaises(ValueError):
                load_config(path)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests: real config/data.json is loaded and checked."""

    @classmethod
    def setUpClass(cls):
        # The real config lives two levels above this test file
        project_root = Path(__file__).resolve().parents[2]
        cls.config_path = project_root / "config" / "data.json"
        cls.cfg = load_config(cls.config_path)

    def test_boundaries_match_expected_provisional_values(self):
        sb = self.cfg["split_boundaries"]
        self.assertEqual(sb["train_cutoff"], "2022/09/06 13:36")
        self.assertEqual(sb["val_cutoff"],   "2022/09/08 16:12")
        self.assertEqual(sb["tail_cutoff"],  "2022/09/11 00:00")

    def test_feature_columns_match_expected_ten(self):
        from src.data.features import FEATURE_COLUMNS
        self.assertEqual(self.cfg["feature_columns"], FEATURE_COLUMNS)

    def test_get_split_boundaries_returns_correct_values(self):
        sb = get_split_boundaries(self.config_path)
        self.assertEqual(sb["train_cutoff"], "2022/09/06 13:36")

    def test_get_feature_columns_returns_list(self):
        fc = get_feature_columns(self.config_path)
        self.assertIsInstance(fc, list)
        self.assertEqual(len(fc), 10)

    def test_assign_split_with_config_boundaries(self):
        """assign_split called with config-read boundaries gives expected labels."""
        sb = get_split_boundaries(self.config_path)
        self.assertEqual(assign_split("2022/09/01 00:00", sb), "train")
        self.assertEqual(assign_split("2022/09/06 13:36", sb), "val")
        self.assertEqual(assign_split("2022/09/08 16:12", sb), "test_main")
        self.assertEqual(assign_split("2022/09/11 00:00", sb), "test_tail")

    def test_rolling_window_durations_match_features_constants(self):
        """Window duration seconds declared in config/data.json must match features.py."""
        from src.data.features import _ONE_HOUR, _TWENTY_FOUR_HOURS
        rw = self.cfg["rolling_window_rules"]
        self.assertEqual(rw["window_1h_seconds"], int(_ONE_HOUR.total_seconds()))
        self.assertEqual(rw["window_24h_seconds"], int(_TWENTY_FOUR_HOURS.total_seconds()))

    def test_timestamp_format_matches_features_constant(self):
        """Timestamp format declared in config/data.json must match features.py."""
        from src.data.features import TIMESTAMP_FORMAT
        rw = self.cfg["rolling_window_rules"]
        self.assertEqual(rw["timestamp_format"], TIMESTAMP_FORMAT)


if __name__ == "__main__":
    unittest.main()
