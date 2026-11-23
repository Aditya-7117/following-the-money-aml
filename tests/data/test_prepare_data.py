"""
tests/data/test_prepare_data.py
Unit tests for data preparation logic in scripts/prepare_data.py.

All tests use synthetic input files and do not depend on the
real 5-million-row raw dataset.
"""

import csv
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_data import (
    select_earliest_transactions,
    validate_prepared_data,
    save_sample_outputs,
    prepare_sample,
    prepare_full,
    get_peak_memory_mb,
    print_summary,
    main,
    FEATURE_COLUMNS,
)
from src.data.load import RAW_COLUMNS


def _create_synthetic_csv(file_path: Path, rows: list[list[str]]) -> None:
    """Helper to write a synthetic CSV with the exact RAW_COLUMNS header."""
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)


class TestSelectEarliestTransactions(unittest.TestCase):
    """Memory-bounded selection must sort chronologically with tie-breaking."""

    def test_selection_sort_and_tie_breaking(self):
        # Unsorted input with timestamps and tie-breaking IDs
        txns = [
            {"transaction_id": 10, "timestamp": "2022/09/01 00:05"},
            {"transaction_id": 5, "timestamp": "2022/09/01 00:01"},
            {"transaction_id": 2, "timestamp": "2022/09/01 00:01"},  # tie with id 5, smaller id
            {"transaction_id": 8, "timestamp": "2022/09/01 00:03"},
            {"transaction_id": 1, "timestamp": "2022/09/01 00:00"},
            {"transaction_id": 9, "timestamp": "2022/09/01 00:04"},
        ]

        # Select earliest 3
        selected = select_earliest_transactions(txns, n=3)
        self.assertEqual(len(selected), 3)

        # Expected earliest 3 in order:
        # 1. id=1 at 00:00
        # 2. id=2 at 00:01 (breaks tie with id=5)
        # 3. id=5 at 00:01
        self.assertEqual([t["transaction_id"] for t in selected], [1, 2, 5])
        self.assertEqual([t["timestamp"] for t in selected], [
            "2022/09/01 00:00",
            "2022/09/01 00:01",
            "2022/09/01 00:01",
        ])

    def test_selection_fewer_than_n(self):
        txns = [
            {"transaction_id": 2, "timestamp": "2022/09/01 00:02"},
            {"transaction_id": 1, "timestamp": "2022/09/01 00:01"},
        ]
        selected = select_earliest_transactions(txns, n=5)
        self.assertEqual(len(selected), 2)
        self.assertEqual([t["transaction_id"] for t in selected], [1, 2])


class TestValidatePreparedData(unittest.TestCase):
    """Validation checks on prepared feature and label outputs."""

    def _sample_valid_row(self, tid: int = 1) -> dict:
        row = {"transaction_id": tid}
        for feat in FEATURE_COLUMNS:
            if feat.endswith("_count_1h") or feat.endswith("_count_24h") or feat.endswith("_24h"):
                row[feat] = 0
            elif feat == "log_amount_paid":
                row[feat] = 4.6151
            elif feat == "same_bank":
                row[feat] = True
            else:
                row[feat] = "US Dollar"
        return row

    def test_valid_data_passes(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(2)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 2, "is_laundering": "1"},
        ]
        res = validate_prepared_data(f_rows, l_rows)
        self.assertEqual(res["duplicate_id_count"], 0)
        self.assertEqual(res["unique_id_count"], 2)

    def test_duplicate_transaction_id_fails(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(1)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 1, "is_laundering": "0"},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data(f_rows, l_rows)
        self.assertIn("duplicate transaction IDs", str(ctx.exception))

    def test_non_finite_numeric_fails(self):
        f_row = self._sample_valid_row(1)
        f_row["log_amount_paid"] = float("nan")
        l_rows = [{"transaction_id": 1, "is_laundering": "0"}]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data([f_row], l_rows)
        self.assertIn("Non-finite value", str(ctx.exception))

    def test_infinite_numeric_fails(self):
        f_row = self._sample_valid_row(1)
        f_row["log_amount_paid"] = float("inf")
        l_rows = [{"transaction_id": 1, "is_laundering": "0"}]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data([f_row], l_rows)
        self.assertIn("Non-finite value", str(ctx.exception))

    def test_negative_count_fails(self):
        f_row = self._sample_valid_row(1)
        f_row["sender_txn_count_1h"] = -1
        l_rows = [{"transaction_id": 1, "is_laundering": "0"}]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data([f_row], l_rows)
        self.assertIn("Negative count", str(ctx.exception))

    def test_id_mismatch_between_features_and_labels_fails(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(2)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 3, "is_laundering": "0"},  # mismatched id 3 vs 2
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data(f_rows, l_rows)
        self.assertIn("Transaction ID mismatch", str(ctx.exception))

    def test_valid_split_rows_passes(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(2)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 2, "is_laundering": "1"},
        ]
        s_rows = [
            {"transaction_id": 1, "split": "train"},
            {"transaction_id": 2, "split": "val"},
        ]
        res = validate_prepared_data(f_rows, l_rows, s_rows)
        self.assertEqual(res["duplicate_id_count"], 0)
        self.assertEqual(res["unique_id_count"], 2)

    def test_split_count_mismatch_fails(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(2)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 2, "is_laundering": "0"},
        ]
        s_rows = [{"transaction_id": 1, "split": "train"}]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data(f_rows, l_rows, s_rows)
        self.assertIn("Row count mismatch", str(ctx.exception))

    def test_split_id_mismatch_fails(self):
        f_rows = [self._sample_valid_row(1), self._sample_valid_row(2)]
        l_rows = [
            {"transaction_id": 1, "is_laundering": "0"},
            {"transaction_id": 2, "is_laundering": "0"},
        ]
        s_rows = [
            {"transaction_id": 1, "split": "train"},
            {"transaction_id": 99, "split": "train"},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data(f_rows, l_rows, s_rows)
        self.assertIn("Transaction ID mismatch between features", str(ctx.exception))

    def test_invalid_split_label_fails(self):
        f_rows = [self._sample_valid_row(1)]
        l_rows = [{"transaction_id": 1, "is_laundering": "0"}]
        s_rows = [{"transaction_id": 1, "split": "invalid_split"}]
        with self.assertRaises(ValueError) as ctx:
            validate_prepared_data(f_rows, l_rows, s_rows)
        self.assertIn("Invalid split label", str(ctx.exception))


class TestPrepareSampleEndToEnd(unittest.TestCase):
    """End-to-end sample preparation test on synthetic CSV input."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.csv_path = self.work_dir / "raw_synthetic.csv"
        self.out_dir = self.work_dir / "processed"

        # 8 synthetic rows, out of chronological order
        synthetic_rows = [
            # Row 1 (will have tid=1): 00:04, legit
            ["2022/09/01 00:04", "010", "A1", "010", "A1", "100.0", "US Dollar", "100.0", "US Dollar", "Cheque", "0"],
            # Row 2 (will have tid=2): 00:01, laundering
            ["2022/09/01 00:01", "001", "A2", "002", "B2", "250.0", "US Dollar", "250.0", "US Dollar", "Wire", "1"],
            # Row 3 (will have tid=3): 00:00, legit
            ["2022/09/01 00:00", "012", "A3", "012", "A3", "50.0", "US Dollar", "50.0", "US Dollar", "Credit Card", "0"],
            # Row 4 (will have tid=4): 00:01, legit (tie with row 2 at 00:01, but tid 4 > tid 2)
            ["2022/09/01 00:01", "001", "A4", "003", "B4", "300.0", "US Dollar", "300.0", "US Dollar", "Wire", "0"],
            # Row 5 (will have tid=5): 00:02, legit
            ["2022/09/01 00:02", "002", "A5", "002", "A5", "75.0", "US Dollar", "75.0", "US Dollar", "ACH", "0"],
            # Row 6 (will have tid=6): 00:03, laundering
            ["2022/09/01 00:03", "001", "A6", "004", "B6", "500.0", "US Dollar", "500.0", "US Dollar", "Wire", "1"],
            # Row 7 (will have tid=7): 00:02, legit (tie with row 5 at 00:02, but tid 7 > tid 5)
            ["2022/09/01 00:02", "003", "A7", "005", "B7", "120.0", "US Dollar", "120.0", "US Dollar", "Cheque", "0"],
            # Row 8 (will have tid=8): 00:05, legit
            ["2022/09/01 00:05", "001", "A8", "001", "A8", "10.0", "US Dollar", "10.0", "US Dollar", "Cash", "0"],
        ]
        _create_synthetic_csv(self.csv_path, synthetic_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_sample_selection_and_files(self):
        # Select sample of 5 earliest rows
        # Chronological order of all 8 rows:
        # 1. 00:00 (tid=3)
        # 2. 00:01 (tid=2)
        # 3. 00:01 (tid=4)
        # 4. 00:02 (tid=5)
        # 5. 00:02 (tid=7)
        # --- cut off here for n=5 ---
        # 6. 00:03 (tid=6)
        # 7. 00:04 (tid=1)
        # 8. 00:05 (tid=8)
        summary = prepare_sample(self.csv_path, self.out_dir, sample_size=5)

        self.assertEqual(summary["processed_count"], 5)
        self.assertEqual(summary["earliest_timestamp"], "2022/09/01 00:00")
        self.assertEqual(summary["latest_timestamp"], "2022/09/01 00:02")
        self.assertEqual(summary["legitimate_count"], 4)
        self.assertEqual(summary["laundering_count"], 1)  # tid=2 is laundering
        self.assertEqual(summary["feature_count"], 10)
        self.assertEqual(summary["feature_names"], FEATURE_COLUMNS)
        self.assertEqual(summary["duplicate_id_count"], 0)

        # Check output files exist
        features_path = self.out_dir / "sample_features.csv"
        labels_path = self.out_dir / "sample_labels.csv"
        splits_path = self.out_dir / "sample_split_assignments.csv"
        self.assertTrue(features_path.exists())
        self.assertTrue(labels_path.exists())
        self.assertTrue(splits_path.exists())

        # Inspect features CSV
        with features_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["transaction_id"] + FEATURE_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            # Verify chronological order of transaction IDs: 3, 2, 4, 5, 7
            feat_ids = [int(r[0]) for r in rows]
            self.assertEqual(feat_ids, [3, 2, 4, 5, 7])

        # Inspect labels CSV
        with labels_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["transaction_id", "is_laundering"])
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            label_ids = [int(r[0]) for r in rows]
            labels = [r[1] for r in rows]
            self.assertEqual(label_ids, [3, 2, 4, 5, 7])
            self.assertEqual(labels, ["0", "1", "0", "0", "0"])

        # Inspect split assignments CSV
        with splits_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["transaction_id", "split"])
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            split_ids = [int(r[0]) for r in rows]
            assigned_splits = [r[1] for r in rows]
            self.assertEqual(split_ids, [3, 2, 4, 5, 7])
            # All 5 transactions are on 2022/09/01 (before 2022/09/06 13:36) -> train
            self.assertEqual(assigned_splits, ["train", "train", "train", "train", "train"])

    def test_cli_sample_execution(self):
        """CLI main() with --sample flag runs cleanly and prints summary."""
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            main([
                "--sample",
                "--data-path", str(self.csv_path),
                "--output-dir", str(self.out_dir),
                "--sample-size", "4",
            ])

        output = stdout_capture.getvalue()
        self.assertIn("AML Sample Preparation Summary", output)
        self.assertIn("Number of processed transactions: 4", output)
        self.assertIn("Duplicate transaction_id count:  0", output)
        self.assertTrue((self.out_dir / "sample_features.csv").exists())
        self.assertTrue((self.out_dir / "sample_labels.csv").exists())
        self.assertTrue((self.out_dir / "sample_split_assignments.csv").exists())

    def test_cli_requires_sample_flag(self):
        """Omitting --sample should display help and exit with status 1."""
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            with self.assertRaises(SystemExit) as ctx:
                main([])
        self.assertEqual(ctx.exception.code, 1)


class TestExternalSort(unittest.TestCase):
    """Bounded-memory external sort correctness on synthetic transactions."""

    def test_external_sort_unsorted_chunks(self):
        txns = [
            {"transaction_id": 10, "timestamp": "2022/09/01 00:05", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 5, "timestamp": "2022/09/01 00:01", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 2, "timestamp": "2022/09/01 00:01", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 1, "timestamp": "2022/09/01 00:00", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 9, "timestamp": "2022/09/01 00:04", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 8, "timestamp": "2022/09/01 00:03", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
        ]

        from scripts.prepare_data import sort_transactions_external
        with tempfile.TemporaryDirectory() as td:
            # Chunk size 2 forces 3 separate chunk files + merge
            sorted_txns = list(sort_transactions_external(txns, Path(td), chunk_size=2))

            # Expected order:
            # ts 00:00 (id 1)
            # ts 00:01 (id 2) - tie broken by id
            # ts 00:01 (id 5)
            # ts 00:03 (id 8)
            # ts 00:04 (id 9)
            # ts 00:05 (id 10)
            self.assertEqual([t["transaction_id"] for t in sorted_txns], [1, 2, 5, 8, 9, 10])
            self.assertEqual([t["timestamp"] for t in sorted_txns], [
                "2022/09/01 00:00",
                "2022/09/01 00:01",
                "2022/09/01 00:01",
                "2022/09/01 00:03",
                "2022/09/01 00:04",
                "2022/09/01 00:05",
            ])

            # Check that temporary chunk files were cleaned up
            remaining_csvs = list(Path(td).glob("*.csv"))
            self.assertEqual(len(remaining_csvs), 0)

    def test_external_sort_timestamp_ties(self):
        # 4 txns with identical timestamps
        from scripts.prepare_data import sort_transactions_external
        txns = [
            {"transaction_id": 40, "timestamp": "2022/09/01 00:00", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 10, "timestamp": "2022/09/01 00:00", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 30, "timestamp": "2022/09/01 00:00", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
            {"transaction_id": 20, "timestamp": "2022/09/01 00:00", "sender_bank": "1", "sender_account": "A", "receiver_bank": "2", "receiver_account": "B", "amount_received": "1", "receiving_currency": "USD", "amount_paid": "1", "payment_currency": "USD", "payment_format": "ACH", "is_laundering": "0"},
        ]
        with tempfile.TemporaryDirectory() as td:
            sorted_txns = list(sort_transactions_external(txns, Path(td), chunk_size=2))
            self.assertEqual([t["transaction_id"] for t in sorted_txns], [10, 20, 30, 40])


class TestPrepareFullEndToEnd(unittest.TestCase):
    """End-to-end full preparation pipeline on synthetic CSV input."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.csv_path = self.work_dir / "synthetic_full.csv"
        self.out_dir = self.work_dir / "processed"

        # 10 synthetic rows spanning different dates/times out of order
        synthetic_rows = [
            # Row 1 (tid=1): 2022/09/07 10:00 (val)
            ["2022/09/07 10:00", "010", "A1", "010", "A1", "100.0", "US Dollar", "100.0", "US Dollar", "Cheque", "0"],
            # Row 2 (tid=2): 2022/09/01 00:01 (train), laundering
            ["2022/09/01 00:01", "001", "A2", "002", "B2", "250.0", "US Dollar", "250.0", "US Dollar", "Wire", "1"],
            # Row 3 (tid=3): 2022/09/01 00:00 (train)
            ["2022/09/01 00:00", "012", "A3", "012", "A3", "50.0", "US Dollar", "50.0", "US Dollar", "Credit Card", "0"],
            # Row 4 (tid=4): 2022/09/12 09:00 (test_tail), laundering
            ["2022/09/12 09:00", "001", "A4", "003", "B4", "300.0", "US Dollar", "300.0", "US Dollar", "ACH", "1"],
            # Row 5 (tid=5): 2022/09/01 00:01 (train), tie with row 2
            ["2022/09/01 00:01", "002", "A5", "002", "A5", "75.0", "US Dollar", "75.0", "US Dollar", "ACH", "0"],
            # Row 6 (tid=6): 2022/09/09 14:00 (test_main)
            ["2022/09/09 14:00", "001", "A6", "004", "B6", "500.0", "US Dollar", "500.0", "US Dollar", "Wire", "0"],
            # Row 7 (tid=7): 2022/09/05 12:00 (train)
            ["2022/09/05 12:00", "003", "A7", "005", "B7", "120.0", "US Dollar", "120.0", "US Dollar", "Cheque", "0"],
            # Row 8 (tid=8): 2022/09/10 18:00 (test_main)
            ["2022/09/10 18:00", "001", "A8", "001", "A8", "10.0", "US Dollar", "10.0", "US Dollar", "Cash", "0"],
            # Row 9 (tid=9): 2022/09/08 10:00 (val)
            ["2022/09/08 10:00", "004", "A9", "004", "A9", "80.0", "US Dollar", "80.0", "US Dollar", "ACH", "0"],
            # Row 10 (tid=10): 2022/09/15 12:00 (test_tail)
            ["2022/09/15 12:00", "005", "A10", "005", "A10", "45.0", "US Dollar", "45.0", "US Dollar", "Credit Card", "0"],
        ]
        _create_synthetic_csv(self.csv_path, synthetic_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_full_execution_and_manifest(self):
        from scripts.prepare_data import prepare_full
        test_artifacts_dir = self.work_dir / "artifacts"
        summary = prepare_full(
            self.csv_path,
            self.out_dir,
            chunk_size=3,
            artifacts_dir=test_artifacts_dir,
        )

        self.assertEqual(summary["processed_count"], 10)
        self.assertEqual(summary["earliest_timestamp"], "2022/09/01 00:00")
        self.assertEqual(summary["latest_timestamp"], "2022/09/15 12:00")
        self.assertEqual(summary["legitimate_count"], 8)
        self.assertEqual(summary["laundering_count"], 2)
        self.assertEqual(summary["duplicate_id_count"], 0)

        # Check output files exist
        features_path = self.out_dir / "full_features.csv"
        labels_path = self.out_dir / "full_labels.csv"
        splits_path = self.out_dir / "full_split_assignments.csv"
        manifest_path = self.out_dir / "manifest.json"
        checks_path = test_artifacts_dir / "data_checks.json"
        self.assertTrue(features_path.exists())
        self.assertTrue(labels_path.exists())
        self.assertTrue(splits_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(checks_path.exists())

        # Verify manifest structure
        import json
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["data_summary"]["total_transactions"], 10)
        self.assertEqual(manifest["data_summary"]["laundering_count"], 2)
        self.assertEqual(manifest["schema"]["feature_count"], 10)
        self.assertIn("candidate_splits", manifest)
        self.assertIn("execution_metadata", manifest)

        # Verify output hashes are present and valid in manifest
        self.assertIn("features_sha256", manifest["outputs"])
        self.assertIn("labels_sha256", manifest["outputs"])
        self.assertIn("split_assignments_sha256", manifest["outputs"])
        self.assertEqual(len(manifest["outputs"]["features_sha256"]), 64)
        self.assertEqual(len(manifest["outputs"]["labels_sha256"]), 64)
        self.assertEqual(len(manifest["outputs"]["split_assignments_sha256"]), 64)

        # Verify data_checks.json content
        with checks_path.open("r", encoding="utf-8") as f:
            checks = json.load(f)
        self.assertEqual(checks["row_counts"]["total_transactions"], 10)
        self.assertEqual(checks["row_counts"]["legitimate_count"], 8)
        self.assertEqual(checks["row_counts"]["laundering_count"], 2)
        self.assertEqual(checks["split_counts"]["train"]["total"], 4)
        self.assertEqual(checks["split_counts"]["val"]["total"], 2)
        self.assertEqual(checks["split_counts"]["test_main"]["total"], 2)
        self.assertEqual(checks["split_counts"]["test_tail"]["total"], 2)
        self.assertIn("checks_passed", checks)
        self.assertIn("features_sha256", checks["output_files"])
        self.assertIn("labels_sha256", checks["output_files"])
        self.assertIn("split_assignments_sha256", checks["output_files"])

        # Verify row alignment between features, labels, and split assignments
        with features_path.open("r", encoding="utf-8") as f_in, \
             labels_path.open("r", encoding="utf-8") as l_in, \
             splits_path.open("r", encoding="utf-8") as s_in:
            f_reader = csv.reader(f_in)
            l_reader = csv.reader(l_in)
            s_reader = csv.reader(s_in)

            f_header = next(f_reader)
            l_header = next(l_reader)
            s_header = next(s_reader)
            self.assertEqual(f_header, ["transaction_id"] + FEATURE_COLUMNS)
            self.assertEqual(l_header, ["transaction_id", "is_laundering"])
            self.assertEqual(s_header, ["transaction_id", "split"])

            f_rows = list(f_reader)
            l_rows = list(l_reader)
            s_rows = list(s_reader)
            self.assertEqual(len(f_rows), 10)
            self.assertEqual(len(l_rows), 10)
            self.assertEqual(len(s_rows), 10)

            # Ensure 1-to-1 transaction_id alignment in chronological order
            # Order of timestamps:
            # 1. 09/01 00:00 (tid 3) -> train
            # 2. 09/01 00:01 (tid 2) -> train
            # 3. 09/01 00:01 (tid 5) -> train
            # 4. 09/05 12:00 (tid 7) -> train
            # 5. 09/07 10:00 (tid 1) -> val
            # 6. 09/08 10:00 (tid 9) -> val
            # 7. 09/09 14:00 (tid 6) -> test_main
            # 8. 09/10 18:00 (tid 8) -> test_main
            # 9. 09/12 09:00 (tid 4) -> test_tail
            # 10. 09/15 12:00 (tid 10) -> test_tail
            expected_ids = [3, 2, 5, 7, 1, 9, 6, 8, 4, 10]
            expected_splits = [
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "test_main",
                "test_main",
                "test_tail",
                "test_tail",
            ]
            f_ids = [int(r[0]) for r in f_rows]
            l_ids = [int(r[0]) for r in l_rows]
            s_ids = [int(r[0]) for r in s_rows]
            actual_splits = [r[1] for r in s_rows]
            self.assertEqual(f_ids, expected_ids)
            self.assertEqual(l_ids, expected_ids)
            self.assertEqual(s_ids, expected_ids)
            self.assertEqual(actual_splits, expected_splits)

            # Check that feature columns do NOT contain forbidden identifiers
            forbidden = {"timestamp", "sender_bank", "sender_account", "receiver_bank", "receiver_account", "amount_received", "receiving_currency", "is_laundering"}
            for col in f_header[1:]:
                self.assertNotIn(col, forbidden)

    def test_cli_full_execution(self):
        """CLI main() with --full flag runs cleanly and outputs manifest."""
        stdout_capture = io.StringIO()
        cli_artifacts_dir = self.work_dir / "cli_artifacts"
        with patch("sys.stdout", stdout_capture):
            main([
                "--full",
                "--data-path", str(self.csv_path),
                "--output-dir", str(self.out_dir),
                "--artifacts-dir", str(cli_artifacts_dir),
                "--chunk-size", "3",
            ])
        output = stdout_capture.getvalue()
        self.assertIn("AML Full Data Preparation Summary", output)
        self.assertIn("Number of processed transactions: 10", output)
        self.assertTrue((self.out_dir / "full_features.csv").exists())
        self.assertTrue((self.out_dir / "full_labels.csv").exists())
        self.assertTrue((self.out_dir / "full_split_assignments.csv").exists())
        self.assertTrue((self.out_dir / "manifest.json").exists())
        self.assertTrue((cli_artifacts_dir / "data_checks.json").exists())

    def test_full_reproducibility(self):
        """Two independent runs produce byte-identical feature and label outputs."""
        from scripts.prepare_data import prepare_full
        out1 = self.work_dir / "out1"
        out2 = self.work_dir / "out2"
        art1 = self.work_dir / "art1"
        art2 = self.work_dir / "art2"

        prepare_full(self.csv_path, out1, chunk_size=3, artifacts_dir=art1)
        prepare_full(self.csv_path, out2, chunk_size=3, artifacts_dir=art2)

        with open(out1 / "full_features.csv", "rb") as f1, open(out2 / "full_features.csv", "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

        with open(out1 / "full_labels.csv", "rb") as l1, open(out2 / "full_labels.csv", "rb") as l2:
            self.assertEqual(l1.read(), l2.read())

        with open(out1 / "full_split_assignments.csv", "rb") as s1, open(out2 / "full_split_assignments.csv", "rb") as s2:
            self.assertEqual(s1.read(), s2.read())

    def test_failure_handling_cleans_tmp_files(self):
        """If a processing error occurs, temporary files are safely deleted."""
        from scripts.prepare_data import prepare_full
        # Introduce a broken generator by mocking generate_features to fail midway
        with patch("scripts.prepare_data.generate_features", side_effect=RuntimeError("Simulated pipeline crash")):
            with self.assertRaises(RuntimeError):
                prepare_full(self.csv_path, self.out_dir, chunk_size=3)

        # Confirm no .tmp files exist
        tmp_files = list(self.out_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)
        # And final files were not created
        self.assertFalse((self.out_dir / "full_features.csv").exists())
        self.assertFalse((self.out_dir / "full_labels.csv").exists())
        self.assertFalse((self.out_dir / "full_split_assignments.csv").exists())

    def test_cli_rejects_both_sample_and_full(self):
        """Passing both --sample and --full must raise SystemExit(1)."""
        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            with self.assertRaises(SystemExit) as ctx:
                main(["--sample", "--full"])
        self.assertEqual(ctx.exception.code, 1)


class TestWindowsResourceCompatibility(unittest.TestCase):
    """Verify graceful fallback when the Unix 'resource' module is unavailable (e.g. on Windows)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.csv_path = self.work_dir / "synthetic_raw.csv"
        self.out_dir = self.work_dir / "processed"
        self.art_dir = self.work_dir / "artifacts"

        # Synthetic 5-row dataset with valid RAW_COLUMNS
        rows = [
            ["2022/09/01 00:01", "10", "100", "20", "200", "100.0", "US Dollar", "100.0", "US Dollar", "Cheque", "0"],
            ["2022/09/01 00:02", "10", "100", "30", "300", "250.0", "US Dollar", "250.0", "US Dollar", "ACH", "0"],
            ["2022/09/02 10:00", "20", "200", "10", "100", "50.0", "US Dollar", "50.0", "US Dollar", "Credit Card", "1"],
            ["2022/09/07 12:00", "30", "300", "40", "400", "300.0", "US Dollar", "300.0", "US Dollar", "ACH", "0"],
            ["2022/09/12 09:00", "40", "400", "50", "500", "400.0", "US Dollar", "400.0", "US Dollar", "ACH", "1"],
        ]
        _create_synthetic_csv(self.csv_path, rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_peak_memory_mb_returns_none_when_resource_is_none(self):
        """When resource is None (as on native Windows), get_peak_memory_mb returns None."""
        with patch("scripts.prepare_data.resource", None):
            self.assertIsNone(get_peak_memory_mb())

    def test_get_peak_memory_mb_handles_attribute_or_value_error(self):
        """When resource lacks getrusage or raises ValueError/AttributeError, return None."""
        class DummyResource:
            def getrusage(self, *args):
                raise AttributeError("no getrusage on this platform")

        with patch("scripts.prepare_data.resource", DummyResource()):
            self.assertIsNone(get_peak_memory_mb())

    def test_print_summary_displays_na_when_memory_is_none(self):
        """Terminal summary outputs 'N/A' when peak memory is None for both modes."""
        base_summary = {
            "processed_count": 100,
            "earliest_timestamp": "2022/09/01 00:00",
            "latest_timestamp": "2022/09/01 01:00",
            "legitimate_count": 99,
            "laundering_count": 1,
            "feature_count": 10,
            "feature_names": FEATURE_COLUMNS,
            "duplicate_id_count": 0,
            "elapsed_seconds": 1.23,
            "peak_memory_mb": None,
            "features_path": Path("dummy_features.csv"),
            "labels_path": Path("dummy_labels.csv"),
        }

        # Sample mode
        sample_sum = dict(base_summary, mode="sample")
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            print_summary(sample_sum)
        output = stdout_capture.getvalue()
        self.assertIn("Peak memory:                     N/A", output)

        # Full mode
        full_sum = dict(
            base_summary,
            mode="full",
            raw_sha256="abc",
            features_sha256="def",
            labels_sha256="123",
            split_assignments_sha256="456",
            manifest_path=Path("dummy_manifest.json"),
            split_assignments_path=Path("dummy_splits.csv"),
        )
        stdout_capture_full = io.StringIO()
        with patch("sys.stdout", stdout_capture_full):
            print_summary(full_sum)
        output_full = stdout_capture_full.getvalue()
        self.assertIn("Peak memory:                     N/A", output_full)

    def test_print_summary_displays_formatted_mb_when_memory_present(self):
        """Terminal summary formats peak memory as xx.xx MB when present."""
        summary = {
            "mode": "sample",
            "processed_count": 50,
            "earliest_timestamp": "2022/09/01 00:00",
            "latest_timestamp": "2022/09/01 01:00",
            "legitimate_count": 50,
            "laundering_count": 0,
            "feature_count": 10,
            "feature_names": FEATURE_COLUMNS,
            "duplicate_id_count": 0,
            "elapsed_seconds": 0.5,
            "peak_memory_mb": 42.1234,
            "features_path": Path("dummy_features.csv"),
            "labels_path": Path("dummy_labels.csv"),
        }
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            print_summary(summary)
        output = stdout_capture.getvalue()
        self.assertIn("Peak memory:                     42.12 MB", output)

    def test_sample_preparation_fallback_when_resource_is_none(self):
        """prepare_sample succeeds with resource=None, producing peak_memory_mb=None."""
        with patch("scripts.prepare_data.resource", None):
            res = prepare_sample(
                data_path=self.csv_path,
                output_dir=self.out_dir,
                sample_size=5,
            )

        self.assertIsNone(res["peak_memory_mb"])
        self.assertEqual(res["processed_count"], 5)
        self.assertTrue((self.out_dir / "sample_features.csv").exists())
        self.assertTrue((self.out_dir / "sample_labels.csv").exists())
        self.assertTrue((self.out_dir / "sample_split_assignments.csv").exists())

    def test_full_preparation_and_manifest_fallback_when_resource_is_none(self):
        """prepare_full succeeds with resource=None, recording peak_memory_mb=null in manifest."""
        with patch("scripts.prepare_data.resource", None):
            res = prepare_full(
                data_path=self.csv_path,
                output_dir=self.out_dir,
                chunk_size=2,
                artifacts_dir=self.art_dir,
            )

        self.assertIsNone(res["peak_memory_mb"])
        self.assertEqual(res["processed_count"], 5)

        # Verify output files exist
        self.assertTrue((self.out_dir / "full_features.csv").exists())
        self.assertTrue((self.out_dir / "full_labels.csv").exists())
        self.assertTrue((self.out_dir / "full_split_assignments.csv").exists())
        manifest_path = self.out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue((self.art_dir / "data_checks.json").exists())

        # Inspect parsed manifest.json
        import json
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.assertIsNone(manifest_data["execution_metadata"]["peak_memory_mb"])

        # Inspect raw JSON text to ensure explicit 'null'
        raw_manifest_text = manifest_path.read_text(encoding="utf-8")
        self.assertIn('"peak_memory_mb": null', raw_manifest_text)


if __name__ == "__main__":
    unittest.main()
