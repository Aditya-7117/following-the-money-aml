import csv
import tempfile
import unittest
from pathlib import Path

from src.data.load import RAW_COLUMNS, load_transactions


class TestLoader(unittest.TestCase):

    def test_account_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"

            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(RAW_COLUMNS)
                writer.writerow([
                    "2022/09/01 00:20",
                    "010", "ACCOUNT_A",
                    "020", "ACCOUNT_B",
                    "100", "US Dollar",
                    "100", "US Dollar",
                    "Wire", "0"
                ])

            transactions = list(load_transactions(path))

        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["sender_bank"], "010")
        self.assertEqual(transactions[0]["sender_account"], "ACCOUNT_A")
        self.assertEqual(transactions[0]["receiver_account"], "ACCOUNT_B")
        self.assertEqual(transactions[0]["transaction_id"], 1)

    def test_incomplete_row(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.csv"

            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(RAW_COLUMNS)
                writer.writerow(["2022/09/01 00:20", "010"])

            with self.assertRaises(ValueError):
                list(load_transactions(path))

    def test_invalid_laundering_label(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid_label.csv"

            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(RAW_COLUMNS)
                writer.writerow([
                    "2022/09/01 00:20",
                    "010", "ACCOUNT_A",
                    "020", "ACCOUNT_B",
                    "100", "US Dollar",
                    "100", "US Dollar",
                    "Wire", "ABC"
                ])

            with self.assertRaisesRegex(ValueError, "Invalid laundering label"):
                list(load_transactions(path))
    
    def test_invalid_amounts(self):
        valid_row = [
            "2022/09/01 00:20",
            "010", "ACCOUNT_A",
            "020", "ACCOUNT_B",
            "100", "US Dollar",
            "100", "US Dollar",
            "Wire", "0"
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid_amount.csv"

            for index in (5, 7):
                for bad_amount in ("ABC", "-5", "nan", "inf"):
                    row = valid_row.copy()
                    row[index] = bad_amount

                    with path.open("w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(RAW_COLUMNS)
                        writer.writerow(row)

                    with self.subTest(index=index, amount=bad_amount):
                        with self.assertRaisesRegex(ValueError, "Invalid amount_"):
                            list(load_transactions(path))