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