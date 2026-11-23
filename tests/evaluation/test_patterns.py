"""
Tests for src/evaluation/patterns.py, using tiny made-up files.

Run:  python -m unittest discover -s tests/evaluation -t .
"""

import tempfile
import unittest
from pathlib import Path

from src.data.load import RAW_COLUMNS
from src.evaluation.patterns import coverage_summary, match_patterns, read_patterns

# Made-up transactions in the raw CSV layout (11 fields)
A = "2022/09/01 00:06,021,800A,012,800B,100.00,Euro,100.00,Euro,ACH,1"
B = "2022/09/01 01:00,021,800A,020,800C,250.50,Euro,250.50,Euro,ACH,1"
C = "2022/09/02 09:00,030,800D,030,800E,75.00,US Dollar,75.00,US Dollar,ACH,1"
D = "2022/09/03 10:00,040,800F,041,800G,9.99,Yuan,9.99,Yuan,Cheque,1"
NORMAL = "2022/09/01 02:00,001,100X,002,100Y,5.00,Euro,5.00,Euro,Cash,0"
EXTRA = "2022/09/04 10:00,050,800H,051,800I,1.00,Euro,1.00,Euro,Wire,1"

PATTERNS = f"""BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out
{A}
{B}
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - BIPARTITE
{C}
END LAUNDERING ATTEMPT - BIPARTITE

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops
{D}
END LAUNDERING ATTEMPT - CYCLE
"""


class TestPatterns(unittest.TestCase):

    def setUp(self):
        # a temporary folder for the made-up files
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.folder / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_csv(self, rows):
        return self.write("trans.csv", ",".join(RAW_COLUMNS) + "\n" + "\n".join(rows) + "\n")

    def test_read_patterns(self):
        p = read_patterns(self.write("p.txt", PATTERNS))
        self.assertEqual(len(p), 4)                             # 4 records
        self.assertEqual(list(p["attempt_id"]), [1, 1, 2, 3])
        self.assertEqual(list(p["pattern_type"]), ["FAN-OUT", "FAN-OUT", "BIPARTITE", "CYCLE"])
        self.assertEqual(p.loc[0, "sender_bank"], "021")        # leading zero kept
        self.assertEqual(p.loc[0, "amount_paid"], "100.00")     # amount kept as text

    def test_broken_file_is_rejected(self):
        with self.assertRaises(ValueError):
            read_patterns(self.write("bad.txt", A + "\n"))       # record outside a block

    def test_matching(self):
        p = read_patterns(self.write("p.txt", PATTERNS))
        # row numbers (= transaction_id): A=1, NORMAL=2, B=3, C=4, C again=5, EXTRA=6
        # D is not in the CSV at all
        csv_path = self.write_csv([A, NORMAL, B, C, C, EXTRA])

        # chunk_size=2 also checks that row numbers keep counting across pieces
        result, n_laundering = match_patterns(p, csv_path, chunk_size=2)

        self.assertEqual(list(result["status"]), ["matched", "matched", "ambiguous", "unmatched"])
        self.assertEqual(list(result["transaction_id"].dropna()), [1, 3])
        self.assertEqual(n_laundering, 5)                       # rows 1, 3, 4, 5, 6

        s = coverage_summary(result, n_laundering)
        self.assertEqual((s["matched"], s["unmatched"], s["ambiguous"]), (2, 1, 1))
        self.assertEqual(s["laundering_without_pattern"], 3)    # 5 laundering - 2 matched

    def test_match_is_exact_text(self):
        # "100.0" in the CSV must not match "100.00" in the patterns file
        p = read_patterns(self.write("p.txt", PATTERNS))
        csv_path = self.write_csv([A.replace("100.00,Euro,100.00", "100.0,Euro,100.0"), B])
        result, _ = match_patterns(p, csv_path)
        self.assertEqual(result.loc[0, "status"], "unmatched")


if __name__ == "__main__":
    unittest.main()
