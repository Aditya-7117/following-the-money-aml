"""
tests/evaluation/test_evaluation.py
Unit tests for src/evaluation (metrics, results tables, patterns) and
scripts/run_evaluation.py.

All tests use small made-up data, and every expected number in the
small examples was worked out by hand.
"""

import contextlib
import io
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_evaluation import main
from src.data.load import RAW_COLUMNS
from src.evaluation.evaluate import build_tables, pattern_recall
from src.evaluation.patterns import match_patterns, pattern_summary, read_patterns
from src.evaluation.metrics import average_precision, recall_at_budget, roc_auc, threshold_metrics
from src.models.data import feature_columns, load_models_config
from src.models.metrics import best_f1_threshold, evaluate_scores
from src.models.train import build_pipeline

CFG = load_models_config()
VIEWS = {"val": ["val"], "test_main": ["test_main"], "test_full": ["test_main", "test_tail"]}

# Small example used by the metric tests.
# score: 0.9  0.8  0.4  0.3  0.1
# label:  1    0    1    0    0
LABELS = [1, 0, 1, 0, 0]
SCORES = [0.9, 0.8, 0.4, 0.3, 0.1]


def budget_frame(labels, scores, ids):
    return pd.DataFrame({"transaction_id": ids, "is_laundering": labels, "score": scores})


class TestThresholdMetrics(unittest.TestCase):

    def test_worked_example(self):
        # threshold 0.35 flags the top 3 rows: 2 laundering, 1 normal
        m = threshold_metrics(LABELS, SCORES, 0.35)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 0, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 0.8)
        self.assertAlmostEqual(m["alert_fraction"], 3 / 5)

    def test_flags_nothing(self):
        m = threshold_metrics(LABELS, SCORES, 2.0)
        self.assertTrue(math.isnan(m["precision"]))   # n/a, not 0
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["alert_fraction"], 0.0)

    def test_flags_everything(self):
        m = threshold_metrics(LABELS, SCORES, 0.0)
        self.assertAlmostEqual(m["precision"], 2 / 5)  # equals the laundering rate
        self.assertEqual(m["recall"], 1.0)

    def test_threshold_is_inclusive(self):
        # a score equal to the threshold is flagged
        self.assertEqual(threshold_metrics([1, 0], [0.5, 0.2], 0.5)["tp"], 1)


class TestRankingMetrics(unittest.TestCase):

    def test_average_precision_worked_example(self):
        # 0.9 is laundering: recall 0.5 at precision 1   -> 0.5 * 1
        # 0.4 is laundering: recall 1.0 at precision 2/3 -> 0.5 * 2/3
        self.assertAlmostEqual(average_precision(LABELS, SCORES), 0.5 + 1 / 3)

    def test_roc_auc_worked_example(self):
        # 6 (laundering, normal) pairs; the laundering row scores higher in 5
        self.assertAlmostEqual(roc_auc(LABELS, SCORES), 5 / 6)

    def test_one_class_gives_nan(self):
        self.assertTrue(math.isnan(average_precision([0, 0], [0.4, 0.2])))
        self.assertTrue(math.isnan(roc_auc([0, 0], [0.4, 0.2])))


class TestRecallAtBudget(unittest.TestCase):

    def test_worked_example(self):
        # 20% of 5 rows = 1 alert, given to the 0.9 row (laundering)
        r = recall_at_budget(budget_frame(LABELS, SCORES, [10, 11, 12, 13, 14]), budget=0.2)
        self.assertEqual((r["alerts_allowed"], r["caught"]), (1, 1))
        self.assertAlmostEqual(r["recall_at_budget"], 0.5)
        self.assertEqual(r["tied_rows_left_out"], 0)

    def test_budget_rounds_up(self):
        self.assertEqual(recall_at_budget(budget_frame([0] * 101, [0.5] * 101, range(101)), 0.01)["alerts_allowed"], 2)
        self.assertEqual(recall_at_budget(budget_frame([0] * 100, [0.5] * 100, range(100)), 0.07)["alerts_allowed"], 7)

    def test_ties_go_to_smallest_id(self):
        # all scores equal, so only the ID decides: ID 1 gets the alert
        r = recall_at_budget(budget_frame([0, 0, 0, 0, 1], [0.0] * 5, [5, 4, 3, 2, 1]), 0.2)
        self.assertEqual(r["caught"], 1)
        self.assertEqual(r["tied_rows_left_out"], 4)

    def test_labels_do_not_choose_the_alerts(self):
        # laundering moved to another row: ID 1 is still the one alerted
        r = recall_at_budget(budget_frame([1, 0, 0, 0, 0], [0.0] * 5, [5, 4, 3, 2, 1]), 0.2)
        self.assertEqual(r["caught"], 0)


class TestResultsTables(unittest.TestCase):

    def setUp(self):
        # 2 val rows, 3 test_main rows, 1 test_tail row
        self.scores = pd.DataFrame({
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "split": ["val", "val", "test_main", "test_main", "test_main", "test_tail"],
            "is_laundering": [1, 0, 1, 0, 0, 1],
            "always_legitimate": [0.0] * 6,
            "ach_rule": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "pattern_type": ["FAN-IN", None, "FAN-IN", None, None, None],
        })
        self.thresholds = {"always_legitimate": 1.0, "ach_rule": 1.0}
        self.results, self.budget = build_tables(self.scores, self.thresholds, VIEWS, 0.01)

    def row(self, model, view):
        r = self.results
        return r[(r["model"] == model) & (r["test_view"] == view)].iloc[0]

    def test_one_row_per_model_and_view(self):
        self.assertEqual(len(self.results), 2 * 3)
        self.assertEqual(len(self.budget), 2 * 3)

    def test_views_use_the_right_rows(self):
        self.assertEqual(self.row("ach_rule", "val")["n_rows"], 2)
        self.assertEqual(self.row("ach_rule", "test_main")["n_rows"], 3)
        self.assertEqual(self.row("ach_rule", "test_full")["n_rows"], 4)   # main + tail

    def test_always_legitimate_flags_nothing(self):
        r = self.row("always_legitimate", "test_main")
        self.assertEqual(r["tp"] + r["fp"], 0)
        self.assertTrue(math.isnan(r["precision"]))

    def test_ach_rule_by_hand(self):
        # test_main: row 3 (ACH, laundering), 4 (ACH, normal), 5 (not ACH, normal)
        r = self.row("ach_rule", "test_main")
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 0, 1))

    def test_pattern_recall(self):
        p = pattern_recall(self.scores, self.thresholds, VIEWS)
        self.assertNotIn("val", set(p["test_view"]))                    # test views only
        r = p[(p["model"] == "ach_rule") & (p["test_view"] == "test_full")]
        fan_in = r[r["pattern_type"] == "FAN-IN"].iloc[0]
        self.assertEqual((fan_in["laundering_rows"], fan_in["caught"]), (1, 1))   # row 3
        none = r[r["pattern_type"] == "none"].iloc[0]
        self.assertEqual((none["laundering_rows"], none["caught"]), (1, 1))       # row 6
        # always_legitimate catches nothing in any pattern
        self.assertEqual(p[p["model"] == "always_legitimate"]["caught"].sum(), 0)


# Made-up raw transactions (11 fields, same layout as HI-Small_Trans.csv)
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
        s = pattern_summary(result, n_laundering)
        self.assertEqual((s["matched"], s["unmatched"], s["ambiguous"]), (2, 1, 1))
        self.assertEqual(s["laundering_without_pattern"], 3)    # 5 laundering - 2 matched

    def test_match_is_exact_text(self):
        # "100.0" in the CSV must not match "100.00" in the patterns file
        p = read_patterns(self.write("p.txt", PATTERNS))
        csv_path = self.write_csv([A.replace("100.00,Euro,100.00", "100.0,Euro,100.0"), B])
        result, _ = match_patterns(p, csv_path)
        self.assertEqual(result.loc[0, "status"], "unmatched")


def make_fake_data(n=900, seed=1):
    """Fake prepared data in the same layout as the data pipeline's output."""
    rng = np.random.default_rng(seed)
    fmt = rng.choice(["ACH", "Cheque", "Wire"], size=n)
    amount = rng.normal(7, 1, size=n)
    df = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "payment_currency": rng.choice(["US Dollar", "Euro"], size=n),
        "payment_format": fmt,
        "same_bank": rng.random(n) < 0.3,
        "log_amount_paid": amount,
        "sender_txn_count_1h": rng.poisson(1, n),
        "sender_txn_count_24h": rng.poisson(5, n),
        "sender_unique_receivers_24h": rng.poisson(2, n),
        "receiver_txn_count_1h": rng.poisson(1, n),
        "receiver_txn_count_24h": rng.poisson(3, n),
        "receiver_unique_senders_24h": rng.poisson(1, n),
        "is_laundering": ((fmt == "ACH") & (amount > 7.5)).astype(int),
    })
    df["split"] = pd.cut(df["transaction_id"], [0, 500, 700, 880, n],
                         labels=["train", "val", "test_main", "test_tail"]).astype(str)
    return df


class TestFullRun(unittest.TestCase):
    """Train tiny versions of the three models, then run the whole script."""

    def test_full_run(self):
        df = make_fake_data()
        params = {
            "logistic_regression": {"class_weight": None, "C": 1.0},
            "naive_bayes": {"var_smoothing": 1e-9},
            "decision_tree": {"max_depth": 3, "min_samples_leaf": 5},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            processed, models_dir, raw = tmp / "processed", tmp / "models", tmp / "raw"
            for folder in (processed, models_dir, raw):
                folder.mkdir()

            # the three prepared files
            df[["transaction_id"] + feature_columns(CFG)].to_csv(processed / "full_features.csv", index=False)
            df[["transaction_id", "is_laundering"]].to_csv(processed / "full_labels.csv", index=False)
            df[["transaction_id", "split"]].to_csv(processed / "full_split_assignments.csv", index=False)

            # raw CSV: one row per transaction, in transaction_id order
            raw_rows = [f"2022/09/01 00:00,001,A{t},002,B{t},{t}.00,Euro,{t}.00,Euro,{f},{y}"
                        for t, f, y in zip(df["transaction_id"], df["payment_format"], df["is_laundering"])]
            (raw / "HI-Small_Trans.csv").write_text(",".join(RAW_COLUMNS) + "\n" + "\n".join(raw_rows) + "\n")

            # patterns file: two laundering rows from the test splits
            test_ids = df[(df["is_laundering"] == 1) & (df["split"] != "train")]["transaction_id"].head(2)
            records = "\n".join(raw_rows[t - 1] for t in test_ids)
            (raw / "HI-Small_Patterns.txt").write_text(
                f"BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 2-degree Fan-In\n{records}\n"
                "END LAUNDERING ATTEMPT - FAN-IN\n")

            # save the models the same way scripts/train_models.py does
            train, val = df[df["split"] == "train"], df[df["split"] == "val"]
            val_check = {}
            for name, p in params.items():
                pipe = build_pipeline(name, p, CFG).fit(train[feature_columns(CFG)], train["is_laundering"])
                val_scores = pipe.predict_proba(val[feature_columns(CFG)])[:, 1]
                threshold = best_f1_threshold(val["is_laundering"], val_scores)
                val_check[name] = evaluate_scores(val["is_laundering"], val_scores, threshold)
                with (models_dir / f"{name}.pkl").open("wb") as f:
                    pickle.dump({"pipeline": pipe, "threshold": threshold, "params": p}, f)

            # run the script quietly (it prints the tables)
            out, scores_dir = tmp / "evaluation", tmp / "scores"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--processed-dir", str(processed), "--models-dir", str(models_dir),
                      "--raw-dir", str(raw), "--output-dir", str(out), "--scores-dir", str(scores_dir)])

            results = pd.read_csv(out / "results_table.csv", na_values="n/a")
            budget = pd.read_csv(out / "budget_table.csv", na_values="n/a")
            by_pattern = pd.read_csv(out / "pattern_recall.csv", na_values="n/a")
            summary = json.loads((out / "evaluation_summary.json").read_text())
            scores = pd.read_csv(scores_dir / "scores.csv")

        # 5 models (3 trained + 2 baselines) x 3 views
        self.assertEqual(len(results), 15)
        self.assertEqual(len(budget), 15)
        self.assertEqual(summary["patterns"]["matched"], 2)
        self.assertIn("FAN-IN", set(by_pattern["pattern_type"]))
        self.assertNotIn("train", set(scores["split"]))         # train rows are never scored

        # on val, our numbers must match the model selection numbers
        for name, theirs in val_check.items():
            ours = results[(results["model"] == name) & (results["test_view"] == "val")].iloc[0]
            with self.subTest(model=name):
                self.assertEqual((ours["tp"], ours["fp"]), (theirs["tp"], theirs["fp"]))
                self.assertAlmostEqual(ours["pr_auc"], theirs["pr_auc"], places=4)
                self.assertAlmostEqual(ours["roc_auc"], theirs["roc_auc"], places=4)


if __name__ == "__main__":
    unittest.main()
