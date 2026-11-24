"""
tests/models/test_models.py
Unit tests for src/models (data loading, metrics, preprocessing, tuning).

All tests use small synthetic data so they run in seconds and do not need
the 5-million-row prepared files.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.data import feature_columns, load_modelling_frame, load_models_config, split_xy
from src.models.metrics import best_f1_threshold, evaluate_scores
from src.models.train import (
    MODEL_NAMES,
    build_pipeline,
    comparison_table,
    logistic_coefficients,
    naive_bayes_separation,
    run_model_selection,
    tree_importances,
)

CFG = load_models_config()


def _small_grid_config() -> dict:
    """Real config with one-value grids so tuning tests stay fast."""
    cfg = copy.deepcopy(CFG)
    cfg["models"]["logistic_regression"]["grid"] = {"class_weight": [None, "balanced"], "C": [1.0]}
    cfg["models"]["naive_bayes"]["grid"] = {"var_smoothing": [1e-9]}
    cfg["models"]["decision_tree"]["grid"] = {"max_depth": [3], "min_samples_leaf": [5]}
    return cfg


def _synthetic_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Return a frame with the 10 features, a learnable label and a split column."""
    rng = np.random.default_rng(seed)
    fmt = rng.choice(["ACH", "Cheque", "Wire"], size=n)
    amount = rng.normal(7, 1, size=n)
    y = ((fmt == "ACH") & (amount > 7.5)).astype(int)
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
        "is_laundering": y,
    })
    df["split"] = np.where(np.arange(n) < int(n * 0.7), "train", "val")
    return df


def _write_processed(df: pd.DataFrame, folder: Path, prefix: str = "full") -> None:
    feats = ["transaction_id"] + feature_columns(CFG)
    df[feats].to_csv(folder / f"{prefix}_features.csv", index=False)
    df[["transaction_id", "is_laundering"]].to_csv(folder / f"{prefix}_labels.csv", index=False)
    df[["transaction_id", "split"]].to_csv(folder / f"{prefix}_split_assignments.csv", index=False)


class TestDataLoading(unittest.TestCase):

    def test_refuses_test_splits_by_default(self):
        with self.assertRaises(ValueError):
            load_modelling_frame(Path("unused"), splits=("train", "test_main"))
        with self.assertRaises(ValueError):
            load_modelling_frame(Path("unused"), splits=("test_tail",))

    def test_join_and_filter(self):
        df = _synthetic_frame()
        df.loc[df.index[-5:], "split"] = "test_main"
        with tempfile.TemporaryDirectory() as folder:
            _write_processed(df, Path(folder))
            loaded = load_modelling_frame(Path(folder))
        self.assertEqual(set(loaded["split"]), {"train", "val"})
        self.assertEqual(len(loaded), len(df) - 5)
        self.assertEqual(loaded["same_bank"].dtype, bool)

    def test_misaligned_files_rejected(self):
        df = _synthetic_frame()
        with tempfile.TemporaryDirectory() as folder:
            _write_processed(df, Path(folder))
            labels = pd.read_csv(Path(folder) / "full_labels.csv").iloc[::-1]
            labels.to_csv(Path(folder) / "full_labels.csv", index=False)
            with self.assertRaises(ValueError):
                load_modelling_frame(Path(folder))

    def test_split_xy_excludes_non_features(self):
        X, y = split_xy(_synthetic_frame(), "train", CFG)
        self.assertEqual(list(X.columns), feature_columns(CFG))
        self.assertEqual(len(X.columns), 10)
        for col in ("transaction_id", "is_laundering", "split"):
            self.assertNotIn(col, X.columns)
        self.assertEqual(len(X), len(y))

    def test_config_features_match_data_config(self):
        from src.data.split import get_feature_columns
        self.assertEqual(sorted(feature_columns(CFG)), sorted(get_feature_columns()))


class TestMetrics(unittest.TestCase):

    def test_confusion_counts(self):
        y = [0, 0, 1, 1, 0]
        s = [0.1, 0.8, 0.9, 0.2, 0.3]
        m = evaluate_scores(y, s, threshold=0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (1, 1, 2, 1))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_threshold_is_inclusive(self):
        m = evaluate_scores([1, 0], [0.5, 0.4], threshold=0.5)
        self.assertEqual(m["tp"], 1)

    def test_no_positive_predictions(self):
        m = evaluate_scores([1, 0], [0.1, 0.2], threshold=0.9)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_best_f1_threshold_separates_perfectly(self):
        y = [0, 0, 0, 1, 1]
        s = [0.1, 0.2, 0.3, 0.7, 0.9]
        t = best_f1_threshold(y, s)
        self.assertAlmostEqual(evaluate_scores(y, s, t)["f1"], 1.0)


class TestPipelines(unittest.TestCase):

    def setUp(self):
        df = _synthetic_frame()
        self.X_train, self.y_train = split_xy(df, "train", CFG)
        self.X_val, self.y_val = split_xy(df, "val", CFG)

    def test_scaler_fitted_on_train_only(self):
        pipe = build_pipeline("logistic_regression", {"C": 1.0, "class_weight": None}, CFG)
        pipe.fit(self.X_train, self.y_train)
        scaler = pipe.named_steps["prep"].named_transformers_["amount"]
        self.assertAlmostEqual(scaler.mean_[0], self.X_train["log_amount_paid"].mean())

    def test_unseen_category_in_val_is_ignored(self):
        pipe = build_pipeline("logistic_regression", {"C": 1.0, "class_weight": None}, CFG)
        pipe.fit(self.X_train, self.y_train)
        X = self.X_val.copy()
        X.iloc[0, X.columns.get_loc("payment_format")] = "Bitcoin"
        probs = pipe.predict_proba(X)[:, 1]
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_tree_has_no_scaling(self):
        pipe = build_pipeline("decision_tree", {"max_depth": 3, "min_samples_leaf": 5}, CFG)
        pipe.fit(self.X_train, self.y_train)
        from sklearn.preprocessing import StandardScaler
        prep = pipe.named_steps["prep"]
        self.assertFalse(any(
            isinstance(step, StandardScaler)
            for _, tf, _ in prep.transformers_
            for step in (tf.named_steps.values() if hasattr(tf, "named_steps") else [tf])
        ))
        self.assertEqual(pipe.named_steps["model"].class_weight, "balanced")

    def test_model_selection_runs_end_to_end(self):
        cfg = _small_grid_config()
        results = run_model_selection(cfg, self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertEqual(set(results), set(MODEL_NAMES))
        self.assertEqual(len(results["logistic_regression"]["grid_results"]), 2)
        for r in results.values():
            m = r["metrics_tuned"]
            self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], len(self.y_val))
            self.assertTrue(0.0 <= m["pr_auc"] <= 1.0)
        table = comparison_table(results)
        self.assertEqual(len(table), 3)
        self.assertTrue(table["pr_auc"].is_monotonic_decreasing)
        # The synthetic label depends on ACH + amount; models should find it.
        self.assertGreater(results["decision_tree"]["metrics_tuned"]["pr_auc"], 0.8)

    def test_interpretation_helpers(self):
        cfg = _small_grid_config()
        results = run_model_selection(cfg, self.X_train, self.y_train, self.X_val, self.y_val)
        coef = logistic_coefficients(results["logistic_regression"]["pipeline"])
        imp = tree_importances(results["decision_tree"]["pipeline"])
        sep = naive_bayes_separation(results["naive_bayes"]["pipeline"])
        for s in (coef, imp, sep):
            self.assertEqual(len(s), 2 + 3 + 1 + 7)  # 2 currencies + 3 formats + bool + 7 numeric
        self.assertAlmostEqual(imp.sum(), 1.0)
        self.assertIn(imp.index[0], {"cat__payment_format_ACH", "amount__log_amount_paid"})


if __name__ == "__main__":
    unittest.main()
