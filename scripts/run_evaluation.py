"""
scripts/run_evaluation.py
Evaluate the saved models and two baseline rules on val and test.

Usage (after prepare_data.py --full and train_models.py):
    python3 scripts/run_evaluation.py

Reads data/processed/full_*.csv, artifacts/models/*.pkl and the two raw
files in data/raw/. Writes the results tables to artifacts/evaluation/
and every score to artifacts/scores/scores.csv, which is gitignored.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import build_tables, load_evaluation_config, load_model, pattern_recall, score_all
from src.evaluation.patterns import match_patterns, pattern_summary, read_patterns
from src.models.data import DEFAULT_PROCESSED_DIR, load_modelling_frame, load_models_config
from src.models.train import MODEL_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the AML models on val and test.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "evaluation")
    parser.add_argument("--scores-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "scores")
    args = parser.parse_args(argv)

    cfg = load_evaluation_config()
    views = cfg["views"]

    # 1. Load val and test rows. allow_test=True is only used here, in evaluation.
    df = load_modelling_frame(args.processed_dir, splits=("val", "test_main", "test_tail"), allow_test=True)
    print(f"Loaded {len(df):,} rows:", df["split"].value_counts().to_dict())

    # 2. Score every row with the saved models and the two baselines
    models = {name: load_model(args.models_dir, name) for name in MODEL_NAMES}
    scores, thresholds = score_all(df, models, load_models_config())

    # 3. Link the laundering patterns to transaction IDs
    patterns, n_laundering = match_patterns(
        read_patterns(args.raw_dir / "HI-Small_Patterns.txt"), args.raw_dir / "HI-Small_Trans.csv"
    )
    matched = patterns.loc[patterns["status"] == "matched", ["transaction_id", "pattern_type"]]
    scores = scores.merge(matched.astype({"transaction_id": "int64"}), on="transaction_id", how="left")

    # 4. Build the three tables
    results, budget = build_tables(scores, thresholds, views, cfg["alert_budget"])
    by_pattern = pattern_recall(scores, thresholds, views)

    # 5. Save. Missing values (NaN) are written as "n/a".
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("results_table", results), ("budget_table", budget), ("pattern_recall", by_pattern)]:
        table.to_csv(args.output_dir / f"{name}.csv", index=False, na_rep="n/a", float_format="%.4f")
    scores.to_csv(args.scores_dir / "scores.csv", index=False)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "rows_per_split": df["split"].value_counts().to_dict(),
        "models": {n: {"params": m["params"], "threshold": float(m["threshold"])} for n, m in models.items()},
        "patterns": pattern_summary(patterns, n_laundering),
    }
    with (args.output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Show the main numbers
    cols = ["model", "test_view", "precision", "recall", "f1", "alert_fraction", "pr_auc", "roc_auc"]
    print("\nResults (threshold chosen on val):")
    print(results[cols].round(4).to_string(index=False, na_rep="n/a"))
    print("\nRecall at the 1% alert budget:")
    print(budget.round(4).to_string(index=False, na_rep="n/a"))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
