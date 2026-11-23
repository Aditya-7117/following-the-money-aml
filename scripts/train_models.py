"""
scripts/train_models.py
Tune and compare the three baseline models on the validation split.

Usage:
    python3 scripts/train_models.py              # full prepared data
    python3 scripts/train_models.py --prefix sample --models naive_bayes

Fits on train, selects hyperparameters and thresholds on val, and never
loads test_main or test_tail. Fitted models and result tables are written
to artifacts/models/, which is gitignored.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data import (
    DEFAULT_PROCESSED_DIR,
    MODELS_CONFIG_PATH,
    load_modelling_frame,
    load_models_config,
    split_xy,
)
from src.models.train import MODEL_NAMES, comparison_table, run_model_selection

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune and compare AML baseline models on val.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--prefix", default="full", choices=["full", "sample"])
    parser.add_argument("--config-path", type=Path, default=MODELS_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    args = parser.parse_args(argv)

    cfg = load_models_config(args.config_path)
    df = load_modelling_frame(args.processed_dir, prefix=args.prefix, splits=("train", "val"))
    X_train, y_train = split_xy(df, cfg["fit_split"], cfg)
    X_val, y_val = split_xy(df, cfg["selection_split"], cfg)
    print(f"train: {len(X_train):,} rows, {int(y_train.sum()):,} laundering")
    print(f"val:   {len(X_val):,} rows, {int(y_val.sum()):,} laundering")

    results = run_model_selection(cfg, X_train, y_train, X_val, y_val, models=args.models, progress=print)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuned = comparison_table(results, "metrics_tuned")
    default = comparison_table(results, "metrics_default")
    tuned.to_csv(args.output_dir / "val_comparison_tuned_threshold.csv")
    default.to_csv(args.output_dir / "val_comparison_threshold_0.5.csv")
    for name, r in results.items():
        r["grid_results"].to_csv(args.output_dir / f"grid_{name}.csv", index=False)
        with (args.output_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump({"pipeline": r["pipeline"], "threshold": r["threshold"], "params": r["best_params"]}, f)
    with (args.output_dir / "val_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {n: {"best_params": r["best_params"], "val_tuned": r["metrics_tuned"],
                 "val_threshold_0.5": r["metrics_default"]} for n, r in results.items()},
            f, indent=2,
        )

    cols = ["pr_auc", "roc_auc", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    print("\nValidation comparison (val-F1-optimal threshold):")
    print(tuned[["best_params"] + cols].to_string())
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
