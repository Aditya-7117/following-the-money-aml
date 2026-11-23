"""
src/models/data.py
Load the prepared feature, label and split files for modelling.

The three CSVs written by scripts/prepare_data.py are joined on
transaction_id. Test partitions are refused unless the caller passes
allow_test=True, so model selection cannot touch them by accident.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.json"

TEST_SPLITS = ("test_main", "test_tail")
LABEL_COLUMN = "is_laundering"
NON_FEATURE_COLUMNS = ("transaction_id", LABEL_COLUMN, "split")


def load_models_config(config_path: Path | None = None) -> dict:
    """Return the parsed config/models.json dict."""
    path = config_path or MODELS_CONFIG_PATH
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def feature_columns(cfg: dict) -> list[str]:
    """Return all model feature names in config order."""
    groups = cfg["feature_groups"]
    return groups["categorical"] + groups["boolean"] + groups["numeric"]


def load_modelling_frame(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    prefix: str = "full",
    splits: tuple[str, ...] = ("train", "val"),
    allow_test: bool = False,
) -> pd.DataFrame:
    """Join features, labels and split assignments, keeping only ``splits``.

    Raises
    ------
    ValueError
        If a test split is requested without ``allow_test=True``, or the
        three files do not share exactly the same transaction_id values.
    """
    blocked = [s for s in splits if s in TEST_SPLITS]
    if blocked and not allow_test:
        raise ValueError(
            f"Refusing to load test split(s) {blocked} during model selection; "
            "pass allow_test=True only for final evaluation."
        )

    processed_dir = Path(processed_dir)
    features = pd.read_csv(
        processed_dir / f"{prefix}_features.csv",
        dtype={"payment_currency": "string", "payment_format": "string"},
    )
    labels = pd.read_csv(processed_dir / f"{prefix}_labels.csv")
    split_df = pd.read_csv(processed_dir / f"{prefix}_split_assignments.csv")

    if not (
        len(features) == len(labels) == len(split_df)
        and features["transaction_id"].equals(labels["transaction_id"])
        and features["transaction_id"].equals(split_df["transaction_id"])
    ):
        raise ValueError("Feature, label and split files are not aligned on transaction_id")

    df = features.merge(labels, on="transaction_id").merge(split_df, on="transaction_id")
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    df["same_bank"] = df["same_bank"].astype(bool)
    return df


def split_xy(df: pd.DataFrame, split: str, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for one split, with only the configured feature columns."""
    part = df[df["split"] == split]
    if part.empty:
        raise ValueError(f"No rows found for split {split!r}")
    return part[feature_columns(cfg)].copy(), part[LABEL_COLUMN].astype(int)
