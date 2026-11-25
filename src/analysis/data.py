"""
src/analysis/data.py
Load the prepared AML data for exploratory data analysis (EDA).

The three CSVs written by scripts/prepare_data.py are joined on
transaction_id. By default only the train split is returned, so that
looking at the data cannot influence decisions about the test period.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_TRANSACTIONS = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"

SPLIT_ORDER = ["train", "val", "test_main", "test_tail"]
LABEL_COLUMN = "is_laundering"


def load_eda_frame(splits=("train",), processed_dir=PROCESSED_DIR, prefix="full"):
    """Return features + label + split for the chosen splits, as one table."""
    unknown = [s for s in splits if s not in SPLIT_ORDER]
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; choose from {SPLIT_ORDER}")

    processed_dir = Path(processed_dir)
    features = pd.read_csv(
        processed_dir / f"{prefix}_features.csv",
        dtype={"payment_currency": "category", "payment_format": "category"},
    )
    labels = pd.read_csv(processed_dir / f"{prefix}_labels.csv")
    split_df = pd.read_csv(processed_dir / f"{prefix}_split_assignments.csv")

    # The three files must describe exactly the same transactions, in the same order.
    if not (
        len(features) == len(labels) == len(split_df)
        and features["transaction_id"].equals(labels["transaction_id"])
        and features["transaction_id"].equals(split_df["transaction_id"])
    ):
        raise ValueError("Feature, label and split files are not aligned on transaction_id")

    df = features.merge(labels, on="transaction_id").merge(split_df, on="transaction_id")
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    if df["same_bank"].dtype != bool:
        df["same_bank"] = df["same_bank"].astype(str).map({"True": True, "False": False})
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER)
    return df


def load_timestamps(raw_path=RAW_TRANSACTIONS):
    """Return transaction_id and a real datetime for every raw transaction.

    transaction_id is the 1-based row number in the raw CSV, the same rule
    src/data/load.py uses, so this joins straight onto load_eda_frame().
    """
    raw = pd.read_csv(raw_path, usecols=["Timestamp"])
    raw["transaction_id"] = range(1, len(raw) + 1)
    raw["timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    return raw[["transaction_id", "timestamp"]]