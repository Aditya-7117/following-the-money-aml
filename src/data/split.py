"""
src/data/split.py
Chronological split assignment for the AML transaction pipeline.

Boundaries (provisional — pending team approval) are read from
config/data.json at call time and passed into functions explicitly;
no import-time cache is kept so callers always see the live config.

Split labels:
  "train"     — timestamp  <  train_cutoff
  "val"       — train_cutoff <= timestamp  <  val_cutoff
  "test_main" — val_cutoff   <= timestamp  <  tail_cutoff
  "test_tail" — timestamp  >= tail_cutoff

test_main and test_tail are physically disjoint.  The full test view
is their union; it is not an independent set of rows distinct from both.

Window-rule discrepancy note
----------------------------
ADITYA-HANDBOOK.md Day 2 step 3 says "lower boundary is included" but
features.py implements a strictly open interval on BOTH sides:

    t - window  <  prev_timestamp  <  t          (both endpoints excluded)

The features.py behaviour is correct per AGENTS.md leak rules and all
50 unit tests.  The handbook wording is imprecise.  This module and
config/data.json document the discrepancy; features.py is NOT changed.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "data.json"

SPLIT_LABELS: tuple[str, ...] = ("train", "val", "test_main", "test_tail")


def load_config(config_path: Path | None = None) -> dict:
    """Load and return the parsed config/data.json dict.

    Parameters
    ----------
    config_path:
        Path to data.json.  Defaults to ``<project_root>/config/data.json``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required keys are missing or boundary strings are absent.
    """
    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Raise ValueError if the config is structurally incomplete."""
    for top_key in ("feature_columns", "split_boundaries", "rolling_window_rules"):
        if top_key not in cfg:
            raise ValueError(f"config/data.json is missing required key: {top_key!r}")

    sb = cfg["split_boundaries"]
    for boundary_key in ("train_cutoff", "val_cutoff", "tail_cutoff"):
        if boundary_key not in sb:
            raise ValueError(
                f"config/data.json split_boundaries is missing {boundary_key!r}"
            )
        val = sb[boundary_key]
        if not isinstance(val, str) or len(val) != 16:
            raise ValueError(
                f"split_boundaries[{boundary_key!r}] must be a 16-char "
                f"'YYYY/MM/DD HH:MM' string, got {val!r}"
            )

    # Boundaries must be strictly ordered
    tc = sb["train_cutoff"]
    vc = sb["val_cutoff"]
    tail = sb["tail_cutoff"]
    if not (tc < vc < tail):
        raise ValueError(
            f"split_boundaries must satisfy train_cutoff < val_cutoff < tail_cutoff; "
            f"got {tc!r}, {vc!r}, {tail!r}"
        )

    fc = cfg.get("feature_columns", [])
    if not isinstance(fc, list) or len(fc) == 0:
        raise ValueError("config/data.json feature_columns must be a non-empty list")


def get_split_boundaries(config_path: Path | None = None) -> dict:
    """Return a copy of the split_boundaries section from config/data.json.

    Parameters
    ----------
    config_path:
        Path to data.json.  Defaults to ``<project_root>/config/data.json``.
    """
    cfg = load_config(config_path)
    return dict(cfg["split_boundaries"])


def get_feature_columns(config_path: Path | None = None) -> list[str]:
    """Return the ordered list of model feature column names from config/data.json."""
    cfg = load_config(config_path)
    return list(cfg["feature_columns"])


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_split(timestamp_str: str, boundaries: dict) -> str:
    """Return the split label for a transaction timestamp string.

    Uses lexicographic comparison of ``YYYY/MM/DD HH:MM`` strings, which
    sorts identically to chronological order for this format.

    Parameters
    ----------
    timestamp_str:
        A 16-character timestamp string in ``YYYY/MM/DD HH:MM`` format, as
        produced by ``src.data.load.load_transactions``.
    boundaries:
        Dict with keys ``train_cutoff``, ``val_cutoff``, ``tail_cutoff``.
        Obtain from ``get_split_boundaries()`` or pass the
        ``split_boundaries`` sub-dict from ``load_config()`` directly.

    Returns
    -------
    str
        One of ``"train"``, ``"val"``, ``"test_main"``, ``"test_tail"``.

    Raises
    ------
    ValueError
        If ``timestamp_str`` is not exactly 16 characters or is empty.
    """
    if not timestamp_str or len(timestamp_str) != 16:
        raise ValueError(
            f"timestamp_str must be a 16-char 'YYYY/MM/DD HH:MM' string, "
            f"got {timestamp_str!r}"
        )

    train_cutoff = boundaries["train_cutoff"]
    val_cutoff   = boundaries["val_cutoff"]
    tail_cutoff  = boundaries["tail_cutoff"]

    if timestamp_str < train_cutoff:
        return "train"
    elif timestamp_str < val_cutoff:
        return "val"
    elif timestamp_str < tail_cutoff:
        return "test_main"
    else:
        return "test_tail"
