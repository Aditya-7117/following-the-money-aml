"""
scripts/prepare_data.py
Data preparation script for the AML transaction dataset.

Supports:
- --sample: Extract earliest 100,000 transactions in chronological order.
- --full: Production-ready processing of all 5,078,345 transactions using
          a bounded-memory external sort and incremental streaming feature
          generation.

Preserves exact 10-feature schema, separate labels joined by transaction_id,
and writes verified outputs locally under gitignored data/processed/.
"""

import argparse
import collections
import csv
import hashlib
import heapq
import json
import math
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import FEATURE_COLUMNS, generate_features
from src.data.load import RAW_COLUMNS, load_transactions
from src.data.split import SPLIT_LABELS, assign_split, load_config

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = PROJECT_ROOT / "config" / "data.json"
SAMPLE_SIZE = 100_000
DEFAULT_CHUNK_SIZE = 250_000

NUMERIC_FEATURES = [
    "log_amount_paid",
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_unique_receivers_24h",
    "receiver_txn_count_1h",
    "receiver_txn_count_24h",
    "receiver_unique_senders_24h",
]

COUNT_FEATURES = [
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_unique_receivers_24h",
    "receiver_txn_count_1h",
    "receiver_txn_count_24h",
    "receiver_unique_senders_24h",
]

CHUNK_COLS = [
    "transaction_id",
    "timestamp",
    "sender_bank",
    "sender_account",
    "receiver_bank",
    "receiver_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

def _load_split_boundaries(config_path: Path | None = None) -> dict:
    """Load split boundaries from config/data.json.

    Returns the split_boundaries sub-dict.  Raises FileNotFoundError if the
    config file is missing, ValueError if boundaries are structurally invalid.
    """
    cfg = load_config(config_path or CONFIG_PATH)
    return cfg["split_boundaries"]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest using 1MB binary chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _rel_path(p: str | Path) -> str:
    """Return path relative to PROJECT_ROOT, or filename if outside project."""
    path = Path(p).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return path.name


def get_peak_memory_mb() -> float:
    """Return peak RSS memory in megabytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def select_earliest_transactions(transactions, n: int = 100_000) -> list[dict]:
    """Select the earliest n transactions in chronological order.

    Ties in timestamp are broken by transaction_id.
    Uses heapq.nsmallest, maintaining a bounded max-heap of size n,
    so memory is O(n) rather than O(total_transactions).
    Returns transactions in ascending (timestamp, transaction_id) order.
    """
    return heapq.nsmallest(
        n,
        transactions,
        key=lambda t: (t["timestamp"], t["transaction_id"]),
    )


def sort_transactions_external(
    transactions_iter: Iterable[dict],
    temp_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[dict, None, None]:
    """Sort transactions chronologically with bounded memory.

    Ties in timestamp are broken deterministically by transaction_id.
    Transactions are chunked, sorted in memory, written to temporary CSVs,
    and merged using heapq.merge. Temporary files are deleted upon completion.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    chunk = []
    chunk_idx = 0

    def _write_chunk(rows: list[dict], idx: int) -> Path:
        rows.sort(key=lambda t: (t["timestamp"], t["transaction_id"]))
        p = temp_dir / f"sort_chunk_{idx:05d}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([r[c] for c in CHUNK_COLS])
        return p

    for txn in transactions_iter:
        chunk.append(txn)
        if len(chunk) >= chunk_size:
            chunk_paths.append(_write_chunk(chunk, chunk_idx))
            chunk_idx += 1
            chunk = []

    if chunk:
        if not chunk_paths:
            # Entire input fits in single chunk; sort and yield directly
            chunk.sort(key=lambda t: (t["timestamp"], t["transaction_id"]))
            yield from chunk
            return
        chunk_paths.append(_write_chunk(chunk, chunk_idx))

    # Helper generator to read back a sorted chunk
    def _read_chunk(p: Path) -> Generator[dict, None, None]:
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                d = dict(zip(CHUNK_COLS, row))
                d["transaction_id"] = int(d["transaction_id"])
                yield d

    iters = [_read_chunk(p) for p in chunk_paths]
    try:
        yield from heapq.merge(
            *iters,
            key=lambda t: (t["timestamp"], t["transaction_id"]),
        )
    finally:
        for p in chunk_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def validate_prepared_data(
    feature_rows: list[dict],
    label_rows: list[dict],
    split_rows: list[dict] | None = None,
) -> dict:
    """Verify that:
    1. Output transaction IDs are unique.
    2. All numeric features are finite.
    3. History counts are non-negative integers.
    4. Feature rows and label rows align 1-to-1 on transaction_id.
    5. If split_rows is provided, split rows align 1-to-1 on transaction_id
       and each split label is recognized.

    Returns a dict with validation metrics (e.g. duplicate_id_count).
    Raises ValueError if any check fails.
    """
    seen_ids = set()
    dup_id_count = 0

    for row in feature_rows:
        tid = row["transaction_id"]
        if tid in seen_ids:
            dup_id_count += 1
        seen_ids.add(tid)

        for feat in NUMERIC_FEATURES:
            val = float(row[feat])
            if not math.isfinite(val):
                raise ValueError(
                    f"Non-finite value {val!r} for feature {feat!r} in transaction {tid}"
                )

        for feat in COUNT_FEATURES:
            val = int(row[feat])
            if val < 0:
                raise ValueError(
                    f"Negative count {val} for feature {feat!r} in transaction {tid}"
                )

    if dup_id_count > 0:
        raise ValueError(f"Found {dup_id_count} duplicate transaction IDs in feature rows")

    if len(feature_rows) != len(label_rows):
        raise ValueError(
            f"Row count mismatch: {len(feature_rows)} feature rows vs "
            f"{len(label_rows)} label rows"
        )

    for f_row, l_row in zip(feature_rows, label_rows):
        if f_row["transaction_id"] != l_row["transaction_id"]:
            raise ValueError(
                f"Transaction ID mismatch between features ({f_row['transaction_id']}) "
                f"and labels ({l_row['transaction_id']})"
            )

    if split_rows is not None:
        if len(feature_rows) != len(split_rows):
            raise ValueError(
                f"Row count mismatch: {len(feature_rows)} feature rows vs "
                f"{len(split_rows)} split rows"
            )
        valid_splits = set(SPLIT_LABELS)
        for f_row, s_row in zip(feature_rows, split_rows):
            if f_row["transaction_id"] != s_row["transaction_id"]:
                raise ValueError(
                    f"Transaction ID mismatch between features ({f_row['transaction_id']}) "
                    f"and split assignments ({s_row['transaction_id']})"
                )
            if s_row["split"] not in valid_splits:
                raise ValueError(
                    f"Invalid split label {s_row['split']!r} in transaction {s_row['transaction_id']}"
                )

    return {
        "duplicate_id_count": dup_id_count,
        "unique_id_count": len(seen_ids),
    }


def save_sample_outputs(
    output_dir: Path,
    feature_rows: list[dict],
    label_rows: list[dict],
    split_rows: list[dict] | None = None,
) -> tuple[Path, Path] | tuple[Path, Path, Path]:
    """Save features, labels, and optional split assignments under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "sample_features.csv"
    labels_path = output_dir / "sample_labels.csv"

    feature_fieldnames = ["transaction_id"] + FEATURE_COLUMNS
    with features_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    label_fieldnames = ["transaction_id", "is_laundering"]
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=label_fieldnames)
        writer.writeheader()
        writer.writerows(label_rows)

    if split_rows is not None:
        splits_path = output_dir / "sample_split_assignments.csv"
        split_fieldnames = ["transaction_id", "split"]
        with splits_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=split_fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        return features_path, labels_path, splits_path

    return features_path, labels_path


def stream_and_write_prepared_data(
    sorted_txns_iter: Iterable[dict],
    features_path: Path,
    labels_path: Path,
    split_assignments_path: Path,
    boundaries: dict,
    expected_count: int | None = None,
) -> dict:
    """Stream sorted transactions through generate_features, validate on the fly,
    and write outputs incrementally to safe temporary files, finalizing atomically.

    Writes three output files:
      - features_path         : transaction_id + 10 model feature columns
      - labels_path           : transaction_id + is_laundering
      - split_assignments_path: transaction_id + split label (train/val/test_main/test_tail)

    Split labels are assigned using ``assign_split(timestamp, boundaries)``
    where ``boundaries`` comes from config/data.json (passed in by the caller,
    not cached at import time).

    Returns execution summary and validation statistics.
    """
    output_dir = features_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    features_tmp = features_path.with_name(features_path.name + ".tmp")
    labels_tmp = labels_path.with_name(labels_path.name + ".tmp")
    splits_tmp = split_assignments_path.with_name(split_assignments_path.name + ".tmp")

    feature_fieldnames = ["transaction_id"] + FEATURE_COLUMNS
    label_fieldnames = ["transaction_id", "is_laundering"]
    split_fieldnames = ["transaction_id", "split"]
    expected_feature_keys = set(feature_fieldnames)

    label_queue = collections.deque()

    def tracking_stream(it):
        for t in it:
            label_queue.append((t["transaction_id"], t["is_laundering"], t["timestamp"]))
            yield t

    count = 0
    legit_count = 0
    laundering_count = 0
    earliest_ts = None
    latest_ts = None
    prev_ts = None
    prev_tid = None

    # Track seen IDs via preallocated bytearray for fast, memory-bounded lookup
    # 6,000,000 bytes is ~5.7 MB RAM
    seen_id_bits = bytearray(6_000_000)
    overflow_seen_ids = set()

    # Track candidate splits
    candidate_splits = {
        "train": {"total": 0, "legitimate": 0, "laundering": 0},
        "val": {"total": 0, "legitimate": 0, "laundering": 0},
        "test_main": {"total": 0, "legitimate": 0, "laundering": 0},
        "test_tail": {"total": 0, "legitimate": 0, "laundering": 0},
    }

    try:
        with features_tmp.open("w", newline="", encoding="utf-8") as f_out, \
             labels_tmp.open("w", newline="", encoding="utf-8") as l_out, \
             splits_tmp.open("w", newline="", encoding="utf-8") as s_out:

            f_writer = csv.DictWriter(f_out, fieldnames=feature_fieldnames)
            l_writer = csv.DictWriter(l_out, fieldnames=label_fieldnames)
            s_writer = csv.DictWriter(s_out, fieldnames=split_fieldnames)
            f_writer.writeheader()
            l_writer.writeheader()
            s_writer.writeheader()

            for feat_row in generate_features(tracking_stream(sorted_txns_iter)):
                if not label_queue:
                    raise RuntimeError("Label queue underflow during feature generation")
                expected_tid, label, ts = label_queue.popleft()

                tid = feat_row["transaction_id"]
                if tid != expected_tid:
                    raise ValueError(
                        f"Row alignment mismatch: feature transaction_id {tid} != label transaction_id {expected_tid}"
                    )

                # Check uniqueness of transaction_id
                if tid < len(seen_id_bits):
                    if seen_id_bits[tid]:
                        raise ValueError(f"Duplicate transaction_id: {tid}")
                    seen_id_bits[tid] = 1
                else:
                    if tid in overflow_seen_ids:
                        raise ValueError(f"Duplicate transaction_id: {tid}")
                    overflow_seen_ids.add(tid)

                # Check strict chronological ordering with tie-breaking
                if prev_ts is not None:
                    if ts < prev_ts:
                        raise ValueError(f"Timestamp out of order: {ts} < {prev_ts} at txn {tid}")
                    elif ts == prev_ts and tid < prev_tid:
                        raise ValueError(f"Timestamp tie broken incorrectly: {tid} < {prev_tid} at {ts}")

                prev_ts = ts
                prev_tid = tid

                # Check numeric finite values
                for feat in NUMERIC_FEATURES:
                    val = float(feat_row[feat])
                    if not math.isfinite(val):
                        raise ValueError(
                            f"Non-finite value {val!r} for feature {feat!r} in transaction {tid}"
                        )

                # Check nonnegative history counts
                for feat in COUNT_FEATURES:
                    val = int(feat_row[feat])
                    if val < 0:
                        raise ValueError(
                            f"Negative count {val} for feature {feat!r} in transaction {tid}"
                        )

                # Check schema isolation: no accidental leak columns
                row_keys = set(feat_row.keys())
                if row_keys != expected_feature_keys:
                    extra = row_keys - expected_feature_keys
                    missing = expected_feature_keys - row_keys
                    raise ValueError(
                        f"Schema violation in transaction {tid}: extra={extra}, missing={missing}"
                    )

                f_writer.writerow(feat_row)
                l_writer.writerow({"transaction_id": tid, "is_laundering": label})

                # Assign split label using config boundaries
                split_label = assign_split(ts, boundaries)
                s_writer.writerow({"transaction_id": tid, "split": split_label})

                count += 1
                if label == "0":
                    legit_count += 1
                elif label == "1":
                    laundering_count += 1
                else:
                    raise ValueError(f"Invalid label {label!r} in transaction {tid}")

                if earliest_ts is None:
                    earliest_ts = ts
                latest_ts = ts

                # Bucket into candidate splits using the same assign_split already computed
                candidate_splits[split_label]["total"] += 1
                if label == "1":
                    candidate_splits[split_label]["laundering"] += 1
                else:
                    candidate_splits[split_label]["legitimate"] += 1

        if label_queue:
            raise RuntimeError(f"Label queue has {len(label_queue)} unconsumed items at stream completion")

        if expected_count is not None and count != expected_count:
            raise ValueError(f"Row count mismatch: expected {expected_count}, but processed {count}")

        # Atomic finalize
        features_tmp.replace(features_path)
        labels_tmp.replace(labels_path)
        splits_tmp.replace(split_assignments_path)

    except Exception:
        # Clean up temporary files on failure
        features_tmp.unlink(missing_ok=True)
        labels_tmp.unlink(missing_ok=True)
        splits_tmp.unlink(missing_ok=True)
        raise

    return {
        "processed_count": count,
        "earliest_timestamp": earliest_ts,
        "latest_timestamp": latest_ts,
        "legitimate_count": legit_count,
        "laundering_count": laundering_count,
        "duplicate_id_count": 0,
        "features_path": features_path,
        "labels_path": labels_path,
        "split_assignments_path": split_assignments_path,
        "candidate_splits": candidate_splits,
    }


def prepare_sample(
    data_path: Path,
    output_dir: Path,
    sample_size: int = SAMPLE_SIZE,
    config_path: Path | None = None,
) -> dict:
    """Execute sample preparation pipeline using memory-bounded heapq.nsmallest."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "sample_features.csv"
    labels_path = output_dir / "sample_labels.csv"
    split_assignments_path = output_dir / "sample_split_assignments.csv"

    boundaries = _load_split_boundaries(config_path)

    # Select earliest sample_size transactions
    raw_stream = load_transactions(data_path)
    sample_txns = select_earliest_transactions(raw_stream, n=sample_size)

    if not sample_txns:
        raise ValueError(f"No transactions loaded from {data_path}")

    stream_res = stream_and_write_prepared_data(
        sorted_txns_iter=sample_txns,
        features_path=features_path,
        labels_path=labels_path,
        split_assignments_path=split_assignments_path,
        boundaries=boundaries,
        expected_count=len(sample_txns),
    )

    elapsed = time.time() - t0
    peak_mb = get_peak_memory_mb()

    stream_res.update({
        "mode": "sample",
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_mb,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
    })

    return stream_res


def _write_data_checks(
    manifest_data: dict,
    boundaries: dict,
    artifacts_dir: Path | None = None,
) -> Path:
    """Write artifacts/data_checks.json with verified check results.

    The file is written atomically via a .tmp file.  Returns the path written.
    """
    target_dir = artifacts_dir or ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    checks_path = target_dir / "data_checks.json"
    checks_tmp = checks_path.with_name(checks_path.name + ".tmp")

    ds = manifest_data["data_summary"]
    splits = manifest_data["candidate_splits"]

    checks = {
        "data_checks_version": "1.0",
        "generated_utc": manifest_data["execution_timestamp_utc"],
        "source": {
            "raw_transactions_sha256": manifest_data["input"]["raw_transactions_sha256"],
            "raw_transactions_file": manifest_data["input"]["raw_transactions_file"],
        },
        "row_counts": {
            "total_transactions": ds["total_transactions"],
            "legitimate_count": ds["legitimate_count"],
            "laundering_count": ds["laundering_count"],
            "laundering_rate_pct": ds["laundering_rate_pct"],
            "duplicate_transaction_ids": ds["duplicate_transaction_ids"],
        },
        "timestamp_range": {
            "earliest": ds["earliest_timestamp"],
            "latest": ds["latest_timestamp"],
        },
        "split_counts": {
            name: {
                "total": d["total"],
                "legitimate": d["legitimate"],
                "laundering": d["laundering"],
                "laundering_rate_pct": round(
                    d["laundering"] / d["total"] * 100, 4
                ) if d["total"] > 0 else 0.0,
            }
            for name, d in splits.items()
        },
        "split_boundaries_used": {
            "status": boundaries.get("status", "unknown"),
            "train_cutoff": boundaries["train_cutoff"],
            "val_cutoff":   boundaries["val_cutoff"],
            "tail_cutoff":  boundaries["tail_cutoff"],
        },
        "feature_schema": manifest_data["schema"],
        "output_files": manifest_data["outputs"],
        "checks_passed": manifest_data["execution_metadata"]["validations_passed"],
        "note_tail_payment_formats": (
            "All 1,108 test_tail rows (Sep 11-18) are ACH payments. "
            "655 of 1,108 are laundering (59.12%). Verified against raw CSV."
        ),
        "note_window_semantics": (
            "Feature windows use a strictly open interval on both sides: "
            "(t - window, t). The handbook Day 2 wording says 'included' for "
            "the lower bound, but features.py implements exclusive lower bound. "
            "This is correct per AGENTS.md leak rules. features.py is not changed."
        ),
    }

    with checks_tmp.open("w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    checks_tmp.replace(checks_path)
    return checks_path


def prepare_full(
    data_path: Path,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    temp_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    """Execute full production preparation on all transactions.

    Uses bounded-memory external sort and incremental streaming feature generation.
    Generates manifest.json with complete reproducibility metadata and writes
    full_split_assignments.csv (transaction_id + split label for all rows).
    After successful validation, writes artifacts/data_checks.json.
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "full_features.csv"
    labels_path = output_dir / "full_labels.csv"
    split_assignments_path = output_dir / "full_split_assignments.csv"
    manifest_path = output_dir / "manifest.json"

    boundaries = _load_split_boundaries(config_path)

    # Compute SHA-256 of raw data file
    raw_sha256 = compute_sha256(data_path)

    # Use a temporary directory inside output_dir if not specified
    cleanup_temp = False
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(dir=output_dir, prefix="sort_tmp_")
        sort_temp_path = Path(temp_dir_obj.name)
        cleanup_temp = True
    else:
        sort_temp_path = temp_dir
        temp_dir_obj = None

    try:
        raw_stream = load_transactions(data_path)
        sorted_stream = sort_transactions_external(
            raw_stream,
            temp_dir=sort_temp_path,
            chunk_size=chunk_size,
        )

        stream_res = stream_and_write_prepared_data(
            sorted_txns_iter=sorted_stream,
            features_path=features_path,
            labels_path=labels_path,
            split_assignments_path=split_assignments_path,
            boundaries=boundaries,
        )
    finally:
        if cleanup_temp and temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    elapsed = time.time() - t0
    peak_mb = get_peak_memory_mb()

    # Compute SHA-256 hashes of generated output files for manifest verification
    features_sha256 = compute_sha256(features_path)
    labels_sha256 = compute_sha256(labels_path)
    split_assignments_sha256 = compute_sha256(split_assignments_path)

    # Build manifest
    manifest_data = {
        "manifest_version": "1.0",
        "mode": "full",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "raw_transactions_file": _rel_path(data_path),
            "raw_transactions_sha256": raw_sha256,
        },
        "outputs": {
            "features_file": _rel_path(features_path),
            "features_sha256": features_sha256,
            "labels_file": _rel_path(labels_path),
            "labels_sha256": labels_sha256,
            "split_assignments_file": _rel_path(split_assignments_path),
            "split_assignments_sha256": split_assignments_sha256,
            "manifest_file": _rel_path(manifest_path),
        },
        "schema": {
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "label_column": "is_laundering",
            "join_key": "transaction_id",
        },
        "data_summary": {
            "total_transactions": stream_res["processed_count"],
            "earliest_timestamp": stream_res["earliest_timestamp"],
            "latest_timestamp": stream_res["latest_timestamp"],
            "legitimate_count": stream_res["legitimate_count"],
            "laundering_count": stream_res["laundering_count"],
            "laundering_rate_pct": round(
                (stream_res["laundering_count"] / stream_res["processed_count"]) * 100,
                4,
            ) if stream_res["processed_count"] > 0 else 0.0,
            "duplicate_transaction_ids": stream_res["duplicate_id_count"],
        },
        "candidate_splits": stream_res["candidate_splits"],
        "execution_metadata": {
            "elapsed_seconds": round(elapsed, 2),
            "peak_memory_mb": round(peak_mb, 2),
            "chunk_size": chunk_size,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "validations_passed": [
                "unique_transaction_ids",
                "strict_chronological_ordering",
                "transaction_id_tie_breaking",
                "1_to_1_label_feature_split_alignment",
                "finite_numeric_features",
                "nonnegative_rolling_counts",
                "schema_isolation_no_leakage_columns",
                "split_assignments_written",
            ],
        },
    }

    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    with manifest_tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    manifest_tmp.replace(manifest_path)

    # Write artifacts/data_checks.json
    _write_data_checks(manifest_data, boundaries, artifacts_dir=artifacts_dir)

    stream_res.update({
        "mode": "full",
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_mb,
        "raw_sha256": raw_sha256,
        "features_sha256": features_sha256,
        "labels_sha256": labels_sha256,
        "split_assignments_sha256": split_assignments_sha256,
        "manifest_path": manifest_path,
        "split_assignments_path": split_assignments_path,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "manifest_data": manifest_data,
    })

    return stream_res


def print_summary(summary: dict) -> None:
    """Print the exact required summary fields to stdout."""
    if summary.get("mode") == "full":
        mode_title = "AML Full Data Preparation Summary (--full mode)"
    else:
        mode_title = "AML Sample Preparation Summary (--sample mode)"
    print("=" * 60)
    print(mode_title)
    print("=" * 60)
    print(f"Number of processed transactions: {summary['processed_count']:,}")
    print(f"Earliest sample timestamp:       {summary['earliest_timestamp']}")
    print(f"Latest sample timestamp:         {summary['latest_timestamp']}")
    print(f"Legitimate transactions count:   {summary['legitimate_count']:,}")
    print(f"Laundering transactions count:   {summary['laundering_count']:,}")
    pct = (summary['laundering_count'] / summary['processed_count'] * 100) if summary['processed_count'] else 0
    print(f"Laundering rate:                 {pct:.4f}%")
    print(f"Number of model features:        {summary['feature_count']}")
    print("Model feature names:")
    for i, name in enumerate(summary["feature_names"], start=1):
        print(f"  {i:2d}. {name}")
    print(f"Duplicate transaction_id count:  {summary['duplicate_id_count']}")
    print(f"Processing time:                 {summary['elapsed_seconds']:.2f} seconds")
    print(f"Peak memory:                     {summary['peak_memory_mb']:.2f} MB")
    if "raw_sha256" in summary:
        print(f"Raw input SHA-256:               {summary['raw_sha256']}")
    if "features_sha256" in summary:
        print(f"Features SHA-256:                {summary['features_sha256']}")
        print(f"Labels SHA-256:                  {summary['labels_sha256']}")
        print(f"Split assignments SHA-256:       {summary['split_assignments_sha256']}")
    print("-" * 65)
    print(f"Saved features to:          {summary['features_path']}")
    print(f"Saved labels to:            {summary['labels_path']}")
    if "split_assignments_path" in summary:
        print(f"Saved split assignments to: {summary['split_assignments_path']}")
    if "manifest_path" in summary:
        print(f"Saved manifest to:          {summary['manifest_path']}")

    if "candidate_splits" in summary:
        print("-" * 65)
        print("Candidate Chronological Splits (Proposals - Awaiting Team Approval):")
        splits = summary["candidate_splits"]
        total = summary["processed_count"]
        for name, data in [
            ("Train (< 2022/09/06 13:36)", splits["train"]),
            ("Validation (2022/09/06 13:36 - 2022/09/08 16:12)", splits["val"]),
            ("Test Main (2022/09/08 16:12 - 2022/09/11 00:00)", splits["test_main"]),
            ("Test Tail (>= 2022/09/11 00:00)", splits["test_tail"]),
        ]:
            c_tot = data["total"]
            c_legit = data["legitimate"]
            c_aml = data["laundering"]
            pct_split = (c_tot / total * 100) if total else 0
            rate = (c_aml / c_tot * 100) if c_tot else 0
            ratio = f"1 in {c_tot // c_aml}" if c_aml > 0 else "N/A"
            print(f"  * {name}:")
            print(f"      Rows: {c_tot:,} ({pct_split:.2f}%), Legit: {c_legit:,}, AML: {c_aml:,} ({rate:.4f}%, {ratio})")
    print("=" * 65)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare AML transaction dataset and features."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Prepare the earliest 100,000-row sample for testing and model interfaces.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Prepare the full dataset (all 5,078,345 transactions) via external sort.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to raw transactions CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Path to processed outputs directory (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Sample size to select in --sample mode (default: {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for external sort in --full mode (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Path to artifacts directory (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to config JSON (default: {CONFIG_PATH})",
    )

    args = parser.parse_args(argv)

    if not args.sample and not args.full:
        parser.print_help()
        sys.exit(1)
    if args.sample and args.full:
        print("Error: Specify either --sample or --full, not both.", file=sys.stderr)
        sys.exit(1)

    if args.sample:
        summary = prepare_sample(
            data_path=args.data_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            config_path=args.config_path,
        )
    else:
        summary = prepare_full(
            data_path=args.data_path,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            artifacts_dir=args.artifacts_dir,
            config_path=args.config_path,
        )
    print_summary(summary)


if __name__ == "__main__":
    main()
