"""
src/data/features.py
Streaming feature generator for the AML transaction pipeline.

Accepts transactions already sorted in chronological order (same-minute
transactions may appear in any order within that minute).

Produces exactly 10 model features per transaction plus transaction_id for
joining. Timestamps and account identifiers are excluded from model inputs.

Account identity is always the pair (bank_id, account_id). Using
account_id alone is incorrect because eight account numbers appear at more
than one bank in this dataset.

Window semantics: for a transaction at time t, only prior transactions with
    t - window  <  previous_timestamp  <  t
contribute to the window count (both bounds exclusive). Transactions sharing
timestamp t are scored BEFORE any of them updates historical state, so
none of them appear in each other's counts.

Out-of-order input raises ValueError immediately rather than silently
producing incorrect histories.

Design (amortised O(1) per transaction):
- Each _AccountHistory maintains running integer counters (count_1h,
  count_24h) and a Counter for counterpart reference counts inside the
  24-h window (len(counter) == unique_counterparts_24h). Snapshot reads
  are O(1): they return the counters directly.
- A global chronological expiry queue (two deques: one per window size)
  is kept in expiry-timestamp order. When time advances to `now`, all
  entries with expiry_ts <= now are popped in one left-to-right sweep and
  the corresponding account counters are decremented. Inactive accounts
  therefore expire as soon as any later transaction arrives — no per-account
  lazy pruning is needed.
- Memory is proportional to active 24-h history across all accounts, not
  to total transactions ever processed.
"""

import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

# The 10 model features in the agreed column order.
FEATURE_COLUMNS = [
    "log_amount_paid",
    "payment_currency",
    "payment_format",
    "same_bank",
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_unique_receivers_24h",
    "receiver_txn_count_1h",
    "receiver_txn_count_24h",
    "receiver_unique_senders_24h",
]

_ONE_HOUR = timedelta(hours=1)
_TWENTY_FOUR_HOURS = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Internal rolling-window state
# ---------------------------------------------------------------------------

class _AccountHistory:
    """Per-account rolling state with O(1) snapshot reads.

    Running counters are incremented at commit time (record()) and decremented
    at expiry time (expire_1h() / expire_24h()), both driven by the global
    expiry logic in generate_features.  The object never scans its own history.

    Attributes
    ----------
    count_1h : int
        Number of transactions in the strict 1-h window.
    count_24h : int
        Number of transactions in the strict 24-h window.
    _freq : Counter[tuple]
        Counterpart reference counts inside the 24-h window.
        len(_freq) is the number of unique counterparts.
    """

    __slots__ = ("count_1h", "count_24h", "_freq")

    def __init__(self):
        self.count_1h: int = 0
        self.count_24h: int = 0
        self._freq: Counter = Counter()

    def is_empty(self) -> bool:
        """Return True if all rolling counts and counterpart frequencies are empty."""
        return self.count_24h == 0 and self.count_1h == 0 and not self._freq

    # ------------------------------------------------------------------
    # Read -- O(1)
    # ------------------------------------------------------------------

    def snapshot(self) -> tuple[int, int, int]:
        """Return (txn_count_1h, txn_count_24h, unique_counterparts_24h).

        The caller must ensure global expiry has been applied for the current
        time before calling this method so that expired entries are not counted.
        """
        return self.count_1h, self.count_24h, len(self._freq)

    # ------------------------------------------------------------------
    # Write: called by generate_features after scoring a minute
    # ------------------------------------------------------------------

    def record(self, counterpart: tuple) -> None:
        """Increment all counters for a newly committed transaction."""
        self.count_1h += 1
        self.count_24h += 1
        self._freq[counterpart] += 1

    # ------------------------------------------------------------------
    # Expiry: called by the global expiry sweep in generate_features
    # ------------------------------------------------------------------

    def expire_1h(self) -> None:
        """Decrement the 1-h transaction counter for one expired entry."""
        self.count_1h -= 1

    def expire_24h(self, counterpart: tuple) -> None:
        """Decrement the 24-h counter and counterpart frequency for one expired entry."""
        self.count_24h -= 1
        self._freq[counterpart] -= 1
        if self._freq[counterpart] == 0:
            del self._freq[counterpart]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_features(transactions, *, _sender_hist=None, _receiver_hist=None):
    """Yield one feature-row dict per transaction, in input order.

    Parameters
    ----------
    transactions : iterable of dict
        Transaction dicts as produced by ``src.data.load.load_transactions``.
        Must be in non-decreasing timestamp order. Out-of-order input raises
        ``ValueError``.

    Yields
    ------
    dict
        Keys: ``transaction_id`` plus the 10 names listed in
        ``FEATURE_COLUMNS``. The ``is_laundering`` label, raw amounts other
        than ``amount_paid``, and all account-identifier fields are excluded.

    Notes
    -----
    Rows for a completed minute are yielded as soon as the first transaction
    of the *next* minute is encountered (or at stream end). The generator is
    genuinely streaming: callers receive rows incrementally rather than waiting
    for all input to be consumed first.
    """
    # Per-account history keyed by (bank, account_number).
    sender_hist: dict[tuple, _AccountHistory] = {} if _sender_hist is None else _sender_hist
    receiver_hist: dict[tuple, _AccountHistory] = {} if _receiver_hist is None else _receiver_hist

    # Global chronological expiry queues.
    # Entry format: (expiry_datetime, account_id, role, counterpart)
    #   role == "s" -> sender_hist, role == "r" -> receiver_hist
    # Entries are appended in non-decreasing expiry order because input is
    # chronological, so popleft() always removes the oldest entry first.
    expiry_1h: deque[tuple[datetime, tuple, str, tuple]] = deque()
    expiry_24h: deque[tuple[datetime, tuple, str, tuple]] = deque()

    last_ts: datetime | None = None
    minute_buffer: list[tuple[dict, tuple, tuple]] = []
    current_minute_ts: datetime | None = None

    def _expire_global(now: datetime) -> None:
        """Pop all expiry entries whose deadline <= now and decrement counters.

        This enforces the strict lower-bound semantics: entries recorded at
        time t expire at t + window, so at t + window they are no longer
        inside the open interval (t + window - window, t + window) = (t, t+w).
        When an account's 24-hour count reaches zero and its other counters are
        empty, its _AccountHistory is removed from the dictionary.
        """
        while expiry_1h and expiry_1h[0][0] <= now:
            _, acct_id, role, _ = expiry_1h.popleft()
            hist = sender_hist if role == "s" else receiver_hist
            if acct_id in hist:
                acct = hist[acct_id]
                acct.expire_1h()
                if acct.is_empty():
                    del hist[acct_id]

        while expiry_24h and expiry_24h[0][0] <= now:
            _, acct_id, role, counterpart = expiry_24h.popleft()
            hist = sender_hist if role == "s" else receiver_hist
            if acct_id in hist:
                acct = hist[acct_id]
                acct.expire_24h(counterpart)
                if acct.is_empty():
                    del hist[acct_id]

    def _flush_minute():
        """Score all transactions in minute_buffer; return their feature rows.

        Steps:
        1. Apply global expiry up to current_minute_ts (evicts entries with
           deadline <= ts, preserving strict window semantics).
        2. Read O(1) snapshots for each transaction (all share the same ts,
           and none of them have been committed yet -- same-minute atomicity).
        3. Commit all transactions and enqueue their expiry entries.
        """
        ts = current_minute_ts
        _expire_global(ts)

        rows = []
        for txn, sender_id, receiver_id in minute_buffer:
            s_hist = sender_hist.get(sender_id)
            s_1h, s_24h, s_uniq = s_hist.snapshot() if s_hist is not None else (0, 0, 0)
            r_hist = receiver_hist.get(receiver_id)
            r_1h, r_24h, r_uniq = r_hist.snapshot() if r_hist is not None else (0, 0, 0)

            rows.append({
                "transaction_id": txn["transaction_id"],
                "log_amount_paid": math.log1p(float(txn["amount_paid"])),
                "payment_currency": txn["payment_currency"],
                "payment_format": txn["payment_format"],
                "same_bank": txn["sender_bank"] == txn["receiver_bank"],
                "sender_txn_count_1h": s_1h,
                "sender_txn_count_24h": s_24h,
                "sender_unique_receivers_24h": s_uniq,
                "receiver_txn_count_1h": r_1h,
                "receiver_txn_count_24h": r_24h,
                "receiver_unique_senders_24h": r_uniq,
            })

        # Commit all after scoring to preserve same-minute atomicity.
        exp_1h_ts = ts + _ONE_HOUR
        exp_24h_ts = ts + _TWENTY_FOUR_HOURS
        for txn, sender_id, receiver_id in minute_buffer:
            if sender_id not in sender_hist:
                sender_hist[sender_id] = _AccountHistory()
            sender_hist[sender_id].record(receiver_id)

            if receiver_id not in receiver_hist:
                receiver_hist[receiver_id] = _AccountHistory()
            receiver_hist[receiver_id].record(sender_id)

            expiry_1h.append((exp_1h_ts, sender_id, "s", receiver_id))
            expiry_1h.append((exp_1h_ts, receiver_id, "r", sender_id))
            expiry_24h.append((exp_24h_ts, sender_id, "s", receiver_id))
            expiry_24h.append((exp_24h_ts, receiver_id, "r", sender_id))

        return rows

    for txn in transactions:
        ts = datetime.strptime(txn["timestamp"], TIMESTAMP_FORMAT)

        # Enforce chronological order.
        if last_ts is not None and ts < last_ts:
            raise ValueError(
                f"Out-of-order transaction: timestamp {txn['timestamp']!r} "
                f"is before previous timestamp "
                f"{last_ts.strftime(TIMESTAMP_FORMAT)!r}."
            )

        # When the minute advances, flush the completed minute and yield immediately.
        if current_minute_ts is not None and ts != current_minute_ts:
            yield from _flush_minute()
            minute_buffer.clear()

        current_minute_ts = ts
        last_ts = ts

        sender_id = (txn["sender_bank"], txn["sender_account"])
        receiver_id = (txn["receiver_bank"], txn["receiver_account"])
        minute_buffer.append((txn, sender_id, receiver_id))

    # Flush any transactions remaining in the final minute.
    if minute_buffer:
        yield from _flush_minute()
