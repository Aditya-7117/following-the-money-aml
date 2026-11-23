"""
tests/data/test_features.py
Tests for the streaming feature generator in src/data/features.py.

All tests use Python 3.12 stdlib unittest only.
Tests are written to spec before the implementation exists.
"""

import math
import unittest
from datetime import datetime

from src.data.features import FEATURE_COLUMNS, _AccountHistory, generate_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(year=2022, month=9, day=1, hour=0, minute=0):
    """Return a timestamp string in the format used by load.py."""
    return datetime(year, month, day, hour, minute).strftime("%Y/%m/%d %H:%M")


def _make_txn(
    transaction_id,
    timestamp,
    sender_bank="001",
    sender_account="A",
    receiver_bank="002",
    receiver_account="B",
    amount_paid=100.0,
    payment_currency="US Dollar",
    payment_format="Wire",
    is_laundering="0",
    amount_received=100.0,
    receiving_currency="US Dollar",
):
    """Build a transaction dict that matches the shape produced by load.py."""
    return {
        "transaction_id": transaction_id,
        "timestamp": timestamp,
        "sender_bank": sender_bank,
        "sender_account": sender_account,
        "receiver_bank": receiver_bank,
        "receiver_account": receiver_account,
        "amount_paid": str(amount_paid),
        "payment_currency": payment_currency,
        "payment_format": payment_format,
        "is_laundering": is_laundering,
        "amount_received": str(amount_received),
        "receiving_currency": receiving_currency,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureColumns(unittest.TestCase):
    """FEATURE_COLUMNS must list exactly the 10 model features, in order."""

    EXPECTED = [
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

    def test_column_names_and_order(self):
        self.assertEqual(list(FEATURE_COLUMNS), self.EXPECTED)

    def test_transaction_id_not_in_feature_columns(self):
        self.assertNotIn("transaction_id", FEATURE_COLUMNS)

    def test_timestamp_not_in_feature_columns(self):
        self.assertNotIn("timestamp", FEATURE_COLUMNS)

    def test_account_identifiers_not_in_feature_columns(self):
        for col in ("sender_bank", "sender_account", "receiver_bank", "receiver_account"):
            self.assertNotIn(col, FEATURE_COLUMNS)

    def test_features_constants_match_config_data_json(self):
        """Feature names, window durations and timestamp format must match config/data.json."""
        import json
        from pathlib import Path
        from src.data.features import TIMESTAMP_FORMAT, _ONE_HOUR, _TWENTY_FOUR_HOURS
        config_path = Path(__file__).resolve().parents[2] / "config" / "data.json"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(FEATURE_COLUMNS, cfg["feature_columns"])
        self.assertEqual(TIMESTAMP_FORMAT, cfg["rolling_window_rules"]["timestamp_format"])
        self.assertEqual(int(_ONE_HOUR.total_seconds()), cfg["rolling_window_rules"]["window_1h_seconds"])
        self.assertEqual(int(_TWENTY_FOUR_HOURS.total_seconds()), cfg["rolling_window_rules"]["window_24h_seconds"])


class TestEmptyInput(unittest.TestCase):
    """generate_features([]) must return an empty list without error."""

    def test_empty_returns_empty(self):
        result = list(generate_features([]))
        self.assertEqual(result, [])


class TestOutputShape(unittest.TestCase):
    """Each output row must contain transaction_id plus exactly the 10 features."""

    def _single(self):
        txn = _make_txn(1, _ts(hour=1))
        return list(generate_features([txn]))[0]

    def test_transaction_id_present(self):
        row = self._single()
        self.assertIn("transaction_id", row)

    def test_exactly_ten_feature_keys(self):
        row = self._single()
        feature_keys = {k for k in row if k != "transaction_id"}
        self.assertEqual(feature_keys, set(FEATURE_COLUMNS))

    def test_no_extra_keys(self):
        row = self._single()
        allowed = set(FEATURE_COLUMNS) | {"transaction_id"}
        self.assertEqual(set(row.keys()), allowed)

    def test_log_amount_paid_value(self):
        txn = _make_txn(1, _ts(hour=1), amount_paid=99.0)
        row = list(generate_features([txn]))[0]
        self.assertAlmostEqual(row["log_amount_paid"], math.log1p(99.0))

    def test_same_bank_true(self):
        txn = _make_txn(1, _ts(hour=1), sender_bank="010", receiver_bank="010")
        row = list(generate_features([txn]))[0]
        self.assertTrue(row["same_bank"])

    def test_same_bank_false(self):
        txn = _make_txn(1, _ts(hour=1), sender_bank="010", receiver_bank="020")
        row = list(generate_features([txn]))[0]
        self.assertFalse(row["same_bank"])

    def test_passthrough_currency_and_format(self):
        txn = _make_txn(1, _ts(hour=1), payment_currency="Euro", payment_format="ACH")
        row = list(generate_features([txn]))[0]
        self.assertEqual(row["payment_currency"], "Euro")
        self.assertEqual(row["payment_format"], "ACH")

    def test_transaction_id_preserved(self):
        txn = _make_txn(42, _ts(hour=1))
        row = list(generate_features([txn]))[0]
        self.assertEqual(row["transaction_id"], 42)


class TestFirstTransactionHasCounts(unittest.TestCase):
    """The very first transaction ever seen has no prior history."""

    def test_all_counts_zero(self):
        txn = _make_txn(1, _ts(hour=0, minute=0))
        row = list(generate_features([txn]))[0]
        self.assertEqual(row["sender_txn_count_1h"], 0)
        self.assertEqual(row["sender_txn_count_24h"], 0)
        self.assertEqual(row["sender_unique_receivers_24h"], 0)
        self.assertEqual(row["receiver_txn_count_1h"], 0)
        self.assertEqual(row["receiver_txn_count_24h"], 0)
        self.assertEqual(row["receiver_unique_senders_24h"], 0)


class TestExact1HourBoundary(unittest.TestCase):
    """Window uses strict inequality: transactions exactly 1 h old are excluded."""

    def test_exactly_one_hour_old_excluded(self):
        # t0 at 00:00, t1 at 01:00; window for t1 is (00:00, 01:00) exclusive
        t0 = _ts(hour=0, minute=0)
        t1 = _ts(hour=1, minute=0)
        txns = [
            _make_txn(1, t0),  # sender 001/A -> receiver 002/B
            _make_txn(2, t1),  # same sender
        ]
        rows = list(generate_features(txns))
        # t0 is not strictly inside (t1 - 1h, t1) so count must be 0
        self.assertEqual(rows[1]["sender_txn_count_1h"], 0)

    def test_one_minute_inside_window_counted(self):
        # t0 at 00:01, t1 at 01:00; t0 is 59 min before t1 -> inside window
        t0 = _ts(hour=0, minute=1)
        t1 = _ts(hour=1, minute=0)
        txns = [
            _make_txn(1, t0),
            _make_txn(2, t1),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["sender_txn_count_1h"], 1)


class TestExact24HourBoundary(unittest.TestCase):
    """Window uses strict inequality: transactions exactly 24 h old are excluded."""

    def test_exactly_24_hours_old_excluded(self):
        t0 = _ts(day=1, hour=0, minute=0)
        t1 = _ts(day=2, hour=0, minute=0)
        txns = [
            _make_txn(1, t0),
            _make_txn(2, t1),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)

    def test_one_minute_inside_24h_window_counted(self):
        t0 = _ts(day=1, hour=0, minute=1)
        t1 = _ts(day=2, hour=0, minute=0)
        txns = [
            _make_txn(1, t0),
            _make_txn(2, t1),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["sender_txn_count_24h"], 1)


class TestSameMinuteTransactions(unittest.TestCase):
    """All transactions with the same timestamp are scored using the same history
    snapshot -- none of them see each other in their counts.
    Only after all are scored does the history update."""

    def test_same_minute_sees_no_sibling_in_counts(self):
        # Three transactions at the same minute, same sender
        ts = _ts(hour=2)
        txns = [
            _make_txn(1, ts, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, ts, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
            _make_txn(3, ts, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="D"),
        ]
        rows = list(generate_features(txns))
        # No prior history, so all three must see count 0
        for row in rows:
            self.assertEqual(row["sender_txn_count_1h"], 0)
            self.assertEqual(row["sender_txn_count_24h"], 0)

    def test_same_minute_then_later_minute_sees_all(self):
        ts_early = _ts(hour=1, minute=0)
        ts_later = _ts(hour=1, minute=30)
        txns = [
            _make_txn(1, ts_early, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, ts_early, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
            _make_txn(3, ts_later, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="D"),
        ]
        rows = list(generate_features(txns))
        # txn 3 must see both earlier transactions in the 1h window
        self.assertEqual(rows[2]["sender_txn_count_1h"], 2)

    def test_order_of_same_minute_transactions_does_not_change_features(self):
        """Shuffling same-minute transactions must not change feature values
        for a given transaction_id."""
        ts = _ts(hour=3)
        base_txns = [
            _make_txn(1, ts, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, ts, sender_bank="001", sender_account="A",
                      receiver_bank="003", receiver_account="C"),
            _make_txn(3, ts, sender_bank="004", sender_account="D",
                      receiver_bank="001", receiver_account="A"),
        ]
        # Original order
        rows_orig = {r["transaction_id"]: r for r in generate_features(base_txns)}
        # Reversed order
        rows_rev = {r["transaction_id"]: r for r in generate_features(list(reversed(base_txns)))}

        for tid in (1, 2, 3):
            for col in FEATURE_COLUMNS:
                with self.subTest(transaction_id=tid, column=col):
                    self.assertEqual(rows_orig[tid][col], rows_rev[tid][col])


class TestDifferentBanksSameAccountNumber(unittest.TestCase):
    """Account identities are (bank, account_number) pairs.
    Accounts with the same number but different banks must be counted separately."""

    def test_same_account_number_different_banks_are_distinct_senders(self):
        # Bank 001 / account X sends at t0
        # Bank 002 / account X sends at t1 (same account number, different bank)
        t0 = _ts(hour=0, minute=0)
        t1 = _ts(hour=0, minute=30)
        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="X",
                      receiver_bank="003", receiver_account="Y"),
            _make_txn(2, t1, sender_bank="002", sender_account="X",
                      receiver_bank="003", receiver_account="Y"),
        ]
        rows = list(generate_features(txns))
        # txn 2's sender is 002/X -- it has no prior history under that identity
        self.assertEqual(rows[1]["sender_txn_count_1h"], 0)
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)

    def test_same_account_number_same_bank_is_counted(self):
        t0 = _ts(hour=0, minute=0)
        t1 = _ts(hour=0, minute=30)
        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="X",
                      receiver_bank="003", receiver_account="Y"),
            _make_txn(2, t1, sender_bank="001", sender_account="X",
                      receiver_bank="003", receiver_account="Y"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["sender_txn_count_1h"], 1)


class TestDistinctCountFeatures(unittest.TestCase):
    """sender_unique_receivers_24h and receiver_unique_senders_24h count distinct
    account identities, not individual transactions."""

    def test_sender_unique_receivers_distinct(self):
        t0 = _ts(hour=0, minute=0)
        t1 = _ts(hour=0, minute=5)
        t2 = _ts(hour=0, minute=10)
        t3 = _ts(hour=0, minute=15)
        txns = [
            # sender 001/A -> receiver 002/B  (first receiver)
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            # sender 001/A -> receiver 002/C  (second receiver)
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
            # sender 001/A -> receiver 002/B  (repeated, same receiver as txn 1)
            _make_txn(3, t2, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            # scored at t3: sender 001/A should have seen 2 distinct receivers
            _make_txn(4, t3, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="D"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[3]["sender_unique_receivers_24h"], 2)

    def test_receiver_unique_senders_distinct(self):
        t0 = _ts(hour=0, minute=0)
        t1 = _ts(hour=0, minute=5)
        t2 = _ts(hour=0, minute=10)
        t3 = _ts(hour=0, minute=15)
        txns = [
            # sender 001/X -> receiver 003/Z
            _make_txn(1, t0, sender_bank="001", sender_account="X",
                      receiver_bank="003", receiver_account="Z"),
            # sender 001/Y -> receiver 003/Z  (second distinct sender)
            _make_txn(2, t1, sender_bank="001", sender_account="Y",
                      receiver_bank="003", receiver_account="Z"),
            # sender 001/X -> receiver 003/Z  (repeated sender)
            _make_txn(3, t2, sender_bank="001", sender_account="X",
                      receiver_bank="003", receiver_account="Z"),
            # scored at t3: receiver 003/Z should have seen 2 distinct senders
            _make_txn(4, t3, sender_bank="001", sender_account="W",
                      receiver_bank="003", receiver_account="Z"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[3]["receiver_unique_senders_24h"], 2)


class TestFutureTransactionsDoNotChangeEarlierFeatures(unittest.TestCase):
    """Appending extra transactions after the fact must not alter the features
    already produced for earlier rows.  We simulate this by calling
    generate_features with different-length prefixes of the same list."""

    def test_adding_later_transaction_does_not_alter_earlier_feature(self):
        t0 = _ts(hour=1)
        t1 = _ts(hour=2)
        t2 = _ts(hour=3)
        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(3, t2, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
        ]
        rows_short = list(generate_features(txns[:2]))
        rows_long = list(generate_features(txns))

        for col in FEATURE_COLUMNS:
            with self.subTest(column=col):
                self.assertEqual(rows_short[0][col], rows_long[0][col])
                self.assertEqual(rows_short[1][col], rows_long[1][col])


class TestChangingLabelDoesNotChangeFeatures(unittest.TestCase):
    """Flipping is_laundering must not affect any of the 10 model features."""

    def test_label_flip_leaves_features_unchanged(self):
        ts = _ts(hour=5)
        txn_legit = _make_txn(1, ts, is_laundering="0")
        txn_launder = _make_txn(1, ts, is_laundering="1")

        row_legit = list(generate_features([txn_legit]))[0]
        row_launder = list(generate_features([txn_launder]))[0]

        for col in FEATURE_COLUMNS:
            with self.subTest(column=col):
                self.assertEqual(row_legit[col], row_launder[col])


class TestOutOfOrderRejection(unittest.TestCase):
    """Out-of-order input must raise ValueError, not silently produce wrong counts."""

    def test_strictly_out_of_order_raises(self):
        t_later = _ts(hour=2)
        t_earlier = _ts(hour=1)
        txns = [
            _make_txn(1, t_later),
            _make_txn(2, t_earlier),
        ]
        with self.assertRaises(ValueError):
            list(generate_features(txns))

    def test_same_timestamp_accepted(self):
        ts = _ts(hour=2)
        txns = [
            _make_txn(1, ts),
            _make_txn(2, ts),
        ]
        # Must not raise
        result = list(generate_features(txns))
        self.assertEqual(len(result), 2)


class TestMultipleTransactionsCounting(unittest.TestCase):
    """Integration test: verify counts accumulate correctly across distinct minutes."""

    def test_sender_count_grows_with_history(self):
        txns = [
            _make_txn(i + 1, _ts(hour=0, minute=i),
                      sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B")
            for i in range(5)
        ]
        rows = list(generate_features(txns))
        counts = [r["sender_txn_count_1h"] for r in rows]
        # Each successive transaction sees one more prior transaction in the window
        self.assertEqual(counts, [0, 1, 2, 3, 4])

    def test_receiver_count_grows_with_history(self):
        txns = [
            _make_txn(i + 1, _ts(hour=0, minute=i),
                      sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B")
            for i in range(5)
        ]
        rows = list(generate_features(txns))
        counts = [r["receiver_txn_count_1h"] for r in rows]
        self.assertEqual(counts, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# New tests for scalability refactoring
# ---------------------------------------------------------------------------

class TestGenuineStreaming(unittest.TestCase):
    """generate_features must yield rows as minutes complete, not all at the end.

    We drive the generator manually with next() to confirm that the first
    minute's output is produced after reading the first transaction of the
    next minute, but before consuming a third transaction.
    """

    def test_first_minute_yielded_after_second_read_before_third_consumed(self):
        """The first minute's output is yielded after reading the first transaction
        of the next minute, but before consuming a third transaction."""
        t0 = _ts(hour=1, minute=0)
        t1 = _ts(hour=1, minute=1)
        t2 = _ts(hour=1, minute=2)

        consumed = []

        def tracking_iter():
            for txn in [
                _make_txn(1, t0),
                _make_txn(2, t1),
                _make_txn(3, t2),
            ]:
                consumed.append(txn["transaction_id"])
                yield txn

        gen = generate_features(tracking_iter())

        # Pull the first row (from minute 1:00).
        # To know minute 1:00 is complete, generate_features must read txn 2 (1:01).
        # But it must yield row 1 BEFORE reading txn 3 (1:02).
        row1 = next(gen)
        self.assertEqual(row1["transaction_id"], 1)
        self.assertEqual(consumed, [1, 2])
        self.assertNotIn(3, consumed)

        # Pull the second row (from minute 1:01).
        # To know minute 1:01 is complete, generate_features must read txn 3 (1:02).
        row2 = next(gen)
        self.assertEqual(row2["transaction_id"], 2)
        self.assertEqual(consumed, [1, 2, 3])

        # Pull the third row (from minute 1:02, stream ends).
        row3 = next(gen)
        self.assertEqual(row3["transaction_id"], 3)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_streaming_multiple_minutes(self):
        """Rows from each minute are immediately available after that minute ends."""
        minutes = 5
        txns = [
            _make_txn(i + 1, _ts(hour=0, minute=i))
            for i in range(minutes)
        ]
        gen = generate_features(iter(txns))
        for expected_id in range(1, minutes + 1):
            row = next(gen)
            self.assertEqual(row["transaction_id"], expected_id)

        with self.assertRaises(StopIteration):
            next(gen)


class TestAccountHistoryRemoval(unittest.TestCase):
    """When an account's 24-h count reaches zero and other counters are empty,
    its _AccountHistory must be removed from the dictionary.
    """

    def test_account_history_is_empty_lifecycle(self):
        """_AccountHistory.is_empty() reflects counter lifecycle accurately."""
        h = _AccountHistory()
        self.assertTrue(h.is_empty())

        counterpart = ("002", "B")
        h.record(counterpart)
        self.assertFalse(h.is_empty())
        self.assertEqual(h.count_1h, 1)
        self.assertEqual(h.count_24h, 1)

        # 1-h expiry: count_1h drops to 0, but count_24h remains 1 -> not empty
        h.expire_1h()
        self.assertFalse(h.is_empty())
        self.assertEqual(h.count_1h, 0)
        self.assertEqual(h.count_24h, 1)

        # 24-h expiry: count_24h drops to 0 and counterpart removed -> empty
        h.expire_24h(counterpart)
        self.assertTrue(h.is_empty())
        self.assertEqual(h.count_24h, 0)
        self.assertEqual(len(h._freq), 0)

    def test_account_removed_from_dictionary_after_24h(self):
        """Inactive accounts are deleted from sender_hist / receiver_hist after 24h."""
        t0 = _ts(day=1, hour=0, minute=0)
        # Advance time past 24h: day 2, 00:01
        t_after = _ts(day=2, hour=0, minute=1)

        s_hist = {}
        r_hist = {}

        # Txn 1: 001/A -> 002/B at day 1 00:00
        # Txn 2: 003/C -> 004/D at day 2 00:01 (triggers 24h expiry of Txn 1)
        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A", receiver_bank="002", receiver_account="B"),
            _make_txn(2, t_after, sender_bank="003", sender_account="C", receiver_bank="004", receiver_account="D"),
        ]

        rows = list(generate_features(txns, _sender_hist=s_hist, _receiver_hist=r_hist))
        self.assertEqual(len(rows), 2)

        # 001/A and 002/B have reached 24h count 0 with all counters empty -> removed!
        self.assertNotIn(("001", "A"), s_hist)
        self.assertNotIn(("002", "B"), r_hist)

        # 003/C and 004/D transacted at t_after -> present in dictionaries
        self.assertIn(("003", "C"), s_hist)
        self.assertIn(("004", "D"), r_hist)

    def test_account_not_removed_if_partially_expired(self):
        """Account with two transactions is not removed when only the first expires."""
        t1 = _ts(day=1, hour=0, minute=0)
        t2 = _ts(day=1, hour=12, minute=0)
        # t_probe is day 2 00:01 (24h after t1, but only 12h after t2)
        t_probe = _ts(day=2, hour=0, minute=1)
        # t_final is day 2 12:01 (24h after t2)
        t_final = _ts(day=2, hour=12, minute=1)

        s_hist = {}

        txns = [
            _make_txn(1, t1, sender_bank="001", sender_account="A", receiver_bank="002", receiver_account="B"),
            _make_txn(2, t2, sender_bank="001", sender_account="A", receiver_bank="002", receiver_account="C"),
            _make_txn(3, t_probe, sender_bank="099", sender_account="Z", receiver_bank="099", receiver_account="Y"),
            _make_txn(4, t_final, sender_bank="099", sender_account="Z", receiver_bank="099", receiver_account="Y"),
        ]

        gen = generate_features(txns, _sender_hist=s_hist)

        # Process through t_probe (txns 1, 2, 3)
        row1 = next(gen)
        row2 = next(gen)
        row3 = next(gen)

        # At t_probe (row 3): t1 expired, but t2 has NOT expired -> 001/A still in s_hist
        self.assertIn(("001", "A"), s_hist)
        self.assertEqual(s_hist[("001", "A")].count_24h, 1)

        # At t_final (row 4): t2 expires -> 001/A reaches 0 and is removed
        row4 = next(gen)
        self.assertNotIn(("001", "A"), s_hist)

    def test_removed_account_can_transact_again_fresh(self):
        """An account removed after 24h can transact later with clean 0-counts."""
        t1 = _ts(day=1, hour=0, minute=0)
        t2 = _ts(day=2, hour=1, minute=0)  # 25 hours later
        t3 = _ts(day=2, hour=1, minute=5)

        s_hist = {}
        txns = [
            _make_txn(1, t1, sender_bank="001", sender_account="A"),
            _make_txn(2, t2, sender_bank="001", sender_account="A"),
            _make_txn(3, t3, sender_bank="001", sender_account="A"),
        ]

        rows = list(generate_features(txns, _sender_hist=s_hist))
        # First txn: counts 0
        self.assertEqual(rows[0]["sender_txn_count_24h"], 0)
        # Second txn (after expiry + removal): counts start clean at 0
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)
        # Third txn (5 mins later): sees txn 2 in history
        self.assertEqual(rows[2]["sender_txn_count_24h"], 1)



class TestExpiryCorrectness(unittest.TestCase):
    """Counts must drop to zero once all contributing transactions leave the window."""

    def test_sender_count_drops_after_1h_window_expires(self):
        # sender 001/A transacts at 00:01 and 00:02.
        # We probe with a DIFFERENT sender (001/Z) so the probe transaction
        # itself does not appear in 001/A's history.
        #
        # At t_check1 = 01:02:
        #   window for sender 001/A = (01:02 - 1h, 01:02) = (00:02, 01:02) strictly.
        #   t0 = 00:01: NOT in window (00:01 < 00:02).
        #   t1 = 00:02: NOT in window (boundary, excluded).
        #   => count_1h = 0.
        #
        # At t_check2 = 01:03:
        #   window = (00:03, 01:03). Both 00:01 and 00:02 are outside.
        #   => count_1h = 0.
        t0 = _ts(hour=0, minute=1)
        t1 = _ts(hour=0, minute=2)
        t_check1 = _ts(hour=1, minute=2)
        t_check2 = _ts(hour=1, minute=3)

        txns_a = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(3, t_check1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
        ]
        rows_a = list(generate_features(txns_a))
        # At t_check1 (01:02), the strict 1h window is (00:02, 01:02).
        # t0=00:01 < 00:02 (expired), t1=00:02 == boundary (excluded) => count = 0.
        self.assertEqual(rows_a[2]["sender_txn_count_1h"], 0)

        txns_b = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(3, t_check2, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
        ]
        rows_b = list(generate_features(txns_b))
        # At t_check2 (01:03), the strict 1h window is (00:03, 01:03).
        # Both t0=00:01 and t1=00:02 are outside => count = 0.
        self.assertEqual(rows_b[2]["sender_txn_count_1h"], 0)

    def test_sender_count_drops_after_24h_window_expires(self):
        t0 = _ts(day=1, hour=0, minute=1)
        t_check = _ts(day=2, hour=0, minute=1)  # exactly 24h later — t0 just expired

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t_check, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
        ]
        rows = list(generate_features(txns))
        # t_check - 24h = t0, so t0 is on the boundary and excluded.
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)

    def test_unique_counterparts_drops_after_24h_expiry(self):
        # sender 001/A sends to 002/B at day1 00:01, to 002/C at day1 00:02.
        # At day2 00:03 (24h2m and 24h1m later respectively), both 24h entries
        # have expired (cutoff = day2 00:03 - 24h = day1 00:03; both < cutoff).
        t0 = _ts(day=1, hour=0, minute=1)
        t1 = _ts(day=1, hour=0, minute=2)
        t_check = _ts(day=2, hour=0, minute=3)

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
            _make_txn(3, t_check, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="D"),
        ]
        rows = list(generate_features(txns))
        # At t_check: cutoff24 = day2 00:03 - 24h = day1 00:03.
        # t0 = day1 00:01 <= day1 00:03 => expired.
        # t1 = day1 00:02 <= day1 00:03 => expired.
        # unique count must be 0.
        self.assertEqual(rows[2]["sender_unique_receivers_24h"], 0)

    def test_partial_counterpart_expiry_keeps_surviving_entry(self):
        # sender 001/A sends to 002/B at 00:01 and to 002/B again at 00:30.
        # At 01:02, the 00:01 entry is expired but 00:30 is still inside.
        # 002/B is still a distinct counterpart because of the 00:30 entry.
        t0 = _ts(hour=0, minute=1)
        t1 = _ts(hour=0, minute=30)
        t_check = _ts(hour=1, minute=2)

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t1, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(3, t_check, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
        ]
        rows = list(generate_features(txns))
        # At t_check: 00:01 has expired, 00:30 remains → 002/B is still seen
        self.assertEqual(rows[2]["sender_unique_receivers_24h"], 1)
        self.assertEqual(rows[2]["sender_txn_count_1h"], 1)


class TestPreviouslyInactiveAccounts(unittest.TestCase):
    """An account that was idle for more than 24 h must show zero counts
    when it becomes active again."""

    def test_sender_sees_zero_after_long_gap(self):
        t0 = _ts(day=1, hour=0, minute=0)
        t_gap = _ts(day=3, hour=0, minute=0)  # 48 h later

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t_gap, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["sender_txn_count_1h"], 0)
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)
        self.assertEqual(rows[1]["sender_unique_receivers_24h"], 0)

    def test_receiver_sees_zero_after_long_gap(self):
        t0 = _ts(day=1, hour=0, minute=0)
        t_gap = _ts(day=3, hour=0, minute=0)

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t_gap, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[1]["receiver_txn_count_1h"], 0)
        self.assertEqual(rows[1]["receiver_txn_count_24h"], 0)
        self.assertEqual(rows[1]["receiver_unique_senders_24h"], 0)

    def test_inactive_account_not_confused_with_other_account(self):
        """While 001/A is inactive, 001/C builds its own history independently."""
        t0 = _ts(day=1, hour=0, minute=0)
        t1 = _ts(day=1, hour=0, minute=10)
        t_gap = _ts(day=3, hour=0, minute=0)

        txns = [
            # 001/A sends once
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            # 001/C sends once, 10 min later
            _make_txn(2, t1, sender_bank="001", sender_account="C",
                      receiver_bank="002", receiver_account="B"),
            # 001/A returns after a long gap; must see 0, not 001/C's history
            _make_txn(3, t_gap, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
        ]
        rows = list(generate_features(txns))
        self.assertEqual(rows[2]["sender_txn_count_24h"], 0)


class TestCounterpartWindowReentry(unittest.TestCase):
    """A counterpart that leaves and re-enters a window must be counted correctly
    at each stage."""

    def test_counterpart_re_enters_unique_count_correctly(self):
        # sender 001/A -> 002/B at 00:01 (enters 24h window)
        # At 24:02 the 00:01 entry expires; 002/B is no longer in the window.
        # sender 001/A -> 002/B at 24:30 (re-enters the 24h window).
        # At 25:00, the window should contain only the 24:30 entry → count 1.
        t0 = _ts(day=1, hour=0, minute=1)
        t_reenter = _ts(day=2, hour=0, minute=30)  # 24:30 relative to day 1 00:00
        t_check = _ts(day=2, hour=1, minute=0)     # 25:00

        txns = [
            _make_txn(1, t0, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(2, t_reenter, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B"),
            _make_txn(3, t_check, sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C"),
        ]
        rows = list(generate_features(txns))
        # At t_reenter: t0 is 24h29m before t_reenter; cutoff is t_reenter-24h
        # = day1 00:30. t0 = day1 00:01 < day1 00:30, so t0 is expired.
        # t_reenter itself is NOT yet in history when row 2 is scored.
        self.assertEqual(rows[1]["sender_txn_count_24h"], 0)
        self.assertEqual(rows[1]["sender_unique_receivers_24h"], 0)

        # At t_check (25:00): t_reenter is 30 min old → inside both windows.
        self.assertEqual(rows[2]["sender_txn_count_1h"], 1)
        self.assertEqual(rows[2]["sender_txn_count_24h"], 1)
        self.assertEqual(rows[2]["sender_unique_receivers_24h"], 1)

    def test_repeated_counterpart_unique_count_never_exceeds_one(self):
        # sender 001/A -> 002/B at 00:01, 00:02, 00:03 — all same counterpart.
        # At 00:04 unique count must still be 1.
        txns = [
            _make_txn(i + 1, _ts(hour=0, minute=i + 1),
                      sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="B")
            for i in range(3)
        ]
        # Probe at 00:04
        txns.append(
            _make_txn(4, _ts(hour=0, minute=4),
                      sender_bank="001", sender_account="A",
                      receiver_bank="002", receiver_account="C")
        )
        rows = list(generate_features(txns))
        self.assertEqual(rows[3]["sender_unique_receivers_24h"], 1)


if __name__ == "__main__":
    unittest.main()
