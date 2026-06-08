"""Tests for agent-cost-guard.

These tests use only the Python standard library ``unittest`` module so they
run with::

    python3 -m unittest discover -s tests

No third-party test dependencies are required.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_cost_guard import (  # noqa: E402
    CostEntry,
    CostGuard,
    CostLimitExceeded,
    CostSummary,
    CostWarning,
    make_cost_guard,
)


class TestConstruction(unittest.TestCase):
    def test_basic_init(self):
        guard = CostGuard(limit_usd=1.00)
        self.assertEqual(guard.limit_usd, 1.00)
        self.assertEqual(guard.total_usd, 0.0)
        self.assertTrue(guard.ok)

    def test_invalid_limit_zero(self):
        with self.assertRaises(ValueError):
            CostGuard(limit_usd=0)

    def test_invalid_limit_negative(self):
        with self.assertRaises(ValueError):
            CostGuard(limit_usd=-1.0)


class TestAdd(unittest.TestCase):
    def test_add_single(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.05)
        self.assertAlmostEqual(guard.total_usd, 0.05)

    def test_add_multiple(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.05)
        guard.add(0.10)
        self.assertAlmostEqual(guard.total_usd, 0.15)

    def test_add_returns_self(self):
        guard = CostGuard(limit_usd=1.0)
        result = guard.add(0.01)
        self.assertIs(result, guard)

    def test_add_chaining(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.01).add(0.02).add(0.03)
        self.assertAlmostEqual(guard.total_usd, 0.06)

    def test_add_zero(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.0)
        self.assertEqual(guard.total_usd, 0.0)

    def test_add_negative_raises(self):
        guard = CostGuard(limit_usd=1.0)
        with self.assertRaises(ValueError):
            guard.add(-0.01)


class TestLimitEnforcement(unittest.TestCase):
    def test_limit_exceeded_raises(self):
        guard = CostGuard(limit_usd=0.10, stop_on_limit=True)
        guard.add(0.05)
        with self.assertRaises(CostLimitExceeded):
            guard.add(0.06)

    def test_limit_exceeded_message_attrs(self):
        guard = CostGuard(limit_usd=0.10)
        with self.assertRaises(CostLimitExceeded) as ctx:
            guard.add(0.20, label="big_call")
        err = ctx.exception
        self.assertAlmostEqual(err.total_usd, 0.20)
        self.assertAlmostEqual(err.limit_usd, 0.10)
        self.assertEqual(err.label, "big_call")
        self.assertIn("big_call", str(err))

    def test_stop_on_limit_false(self):
        guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
        guard.add(0.20)  # no exception
        self.assertAlmostEqual(guard.total_usd, 0.20)
        self.assertFalse(guard.ok)

    def test_exactly_at_limit_ok(self):
        guard = CostGuard(limit_usd=0.10)
        guard.add(0.10)
        self.assertTrue(guard.ok)

    def test_check_raises_when_over(self):
        guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
        guard.add(0.20)
        with self.assertRaises(CostLimitExceeded):
            guard.check()

    def test_check_ok_when_under(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.5)
        guard.check()  # no exception

    def test_entry_recorded_even_when_limit_exceeded(self):
        # The over-limit entry should still be stored before raising, so the
        # final total reflects the call that tripped the guard.
        guard = CostGuard(limit_usd=0.10)
        with self.assertRaises(CostLimitExceeded):
            guard.add(0.20)
        self.assertEqual(guard.entry_count, 1)
        self.assertAlmostEqual(guard.total_usd, 0.20)


class TestRemainingAndPct(unittest.TestCase):
    def test_remaining_usd(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.25)
        self.assertAlmostEqual(guard.remaining_usd, 0.75)

    def test_remaining_negative_when_over(self):
        guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
        guard.add(0.20)
        self.assertAlmostEqual(guard.remaining_usd, -0.10)

    def test_pct_used_zero(self):
        guard = CostGuard(limit_usd=1.0)
        self.assertEqual(guard.pct_used, 0.0)

    def test_pct_used_half(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.5)
        self.assertAlmostEqual(guard.pct_used, 0.5)

    def test_pct_used_full(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(1.0)
        self.assertAlmostEqual(guard.pct_used, 1.0)


class TestEntries(unittest.TestCase):
    def test_entries_empty(self):
        guard = CostGuard(limit_usd=1.0)
        self.assertEqual(guard.entries, [])

    def test_entries_stored(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.05, label="turn1")
        guard.add(0.10, label="turn2")
        self.assertEqual(guard.entry_count, 2)
        self.assertEqual(guard.entries[0].label, "turn1")
        self.assertEqual(guard.entries[1].label, "turn2")

    def test_entries_returns_copy(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.01)
        entries = guard.entries
        entries.clear()
        self.assertEqual(guard.entry_count, 1)

    def test_entry_cost_stored(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.07, label="search")
        self.assertIsInstance(guard.entries[0], CostEntry)
        self.assertAlmostEqual(guard.entries[0].cost_usd, 0.07)
        self.assertEqual(guard.entries[0].label, "search")

    def test_entry_timestamp_set(self):
        before = time.time()
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.01)
        after = time.time()
        ts = guard.entries[0].ts
        self.assertTrue(before <= ts <= after)


class TestReset(unittest.TestCase):
    def test_reset_clears_total(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.5)
        guard.reset()
        self.assertEqual(guard.total_usd, 0.0)

    def test_reset_clears_entries(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.5)
        guard.reset()
        self.assertEqual(guard.entry_count, 0)

    def test_reset_clears_fired_thresholds(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
        guard.add(0.6)  # fires warning
        guard.reset()
        guard.add(0.6)  # should fire again after reset
        self.assertEqual(len(warnings), 2)


class TestWarnings(unittest.TestCase):
    def test_warn_at_single_fires(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
        guard.add(0.6)
        self.assertEqual(len(warnings), 1)
        self.assertIsInstance(warnings[0], CostWarning)
        self.assertEqual(warnings[0].threshold, 0.5)

    def test_warn_at_only_fires_once(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
        guard.add(0.4)
        guard.add(0.2)  # crosses 0.5 here
        guard.add(0.1)
        self.assertEqual(len(warnings), 1)

    def test_warn_at_multiple(self):
        warnings = []
        guard = CostGuard(
            limit_usd=1.0, warn_at=[0.5, 0.8], on_warn=warnings.append
        )
        guard.add(0.6)  # crosses 0.5
        guard.add(0.3)  # crosses 0.8
        self.assertEqual(len(warnings), 2)
        thresholds = {w.threshold for w in warnings}
        self.assertEqual(thresholds, {0.5, 0.8})

    def test_warn_at_list_is_sorted(self):
        # Thresholds supplied out of order should still fire low-to-high.
        warnings = []
        guard = CostGuard(
            limit_usd=1.0, warn_at=[0.8, 0.5], on_warn=warnings.append
        )
        guard.add(0.9)  # one add crosses both thresholds at once
        fired = [w.threshold for w in warnings]
        self.assertEqual(fired, [0.5, 0.8])

    def test_warn_pct_used_correct(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
        guard.add(0.7)
        self.assertAlmostEqual(warnings[0].pct_used, 0.7)

    def test_warn_label_passed(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
        guard.add(0.7, label="expensive_call")
        self.assertEqual(warnings[0].label, "expensive_call")

    def test_no_warn_when_no_callback(self):
        guard = CostGuard(limit_usd=1.0, warn_at=0.5)
        guard.add(0.7)  # should not raise

    def test_no_warn_below_threshold(self):
        warnings = []
        guard = CostGuard(limit_usd=1.0, warn_at=0.8, on_warn=warnings.append)
        guard.add(0.4)
        self.assertEqual(len(warnings), 0)


class TestSummary(unittest.TestCase):
    def test_summary_basic(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.10, label="turn1")
        guard.add(0.20, label="turn2")
        s = guard.summary()
        self.assertIsInstance(s, CostSummary)
        self.assertAlmostEqual(s.total_usd, 0.30)
        self.assertEqual(s.entry_count, 2)
        self.assertEqual(s.limit_usd, 1.0)

    def test_summary_by_label(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.05, label="search")
        guard.add(0.10, label="search")
        guard.add(0.20, label="llm")
        s = guard.summary()
        self.assertAlmostEqual(s.by_label["search"], 0.15)
        self.assertAlmostEqual(s.by_label["llm"], 0.20)

    def test_summary_unlabelled_excluded_from_by_label(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.05)  # no label
        guard.add(0.10, label="llm")
        s = guard.summary()
        self.assertEqual(set(s.by_label), {"llm"})
        self.assertAlmostEqual(s.total_usd, 0.15)

    def test_summary_ok_true(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.5)
        s = guard.summary()
        self.assertTrue(s.ok)

    def test_summary_ok_false(self):
        guard = CostGuard(limit_usd=0.5, stop_on_limit=False)
        guard.add(0.6)
        s = guard.summary()
        self.assertFalse(s.ok)

    def test_summary_str(self):
        guard = CostGuard(limit_usd=1.0)
        guard.add(0.10, label="search")
        text = str(guard.summary())
        self.assertIn("$", text)
        self.assertIn("search", text)


class TestFactory(unittest.TestCase):
    def test_make_cost_guard(self):
        guard = make_cost_guard(limit_usd=2.0)
        self.assertIsInstance(guard, CostGuard)
        self.assertEqual(guard.limit_usd, 2.0)

    def test_make_cost_guard_default_warn_at(self):
        warnings = []
        guard = make_cost_guard(limit_usd=1.0, on_warn=warnings.append)
        guard.add(0.6)  # crosses default 0.5
        self.assertEqual(len(warnings), 1)

    def test_make_cost_guard_explicit_warn_at_overrides_default(self):
        warnings = []
        guard = make_cost_guard(
            limit_usd=1.0, warn_at=0.9, on_warn=warnings.append
        )
        guard.add(0.6)  # below 0.9, default 0.5 should NOT apply
        self.assertEqual(len(warnings), 0)


class TestRepr(unittest.TestCase):
    def test_repr(self):
        guard = CostGuard(limit_usd=1.0)
        r = repr(guard)
        self.assertIn("CostGuard", r)
        self.assertIn("ok=True", r)


if __name__ == "__main__":
    unittest.main()
