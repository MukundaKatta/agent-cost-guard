"""Tests for agent-cost-guard."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from agent_cost_guard import (
    CostEntry,
    CostGuard,
    CostLimitExceeded,
    CostSummary,
    CostWarning,
    make_cost_guard,
)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_basic_init():
    guard = CostGuard(limit_usd=1.00)
    assert guard.limit_usd == 1.00
    assert guard.total_usd == 0.0
    assert guard.ok is True

def test_invalid_limit():
    with pytest.raises(ValueError):
        CostGuard(limit_usd=0)

def test_invalid_limit_negative():
    with pytest.raises(ValueError):
        CostGuard(limit_usd=-1.0)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

def test_add_single():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.05)
    assert guard.total_usd == pytest.approx(0.05)

def test_add_multiple():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.05)
    guard.add(0.10)
    assert guard.total_usd == pytest.approx(0.15)

def test_add_returns_self():
    guard = CostGuard(limit_usd=1.0)
    result = guard.add(0.01)
    assert result is guard

def test_add_chaining():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.01).add(0.02).add(0.03)
    assert guard.total_usd == pytest.approx(0.06)

def test_add_zero():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.0)
    assert guard.total_usd == 0.0

def test_add_negative_raises():
    guard = CostGuard(limit_usd=1.0)
    with pytest.raises(ValueError):
        guard.add(-0.01)


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------

def test_limit_exceeded_raises():
    guard = CostGuard(limit_usd=0.10, stop_on_limit=True)
    guard.add(0.05)
    with pytest.raises(CostLimitExceeded):
        guard.add(0.06)

def test_limit_exceeded_message():
    guard = CostGuard(limit_usd=0.10)
    try:
        guard.add(0.20)
    except CostLimitExceeded as e:
        assert e.total_usd == pytest.approx(0.20)
        assert e.limit_usd == pytest.approx(0.10)

def test_stop_on_limit_false():
    guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
    guard.add(0.20)  # no exception
    assert guard.total_usd == pytest.approx(0.20)
    assert guard.ok is False

def test_exactly_at_limit_ok():
    guard = CostGuard(limit_usd=0.10)
    guard.add(0.10)
    assert guard.ok is True

def test_check_raises_when_over():
    guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
    guard.add(0.20)
    with pytest.raises(CostLimitExceeded):
        guard.check()

def test_check_ok_when_under():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.5)
    guard.check()  # no exception


# ---------------------------------------------------------------------------
# remaining_usd / pct_used
# ---------------------------------------------------------------------------

def test_remaining_usd():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.25)
    assert guard.remaining_usd == pytest.approx(0.75)

def test_remaining_negative_when_over():
    guard = CostGuard(limit_usd=0.10, stop_on_limit=False)
    guard.add(0.20)
    assert guard.remaining_usd == pytest.approx(-0.10)

def test_pct_used_zero():
    guard = CostGuard(limit_usd=1.0)
    assert guard.pct_used == 0.0

def test_pct_used_half():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.5)
    assert guard.pct_used == pytest.approx(0.5)

def test_pct_used_full():
    guard = CostGuard(limit_usd=1.0)
    guard.add(1.0)
    assert guard.pct_used == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# entries / entry_count
# ---------------------------------------------------------------------------

def test_entries_empty():
    guard = CostGuard(limit_usd=1.0)
    assert guard.entries == []

def test_entries_stored():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.05, label="turn1")
    guard.add(0.10, label="turn2")
    assert guard.entry_count == 2
    assert guard.entries[0].label == "turn1"
    assert guard.entries[1].label == "turn2"

def test_entries_returns_copy():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.01)
    entries = guard.entries
    entries.clear()
    assert guard.entry_count == 1

def test_entry_cost_stored():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.07, label="search")
    assert guard.entries[0].cost_usd == pytest.approx(0.07)
    assert guard.entries[0].label == "search"

def test_entry_timestamp_set():
    import time
    before = time.time()
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.01)
    after = time.time()
    ts = guard.entries[0].ts
    assert before <= ts <= after


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_total():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.5)
    guard.reset()
    assert guard.total_usd == 0.0

def test_reset_clears_entries():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.5)
    guard.reset()
    assert guard.entry_count == 0

def test_reset_clears_fired_thresholds():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
    guard.add(0.6)  # fires warning
    guard.reset()
    guard.add(0.6)  # should fire again after reset
    assert len(warnings) == 2


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_warn_at_single_fires():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
    guard.add(0.6)
    assert len(warnings) == 1
    assert warnings[0].threshold == 0.5

def test_warn_at_only_fires_once():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
    guard.add(0.4)
    guard.add(0.2)  # crosses 0.5 here
    guard.add(0.1)
    assert len(warnings) == 1

def test_warn_at_multiple():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=[0.5, 0.8], on_warn=warnings.append)
    guard.add(0.6)  # crosses 0.5
    guard.add(0.3)  # crosses 0.8
    assert len(warnings) == 2
    thresholds = {w.threshold for w in warnings}
    assert thresholds == {0.5, 0.8}

def test_warn_pct_used_correct():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
    guard.add(0.7)
    assert warnings[0].pct_used == pytest.approx(0.7)

def test_warn_label_passed():
    warnings: list[CostWarning] = []
    guard = CostGuard(limit_usd=1.0, warn_at=0.5, on_warn=warnings.append)
    guard.add(0.7, label="expensive_call")
    assert warnings[0].label == "expensive_call"

def test_no_warn_when_no_callback():
    guard = CostGuard(limit_usd=1.0, warn_at=0.5)
    guard.add(0.7)  # should not raise


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_basic():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.10, label="turn1")
    guard.add(0.20, label="turn2")
    s = guard.summary()
    assert isinstance(s, CostSummary)
    assert s.total_usd == pytest.approx(0.30)
    assert s.entry_count == 2
    assert s.limit_usd == 1.0

def test_summary_by_label():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.05, label="search")
    guard.add(0.10, label="search")
    guard.add(0.20, label="llm")
    s = guard.summary()
    assert s.by_label["search"] == pytest.approx(0.15)
    assert s.by_label["llm"] == pytest.approx(0.20)

def test_summary_ok_true():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.5)
    s = guard.summary()
    assert s.ok is True

def test_summary_ok_false():
    guard = CostGuard(limit_usd=0.5, stop_on_limit=False)
    guard.add(0.6)
    s = guard.summary()
    assert s.ok is False

def test_summary_str():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.10, label="search")
    text = str(guard.summary())
    assert "$" in text
    assert "search" in text


# ---------------------------------------------------------------------------
# make_cost_guard factory
# ---------------------------------------------------------------------------

def test_make_cost_guard():
    guard = make_cost_guard(limit_usd=2.0)
    assert guard.limit_usd == 2.0

def test_make_cost_guard_default_warn_at():
    warnings: list[CostWarning] = []
    guard = make_cost_guard(limit_usd=1.0, on_warn=warnings.append)
    guard.add(0.6)  # crosses 0.5
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    guard = CostGuard(limit_usd=1.0)
    r = repr(guard)
    assert "CostGuard" in r
    assert "ok=True" in r
