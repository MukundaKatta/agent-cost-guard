"""Track cumulative LLM cost and enforce USD limits."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


class CostLimitExceeded(Exception):
    """Raised when the cumulative cost exceeds the configured limit."""

    def __init__(self, total_usd: float, limit_usd: float, label: str = "") -> None:
        self.total_usd = total_usd
        self.limit_usd = limit_usd
        self.label = label
        msg = f"cost limit exceeded: ${total_usd:.4f} > ${limit_usd:.4f}"
        if label:
            msg += f" (after {label!r})"
        super().__init__(msg)


@dataclass
class CostEntry:
    """A single recorded cost event.

    Attributes:
        cost_usd: the cost in US dollars.
        label: optional human-readable label (tool name, turn id, etc.).
        ts: Unix timestamp of the recording.
    """

    cost_usd: float
    label: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class CostWarning:
    """Fired when cumulative cost crosses a warn_at threshold.

    Attributes:
        total_usd: cumulative cost at the time of the warning.
        limit_usd: the configured limit.
        pct_used: fraction of limit used (0.0 – 1.0+).
        threshold: the warn_at fraction that triggered this warning.
        label: label of the cost entry that triggered it.
    """

    total_usd: float
    limit_usd: float
    pct_used: float
    threshold: float
    label: str = ""


@dataclass
class CostSummary:
    """Aggregate cost report.

    Attributes:
        total_usd: cumulative cost across all entries.
        entry_count: number of recorded cost events.
        limit_usd: the configured limit.
        pct_used: fraction of limit used.
        remaining_usd: budget remaining (can be negative if over limit).
        by_label: dict of {label: total_cost_usd} for labelled entries.
    """

    total_usd: float
    entry_count: int
    limit_usd: float
    pct_used: float
    remaining_usd: float
    by_label: dict[str, float]

    @property
    def ok(self) -> bool:
        return self.total_usd <= self.limit_usd

    def __str__(self) -> str:
        lines = [
            f"Cost: ${self.total_usd:.4f} / ${self.limit_usd:.4f} "
            f"({self.pct_used:.1%} used)",
            f"Entries: {self.entry_count}",
        ]
        if self.by_label:
            lines.append("Breakdown:")
            for lbl, cost in sorted(self.by_label.items(), key=lambda x: -x[1]):
                lines.append(f"  {lbl}: ${cost:.4f}")
        return "\n".join(lines)


class CostGuard:
    """Track cumulative LLM cost and enforce a USD limit.

    Args:
        limit_usd: the maximum allowed cumulative cost in US dollars.
        warn_at: fraction(s) of the limit at which to fire the on_warn callback.
            Can be a single float or a list of floats (e.g. [0.5, 0.8]).
        on_warn: callback called when a warn_at threshold is crossed.
            Receives a CostWarning.
        stop_on_limit: if True (default), raise CostLimitExceeded when the
            limit is reached. If False, continue tracking without raising.
        label: optional label for this guard instance.

    Example::

        guard = CostGuard(limit_usd=1.00, warn_at=0.8)
        guard.add(0.05, label="research_turn_1")
        guard.add(0.12, label="web_search")
        print(guard.total_usd)   # 0.17
        print(guard.remaining_usd)  # 0.83
    """

    def __init__(
        self,
        limit_usd: float,
        *,
        warn_at: float | list[float] | None = None,
        on_warn: Callable[[CostWarning], None] | None = None,
        stop_on_limit: bool = True,
        label: str = "",
    ) -> None:
        if limit_usd <= 0:
            raise ValueError(f"limit_usd must be > 0, got {limit_usd}")
        self.limit_usd = limit_usd
        self.stop_on_limit = stop_on_limit
        self.label = label
        self.on_warn = on_warn

        if warn_at is None:
            self._warn_thresholds: list[float] = []
        elif isinstance(warn_at, (int, float)):
            self._warn_thresholds = [float(warn_at)]
        else:
            self._warn_thresholds = sorted(float(w) for w in warn_at)

        self._entries: list[CostEntry] = []
        self._total: float = 0.0
        self._fired_thresholds: set[float] = set()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, cost_usd: float, *, label: str = "") -> "CostGuard":
        """Record a cost entry and check limits.

        Args:
            cost_usd: cost in US dollars (must be >= 0).
            label: optional label for this entry.

        Returns:
            self, for chaining.

        Raises:
            ValueError: if cost_usd < 0.
            CostLimitExceeded: if the new total exceeds limit_usd and
                stop_on_limit is True.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")
        entry = CostEntry(cost_usd=cost_usd, label=label)
        self._entries.append(entry)
        self._total += cost_usd
        self._check_warnings(label)
        if self.stop_on_limit and self._total > self.limit_usd:
            raise CostLimitExceeded(self._total, self.limit_usd, label)
        return self

    def check(self) -> None:
        """Manually check if the limit is exceeded; raise if so."""
        if self._total > self.limit_usd:
            raise CostLimitExceeded(self._total, self.limit_usd)

    def reset(self) -> None:
        """Reset all tracked cost and warning state."""
        self._entries.clear()
        self._total = 0.0
        self._fired_thresholds.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_usd(self) -> float:
        """Cumulative cost in USD."""
        return round(self._total, 10)

    @property
    def remaining_usd(self) -> float:
        """Remaining budget in USD (can be negative)."""
        return round(self.limit_usd - self._total, 10)

    @property
    def pct_used(self) -> float:
        """Fraction of limit used (0.0 to 1.0+)."""
        return self._total / self.limit_usd

    @property
    def ok(self) -> bool:
        """True if cumulative cost has not exceeded the limit."""
        return self._total <= self.limit_usd

    @property
    def entries(self) -> list[CostEntry]:
        """All recorded cost entries (read-only copy)."""
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> CostSummary:
        """Return an aggregate cost summary."""
        by_label: dict[str, float] = {}
        for e in self._entries:
            if e.label:
                by_label[e.label] = round(by_label.get(e.label, 0.0) + e.cost_usd, 10)
        return CostSummary(
            total_usd=self.total_usd,
            entry_count=len(self._entries),
            limit_usd=self.limit_usd,
            pct_used=self.pct_used,
            remaining_usd=self.remaining_usd,
            by_label=by_label,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_warnings(self, label: str) -> None:
        if not self.on_warn:
            return
        pct = self.pct_used
        for threshold in self._warn_thresholds:
            if threshold not in self._fired_thresholds and pct >= threshold:
                self._fired_thresholds.add(threshold)
                self.on_warn(
                    CostWarning(
                        total_usd=self.total_usd,
                        limit_usd=self.limit_usd,
                        pct_used=pct,
                        threshold=threshold,
                        label=label,
                    )
                )

    def __repr__(self) -> str:
        return (
            f"CostGuard(total=${self.total_usd:.4f}, "
            f"limit=${self.limit_usd:.4f}, ok={self.ok})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_cost_guard(
    limit_usd: float,
    *,
    warn_at: float | list[float] | None = None,
    on_warn: Callable[[CostWarning], None] | None = None,
    stop_on_limit: bool = True,
    label: str = "",
) -> CostGuard:
    """Create a CostGuard with common defaults.

    Equivalent to ``CostGuard(limit_usd, warn_at=[0.5, 0.8], ...)`` when
    warn_at is not provided.
    """
    if warn_at is None:
        warn_at = [0.5, 0.8]
    return CostGuard(
        limit_usd,
        warn_at=warn_at,
        on_warn=on_warn,
        stop_on_limit=stop_on_limit,
        label=label,
    )
