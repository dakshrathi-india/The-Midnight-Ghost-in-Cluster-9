"""Global incident-time telemetry query budget."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class BudgetExceededError(RuntimeError):
    """Raised when a telemetry query would exceed the remaining budget."""


@dataclass(frozen=True, slots=True)
class QueryCosts:
    metrics: int = 1
    logs: int = 1
    traces: int = 1
    service_summary: int = 1

    def __post_init__(self) -> None:
        if any(cost < 0 for cost in self.as_dict().values()):
            raise ValueError("query costs cannot be negative")

    def as_dict(self) -> dict[str, int]:
        return {
            "metrics": self.metrics,
            "logs": self.logs,
            "traces": self.traces,
            "service_summary": self.service_summary,
        }

    def for_query(self, query_type: str) -> int:
        try:
            return self.as_dict()[query_type]
        except KeyError as error:
            raise ValueError(f"unknown query type: {query_type}") from error


@dataclass(frozen=True, slots=True)
class QueryHistoryEntry:
    query_type: str
    service: str
    start_time: datetime
    end_time: datetime
    cost: int
    served_from_cache: bool


class QueryBudget:
    """Tracks one shared budget across all incident-time query types."""

    def __init__(self, total: int, costs: QueryCosts | None = None) -> None:
        if total < 0:
            raise ValueError("total budget cannot be negative")
        self._total = total
        self._spent = 0
        self._costs = costs or QueryCosts()

    @property
    def total(self) -> int:
        return self._total

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return self._total - self._spent

    def cost_for(self, query_type: str) -> int:
        return self._costs.for_query(query_type)

    def charge(self, query_type: str) -> int:
        cost = self.cost_for(query_type)
        if cost > self.remaining:
            raise BudgetExceededError(
                f"{query_type} query costs {cost}, but only {self.remaining} remains"
            )
        self._spent += cost
        return cost
