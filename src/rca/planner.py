"""Deterministic pair-separation telemetry query planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Collection, Sequence

from src.core.telemetry import TelemetryQueryAPI

from .diagnosis import Hypothesis
from .signatures import FailureSignatureLibrary


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    query_type: str
    service: str
    start_time: datetime
    end_time: datetime
    pair_separation: int
    cost: int
    utility: float

    @property
    def key(self) -> tuple[str, str, datetime, datetime]:
        return (self.query_type, self.service, self.start_time, self.end_time)


class ActiveQueryPlanner:
    _query_order = {"logs": 0, "traces": 1, "metrics": 2}

    def __init__(self, signatures: FailureSignatureLibrary | None = None) -> None:
        self.signatures = signatures or FailureSignatureLibrary()

    def choose_next(
        self,
        hypotheses: Sequence[Hypothesis],
        services: Collection[str],
        start_time: datetime,
        end_time: datetime,
        api: TelemetryQueryAPI,
        executed_queries: Collection[tuple[str, str, datetime, datetime]],
    ) -> PlannedQuery | None:
        if len(hypotheses) < 2:
            return None
        candidates: list[PlannedQuery] = []
        for service in sorted(services):
            for query_type in ("logs", "traces", "metrics"):
                key = (query_type, service, start_time, end_time)
                if key in executed_queries or not api.can_afford(
                    query_type, service, start_time, end_time
                ):
                    continue
                separation = self._pair_separation(hypotheses, service, query_type)
                if separation == 0:
                    continue
                cost = api.query_cost(query_type, service, start_time, end_time)
                utility = float("inf") if cost == 0 else separation / cost
                candidates.append(
                    PlannedQuery(
                        query_type,
                        service,
                        start_time,
                        end_time,
                        separation,
                        cost,
                        utility,
                    )
                )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda query: (
                -query.utility,
                -query.pair_separation,
                query.cost,
                self._query_order[query.query_type],
                query.service,
            ),
        )

    def has_unexecuted_discriminating_query(
        self,
        hypotheses: Sequence[Hypothesis],
        services: Collection[str],
        start_time: datetime,
        end_time: datetime,
        executed_queries: Collection[tuple[str, str, datetime, datetime]],
    ) -> bool:
        return any(
            (query_type, service, start_time, end_time) not in executed_queries
            and self._pair_separation(hypotheses, service, query_type) > 0
            for service in sorted(services)
            for query_type in ("logs", "traces", "metrics")
        )

    def _pair_separation(
        self,
        hypotheses: Sequence[Hypothesis],
        service: str,
        query_type: str,
    ) -> int:
        return sum(
            self._expected_evidence(left, service, query_type)
            != self._expected_evidence(right, service, query_type)
            for left, right in combinations(hypotheses, 2)
        )

    def _expected_evidence(
        self, hypothesis: Hypothesis, service: str, query_type: str
    ) -> tuple[object, ...]:
        if hypothesis.service != service:
            return ("not-root",)
        signature = self.signatures.get(hypothesis.failure_mode)
        if query_type == "metrics":
            return (
                "root",
                *sorted(
                    (expectation.metric_name, expectation.direction.value)
                    for expectation in signature.metric_expectations
                ),
            )
        if query_type == "logs":
            return ("root", *sorted(signature.log_categories))
        if query_type == "traces":
            return (
                "root",
                *sorted(expectation.value for expectation in signature.trace_expectations),
            )
        raise ValueError(f"unsupported query type: {query_type}")
