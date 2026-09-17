"""Historical context known before an incident."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Mapping

from .models import MetricEvent


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


class BaselineStore:
    def __init__(
        self,
        known_services: set[str] | frozenset[str],
        metric_summaries: Mapping[tuple[str, str], MetricSummary],
        typical_service_latencies_ms: Mapping[str, float],
        cached_dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
        historical_context_counts_toward_budget: bool = False,
    ) -> None:
        self.known_services = frozenset(known_services)
        self.metric_summaries = MappingProxyType(dict(metric_summaries))
        self.typical_service_latencies_ms = MappingProxyType(
            dict(typical_service_latencies_ms)
        )
        self.cached_dependency_edges = frozenset(cached_dependency_edges or ())
        self.historical_context_counts_toward_budget = historical_context_counts_toward_budget

    @classmethod
    def from_metrics(
        cls,
        known_services: set[str],
        metrics: list[MetricEvent] | tuple[MetricEvent, ...],
        cached_dependency_edges: set[tuple[str, str]] | None = None,
        historical_context_counts_toward_budget: bool = False,
    ) -> "BaselineStore":
        grouped: dict[tuple[str, str], list[float]] = {}
        for event in metrics:
            grouped.setdefault((event.service, event.metric_name), []).append(event.value)
        summaries = {
            key: MetricSummary(
                count=len(values),
                mean=fmean(values),
                standard_deviation=pstdev(values),
                minimum=min(values),
                maximum=max(values),
            )
            for key, values in grouped.items()
        }
        latencies = {
            service: summaries[(service, "request_latency_ms")].mean
            for service in known_services
            if (service, "request_latency_ms") in summaries
        }
        return cls(
            known_services,
            summaries,
            latencies,
            cached_dependency_edges,
            historical_context_counts_toward_budget,
        )
