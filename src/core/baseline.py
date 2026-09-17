"""Historical context known before an incident."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median, pstdev
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
    median: float = math.nan
    median_absolute_deviation: float = math.nan

    def __post_init__(self) -> None:
        if math.isnan(self.median):
            object.__setattr__(self, "median", self.mean)
        if math.isnan(self.median_absolute_deviation):
            object.__setattr__(
                self,
                "median_absolute_deviation",
                self.standard_deviation * 0.67448975,
            )


class BaselineStore:
    def __init__(
        self,
        known_services: set[str] | frozenset[str],
        metric_summaries: Mapping[tuple[str, str], MetricSummary],
        typical_service_latencies_ms: Mapping[str, float],
        cached_dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
        metric_history: tuple[MetricEvent, ...] | list[MetricEvent] = (),
    ) -> None:
        self.known_services = frozenset(known_services)
        self.metric_summaries = MappingProxyType(dict(metric_summaries))
        self.typical_service_latencies_ms = MappingProxyType(
            dict(typical_service_latencies_ms)
        )
        self.cached_dependency_edges = frozenset(cached_dependency_edges or ())
        self.metric_history = tuple(
            sorted(metric_history, key=lambda event: (event.event_timestamp, event.service, event.metric_name))
        )

    @classmethod
    def from_metrics(
        cls,
        known_services: set[str],
        metrics: list[MetricEvent] | tuple[MetricEvent, ...],
        cached_dependency_edges: set[tuple[str, str]] | None = None,
    ) -> "BaselineStore":
        grouped: dict[tuple[str, str], list[float]] = {}
        for event in metrics:
            grouped.setdefault((event.service, event.metric_name), []).append(event.value)
        summaries: dict[tuple[str, str], MetricSummary] = {}
        for key, values in grouped.items():
            baseline_median = median(values)
            summaries[key] = MetricSummary(
                count=len(values),
                mean=fmean(values),
                standard_deviation=pstdev(values),
                minimum=min(values),
                maximum=max(values),
                median=baseline_median,
                median_absolute_deviation=median(
                    abs(value - baseline_median) for value in values
                ),
            )
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
            metrics,
        )
