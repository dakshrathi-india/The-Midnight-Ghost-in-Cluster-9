"""Dependency and service evidence reconstruction from canonical spans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Mapping, Sequence

from src.core.baseline import BaselineStore
from src.core.models import SpanEvent


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    caller_service: str
    callee_service: str
    support_count: int


@dataclass(frozen=True, slots=True)
class TraceServiceEvidence:
    service: str
    span_count: int
    failed_span_count: int
    failed_span_fraction: float
    mean_duration_ms: float
    maximum_duration_ms: float
    typical_latency_ms: float | None
    latency_ratio: float | None
    corroborates_abnormality: bool


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    services: frozenset[str]
    edges: tuple[DependencyEdge, ...]
    service_evidence: Mapping[str, TraceServiceEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_evidence", MappingProxyType(dict(self.service_evidence)))


@dataclass(frozen=True, slots=True)
class TraceGraphConfig:
    failed_span_fraction_threshold: float = 0.1
    latency_ratio_threshold: float = 2.0


class TraceGraphAnalyzer:
    def __init__(self, config: TraceGraphConfig | None = None) -> None:
        self.config = config or TraceGraphConfig()

    def reconstruct(
        self,
        spans: Sequence[SpanEvent],
        baseline: BaselineStore | None = None,
    ) -> DependencyGraph:
        ordered_spans = sorted(
            spans,
            key=lambda span: (span.trace_id, span.start_timestamp, span.span_id),
        )
        services = frozenset(span.service for span in ordered_spans)
        spans_by_trace: dict[str, dict[str, SpanEvent]] = {}
        for span in ordered_spans:
            spans_by_trace.setdefault(span.trace_id, {})[span.span_id] = span

        counts: Counter[tuple[str, str]] = Counter()
        for trace_spans in spans_by_trace.values():
            for child in trace_spans.values():
                if child.parent_span_id is None:
                    continue
                parent = trace_spans.get(child.parent_span_id)
                if parent is None or parent.service == child.service:
                    continue
                counts[(parent.service, child.service)] += 1

        edges = tuple(
            DependencyEdge(caller, callee, support)
            for (caller, callee), support in sorted(counts.items())
        )
        evidence = self._service_evidence(ordered_spans, baseline)
        return DependencyGraph(services, edges, evidence)

    def _service_evidence(
        self,
        spans: Sequence[SpanEvent],
        baseline: BaselineStore | None,
    ) -> dict[str, TraceServiceEvidence]:
        grouped: dict[str, list[SpanEvent]] = {}
        for span in spans:
            grouped.setdefault(span.service, []).append(span)

        result: dict[str, TraceServiceEvidence] = {}
        for service in sorted(grouped):
            service_spans = grouped[service]
            failed_count = sum(span.status == "ERROR" for span in service_spans)
            failed_fraction = failed_count / len(service_spans)
            durations = [span.duration_ms for span in service_spans]
            typical = (
                baseline.typical_service_latencies_ms.get(service) if baseline else None
            )
            mean_duration = fmean(durations)
            latency_ratio = (
                mean_duration / typical if typical is not None and typical > 0 else None
            )
            corroborates = (
                failed_fraction >= self.config.failed_span_fraction_threshold
                or (
                    latency_ratio is not None
                    and latency_ratio >= self.config.latency_ratio_threshold
                )
            )
            result[service] = TraceServiceEvidence(
                service=service,
                span_count=len(service_spans),
                failed_span_count=failed_count,
                failed_span_fraction=failed_fraction,
                mean_duration_ms=mean_duration,
                maximum_duration_ms=max(durations),
                typical_latency_ms=typical,
                latency_ratio=latency_ratio,
                corroborates_abnormality=corroborates,
            )
        return result
