"""Telemetry storage and the controlled, budgeted query boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import TypeVar

from .budget import QueryBudget, QueryCosts, QueryHistoryEntry
from .models import LogEvent, MetricEvent, SpanEvent, as_utc

EventT = TypeVar("EventT", MetricEvent, LogEvent, SpanEvent)


@dataclass(frozen=True, slots=True)
class ServiceSummary:
    service: str
    metric_means: dict[str, float]
    log_count: int
    error_log_count: int
    span_count: int
    failed_span_count: int


class TelemetryStore:
    """Complete canonical telemetry for one incident, without ground truth."""

    def __init__(
        self,
        metrics: tuple[MetricEvent, ...] | list[MetricEvent],
        logs: tuple[LogEvent, ...] | list[LogEvent],
        spans: tuple[SpanEvent, ...] | list[SpanEvent],
    ) -> None:
        self._metrics = tuple(sorted(metrics, key=lambda event: event.event_timestamp))
        self._logs = tuple(sorted(logs, key=lambda event: event.event_timestamp))
        self._spans = tuple(sorted(spans, key=lambda event: event.start_timestamp))

    @property
    def metric_count(self) -> int:
        return len(self._metrics)

    @property
    def log_count(self) -> int:
        return len(self._logs)

    @property
    def span_count(self) -> int:
        return len(self._spans)

    @property
    def services(self) -> frozenset[str]:
        return frozenset(
            event.service for event in (*self._metrics, *self._logs, *self._spans)
        )

    def metrics_between(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[MetricEvent, ...]:
        return self._filter(self._metrics, service, start_time, end_time, "event_timestamp")

    def logs_between(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[LogEvent, ...]:
        return self._filter(self._logs, service, start_time, end_time, "event_timestamp")

    def spans_between(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[SpanEvent, ...]:
        return self._filter(self._spans, service, start_time, end_time, "start_timestamp")

    @staticmethod
    def _filter(
        events: tuple[EventT, ...],
        service: str,
        start_time: datetime,
        end_time: datetime,
        timestamp_field: str,
    ) -> tuple[EventT, ...]:
        start, end = as_utc(start_time), as_utc(end_time)
        if end < start:
            raise ValueError("end_time must be at or after start_time")
        return tuple(
            event
            for event in events
            if event.service == service and start <= getattr(event, timestamp_field) <= end
        )


class TelemetryQueryAPI:
    """Only incident telemetry interface intended for future diagnosis code."""

    __slots__ = ("__store", "_budget", "_cache", "_history")

    def __init__(
        self,
        store: TelemetryStore,
        total_budget: int,
        costs: QueryCosts | None = None,
    ) -> None:
        self.__store = store
        self._budget = QueryBudget(total_budget, costs)
        self._cache: dict[tuple[str, str, datetime, datetime], object] = {}
        self._history: list[QueryHistoryEntry] = []

    @property
    def total_budget(self) -> int:
        return self._budget.total

    @property
    def spent_budget(self) -> int:
        return self._budget.spent

    @property
    def remaining_budget(self) -> int:
        return self._budget.remaining

    @property
    def query_history(self) -> tuple[QueryHistoryEntry, ...]:
        return tuple(self._history)

    def query_metrics(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[MetricEvent, ...]:
        return self._query("metrics", service, start_time, end_time)

    def query_logs(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[LogEvent, ...]:
        return self._query("logs", service, start_time, end_time)

    def query_traces(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> tuple[SpanEvent, ...]:
        return self._query("traces", service, start_time, end_time)

    def query_service_summary(
        self, service: str, start_time: datetime, end_time: datetime
    ) -> ServiceSummary:
        return self._query("service_summary", service, start_time, end_time)

    def _query(self, query_type: str, service: str, start_time: datetime, end_time: datetime):
        start, end = as_utc(start_time), as_utc(end_time)
        if end < start:
            raise ValueError("end_time must be at or after start_time")
        cache_key = (query_type, service, start, end)
        if cache_key in self._cache:
            self._record(query_type, service, start, end, 0, True)
            return self._cache[cache_key]

        cost = self._budget.charge(query_type)
        if query_type == "metrics":
            result = self.__store.metrics_between(service, start, end)
        elif query_type == "logs":
            result = self.__store.logs_between(service, start, end)
        elif query_type == "traces":
            result = self.__store.spans_between(service, start, end)
        elif query_type == "service_summary":
            result = self._summarize(service, start, end)
        else:
            raise ValueError(f"unknown query type: {query_type}")
        self._cache[cache_key] = result
        self._record(query_type, service, start, end, cost, False)
        return result

    def _summarize(self, service: str, start: datetime, end: datetime) -> ServiceSummary:
        metrics = self.__store.metrics_between(service, start, end)
        logs = self.__store.logs_between(service, start, end)
        spans = self.__store.spans_between(service, start, end)
        by_name: dict[str, list[float]] = {}
        for metric in metrics:
            by_name.setdefault(metric.metric_name, []).append(metric.value)
        return ServiceSummary(
            service=service,
            metric_means={name: fmean(values) for name, values in sorted(by_name.items())},
            log_count=len(logs),
            error_log_count=sum(log.severity in {"ERROR", "CRITICAL"} for log in logs),
            span_count=len(spans),
            failed_span_count=sum(span.status == "ERROR" for span in spans),
        )

    def _record(
        self,
        query_type: str,
        service: str,
        start: datetime,
        end: datetime,
        cost: int,
        cached: bool,
    ) -> None:
        self._history.append(
            QueryHistoryEntry(query_type, service, start, end, cost, cached)
        )
