"""Canonical, diagnosis-facing telemetry models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def as_utc(timestamp: datetime) -> datetime:
    """Return an aware UTC timestamp, treating naive values as UTC."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class MetricEvent:
    event_timestamp: datetime
    arrival_timestamp: datetime
    service: str
    metric_name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_timestamp", as_utc(self.event_timestamp))
        object.__setattr__(self, "arrival_timestamp", as_utc(self.arrival_timestamp))
        if not self.service.strip() or not self.metric_name.strip():
            raise ValueError("service and metric_name must be non-empty")


@dataclass(frozen=True, slots=True)
class LogEvent:
    event_timestamp: datetime
    arrival_timestamp: datetime
    service: str
    severity: str
    message: str
    trace_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_timestamp", as_utc(self.event_timestamp))
        object.__setattr__(self, "arrival_timestamp", as_utc(self.arrival_timestamp))
        if not self.service.strip() or not self.message.strip():
            raise ValueError("service and message must be non-empty")


@dataclass(frozen=True, slots=True)
class SpanEvent:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_timestamp: datetime
    arrival_timestamp: datetime
    duration_ms: float
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_timestamp", as_utc(self.start_timestamp))
        object.__setattr__(self, "arrival_timestamp", as_utc(self.arrival_timestamp))
        if not self.trace_id or not self.span_id or not self.service or not self.operation:
            raise ValueError("trace, span, service, and operation identifiers are required")
        if self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
