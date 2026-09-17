"""Reusable normalization for canonical telemetry from any backend."""

from __future__ import annotations

import re
from dataclasses import replace

from .models import LogEvent, MetricEvent, SpanEvent, as_utc


class TelemetryNormalizer:
    _severity_map = {
        "WARN": "WARNING",
        "ERR": "ERROR",
        "FATAL": "CRITICAL",
        "TRACE": "DEBUG",
    }
    _unit_conversions = {
        "s": ("ms", 1000.0),
        "seconds": ("ms", 1000.0),
        "us": ("ms", 0.001),
        "percent": ("ratio", 0.01),
        "%": ("ratio", 0.01),
    }

    @staticmethod
    def normalize_service_name(name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not normalized:
            raise ValueError("service name cannot normalize to an empty value")
        return normalized

    def normalize_metrics(self, events: list[MetricEvent] | tuple[MetricEvent, ...]) -> tuple[MetricEvent, ...]:
        normalized: list[MetricEvent] = []
        for event in events:
            target_unit, factor = self._unit_conversions.get(
                event.unit.strip().lower(), (event.unit.strip().lower(), 1.0)
            )
            normalized.append(
                replace(
                    event,
                    event_timestamp=as_utc(event.event_timestamp),
                    arrival_timestamp=as_utc(event.arrival_timestamp),
                    service=self.normalize_service_name(event.service),
                    metric_name=event.metric_name.strip().lower().replace(" ", "_"),
                    value=event.value * factor,
                    unit=target_unit,
                )
            )
        return self._deduplicate(normalized)

    def normalize_logs(self, events: list[LogEvent] | tuple[LogEvent, ...]) -> tuple[LogEvent, ...]:
        normalized = [
            replace(
                event,
                event_timestamp=as_utc(event.event_timestamp),
                arrival_timestamp=as_utc(event.arrival_timestamp),
                service=self.normalize_service_name(event.service),
                severity=self._severity_map.get(
                    event.severity.strip().upper(), event.severity.strip().upper()
                ),
                message=event.message.strip(),
            )
            for event in events
        ]
        return self._deduplicate(normalized)

    def normalize_spans(self, events: list[SpanEvent] | tuple[SpanEvent, ...]) -> tuple[SpanEvent, ...]:
        normalized = [
            replace(
                event,
                start_timestamp=as_utc(event.start_timestamp),
                arrival_timestamp=as_utc(event.arrival_timestamp),
                service=self.normalize_service_name(event.service),
                operation=event.operation.strip(),
                status=event.status.strip().upper(),
            )
            for event in events
        ]
        return self._deduplicate(normalized)

    @staticmethod
    def _deduplicate(events: list[object]) -> tuple:
        return tuple(dict.fromkeys(events))
