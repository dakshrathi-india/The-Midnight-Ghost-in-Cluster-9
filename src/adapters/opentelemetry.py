"""Deterministic OTLP-style JSON/dict conversion to canonical telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

from src.core import LogEvent, MetricEvent, SpanEvent
from src.core.normalization import TelemetryNormalizer


class OpenTelemetryParseError(ValueError):
    """Raised when required OTLP telemetry fields are absent or malformed."""


@dataclass(frozen=True, slots=True)
class OpenTelemetryBatch:
    metrics: tuple[MetricEvent, ...]
    logs: tuple[LogEvent, ...]
    spans: tuple[SpanEvent, ...]


class OpenTelemetryAdapter:
    def __init__(self, normalizer: TelemetryNormalizer | None = None) -> None:
        self._normalizer = normalizer or TelemetryNormalizer()

    def parse_metric(
        self,
        record: Mapping[str, object],
        resource: Mapping[str, object] | None = None,
    ) -> MetricEvent:
        service = _service_name(record, resource)
        name = _required_text(record, "name", "metricName", "metric_name")
        point = _mapping(record.get("dataPoint")) or _mapping(record.get("data_point")) or record
        value = _numeric(point, "asDouble", "asInt", "value")
        timestamp = _timestamp(point, "timeUnixNano", "time_unix_nano", "timestamp")
        arrival = _optional_timestamp(
            point,
            "observedTimeUnixNano",
            "observed_time_unix_nano",
            "arrivalTimeUnixNano",
            "arrival_timestamp",
        ) or _optional_timestamp(
            record,
            "observedTimeUnixNano",
            "receivedTimeUnixNano",
            "arrivalTimeUnixNano",
            "arrival_timestamp",
        ) or timestamp
        unit = str(record.get("unit", ""))
        return self._normalizer.normalize_metrics(
            [MetricEvent(timestamp, arrival, service, name, value, unit)]
        )[0]

    def parse_log(
        self,
        record: Mapping[str, object],
        resource: Mapping[str, object] | None = None,
    ) -> LogEvent:
        service = _service_name(record, resource)
        timestamp = _timestamp(record, "timeUnixNano", "time_unix_nano", "timestamp")
        arrival = _optional_timestamp(
            record,
            "observedTimeUnixNano",
            "observed_time_unix_nano",
            "arrivalTimeUnixNano",
            "arrival_timestamp",
        ) or timestamp
        body = _any_value(record.get("body", record.get("message")))
        if body is None or not str(body).strip():
            raise OpenTelemetryParseError("log requires a non-empty body/message")
        severity = _severity(record)
        trace_id = _optional_text(record, "traceId", "trace_id")
        return self._normalizer.normalize_logs(
            [
                LogEvent(
                    timestamp,
                    arrival,
                    service,
                    severity,
                    str(body),
                    trace_id,
                )
            ]
        )[0]

    def parse_span(
        self,
        record: Mapping[str, object],
        resource: Mapping[str, object] | None = None,
    ) -> SpanEvent:
        service = _service_name(record, resource)
        trace_id = _required_text(record, "traceId", "trace_id")
        span_id = _required_text(record, "spanId", "span_id")
        parent_span_id = _optional_text(record, "parentSpanId", "parent_span_id")
        operation = _required_text(record, "name", "operation")
        start = _timestamp(
            record, "startTimeUnixNano", "start_time_unix_nano", "start_timestamp"
        )
        end = _optional_timestamp(
            record, "endTimeUnixNano", "end_time_unix_nano", "end_timestamp"
        )
        duration_value = record.get("duration_ms", record.get("durationMs"))
        if end is None and duration_value is None:
            raise OpenTelemetryParseError("span requires end timestamp or duration_ms")
        duration_ms = (
            float(duration_value)
            if duration_value is not None
            else (end - start).total_seconds() * 1000.0
        )
        if duration_ms < 0:
            raise OpenTelemetryParseError("span end timestamp precedes start timestamp")
        arrival = _optional_timestamp(
            record,
            "observedTimeUnixNano",
            "arrivalTimeUnixNano",
            "arrival_timestamp",
        ) or end or start + timedelta(milliseconds=duration_ms)
        status = _span_status(record.get("status"))
        return self._normalizer.normalize_spans(
            [
                SpanEvent(
                    trace_id,
                    span_id,
                    parent_span_id,
                    service,
                    operation,
                    start,
                    arrival,
                    duration_ms,
                    status,
                )
            ]
        )[0]

    def parse_export(self, payload: Mapping[str, object]) -> OpenTelemetryBatch:
        metrics = tuple(self._export_metrics(payload))
        logs = tuple(self._export_logs(payload))
        spans = tuple(self._export_spans(payload))
        return OpenTelemetryBatch(metrics, logs, spans)

    def _export_metrics(self, payload: Mapping[str, object]) -> Iterable[MetricEvent]:
        for resource_block in _sequence(payload.get("resourceMetrics")):
            resource_mapping = _mapping(resource_block)
            resource = _mapping(resource_mapping.get("resource"))
            for scope in _sequence(resource_mapping.get("scopeMetrics")):
                for metric in _sequence(_mapping(scope).get("metrics")):
                    metric_mapping = _mapping(metric)
                    points = _metric_points(metric_mapping)
                    for point in points:
                        yield self.parse_metric(
                            {**metric_mapping, "dataPoint": point}, resource
                        )

    def _export_logs(self, payload: Mapping[str, object]) -> Iterable[LogEvent]:
        for resource_block in _sequence(payload.get("resourceLogs")):
            resource_mapping = _mapping(resource_block)
            resource = _mapping(resource_mapping.get("resource"))
            for scope in _sequence(resource_mapping.get("scopeLogs")):
                for record in _sequence(_mapping(scope).get("logRecords")):
                    yield self.parse_log(_mapping(record), resource)

    def _export_spans(self, payload: Mapping[str, object]) -> Iterable[SpanEvent]:
        for resource_block in _sequence(payload.get("resourceSpans")):
            resource_mapping = _mapping(resource_block)
            resource = _mapping(resource_mapping.get("resource"))
            for scope in _sequence(resource_mapping.get("scopeSpans")):
                for span in _sequence(_mapping(scope).get("spans")):
                    yield self.parse_span(_mapping(span), resource)


def _service_name(
    record: Mapping[str, object], resource: Mapping[str, object] | None
) -> str:
    direct = _optional_text(record, "service", "service.name", "service_name")
    if direct:
        return direct
    attributes = _attributes(resource or _mapping(record.get("resource")))
    service = attributes.get("service.name")
    if service is None or not str(service).strip():
        raise OpenTelemetryParseError("resource attribute service.name is required")
    return str(service)


def _attributes(resource: Mapping[str, object]) -> dict[str, object]:
    raw = resource.get("attributes", resource)
    if isinstance(raw, Mapping):
        return {str(key): _any_value(value) for key, value in raw.items()}
    result: dict[str, object] = {}
    for attribute in _sequence(raw):
        item = _mapping(attribute)
        key = item.get("key")
        if key is not None:
            result[str(key)] = _any_value(item.get("value"))
    return result


def _metric_points(metric: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    for kind in ("gauge", "sum"):
        container = _mapping(metric.get(kind))
        if container:
            return tuple(_mapping(point) for point in _sequence(container.get("dataPoints")))
    if "dataPoint" in metric or "data_point" in metric:
        return (_mapping(metric.get("dataPoint") or metric.get("data_point")),)
    raise OpenTelemetryParseError("metric requires gauge/sum dataPoints")


def _timestamp(record: Mapping[str, object], *names: str) -> datetime:
    timestamp = _optional_timestamp(record, *names)
    if timestamp is None:
        raise OpenTelemetryParseError(f"required timestamp missing; expected one of {names}")
    return timestamp


def _optional_timestamp(
    record: Mapping[str, object], *names: str
) -> datetime | None:
    for name in names:
        if name not in record or record[name] in (None, ""):
            continue
        value = record[name]
        if isinstance(value, datetime):
            return value
        try:
            nanoseconds = int(str(value))
        except ValueError as error:
            raise OpenTelemetryParseError(f"invalid nanosecond timestamp {value!r}") from error
        seconds, remainder = divmod(nanoseconds, 1_000_000_000)
        return datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
            microseconds=remainder // 1_000
        )
    return None


def _required_text(record: Mapping[str, object], *names: str) -> str:
    value = _optional_text(record, *names)
    if value is None:
        raise OpenTelemetryParseError(f"required field missing; expected one of {names}")
    return value


def _optional_text(record: Mapping[str, object], *names: str) -> str | None:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _numeric(record: Mapping[str, object], *names: str) -> float:
    for name in names:
        value = record.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError) as error:
                raise OpenTelemetryParseError(f"metric value {value!r} is not numeric") from error
    raise OpenTelemetryParseError(f"metric value missing; expected one of {names}")


def _severity(record: Mapping[str, object]) -> str:
    text = _optional_text(record, "severityText", "severity_text", "severity")
    if text:
        return text
    number = record.get("severityNumber", record.get("severity_number"))
    if number is None:
        return "UNSPECIFIED"
    try:
        value = int(str(number))
    except ValueError as error:
        raise OpenTelemetryParseError(f"invalid severity number {number!r}") from error
    if value <= 0:
        return "UNSPECIFIED"
    if value <= 4:
        return "TRACE"
    if value <= 8:
        return "DEBUG"
    if value <= 12:
        return "INFO"
    if value <= 16:
        return "WARN"
    if value <= 20:
        return "ERROR"
    return "FATAL"


def _span_status(raw: object) -> str:
    if raw is None:
        return "UNSET"
    if isinstance(raw, Mapping):
        raw = raw.get("code", raw.get("statusCode", "UNSET"))
    text = str(raw).upper()
    if text in {"2", "ERROR", "STATUS_CODE_ERROR"}:
        return "ERROR"
    if text in {"1", "OK", "STATUS_CODE_OK"}:
        return "OK"
    return "UNSET"


def _any_value(raw: object) -> object | None:
    if isinstance(raw, Mapping):
        for name in (
            "stringValue",
            "intValue",
            "doubleValue",
            "boolValue",
            "bytesValue",
        ):
            if name in raw:
                return raw[name]
    return raw


def _mapping(raw: object) -> Mapping[str, object]:
    return raw if isinstance(raw, Mapping) else {}


def _sequence(raw: object) -> Sequence[object]:
    return raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()
