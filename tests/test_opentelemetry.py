from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.adapters import OpenTelemetryAdapter, OpenTelemetryParseError
from src.core import TelemetryQueryAPI, TelemetryStore


RESOURCE = {
    "attributes": [
        {"key": "service.name", "value": {"stringValue": "Checkout API"}},
        {"key": "deployment.environment", "value": {"stringValue": "test"}},
    ]
}
START_NS = 1_735_689_600_000_000_000


def test_otlp_metric_parsing_preserves_event_and_arrival_time() -> None:
    event = OpenTelemetryAdapter().parse_metric(
        {
            "name": "request latency",
            "unit": "s",
            "dataPoint": {
                "timeUnixNano": str(START_NS),
                "observedTimeUnixNano": str(START_NS + 2_000_000_000),
                "asDouble": 0.125,
            },
        },
        RESOURCE,
    )

    assert event.service == "checkout-api"
    assert event.metric_name == "request_latency"
    assert event.value == 125.0
    assert event.unit == "ms"
    assert event.event_timestamp == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert (event.arrival_timestamp - event.event_timestamp).total_seconds() == 2


def test_otlp_log_parsing_handles_numeric_severity_and_trace_linkage() -> None:
    event = OpenTelemetryAdapter().parse_log(
        {
            "timeUnixNano": str(START_NS),
            "observedTimeUnixNano": str(START_NS + 1_000_000),
            "severityNumber": 18,
            "body": {"stringValue": "database request failed"},
            "traceId": "abc123",
            "attributes": [{"key": "error.type", "value": {"stringValue": "timeout"}}],
        },
        RESOURCE,
    )

    assert event.service == "checkout-api"
    assert event.severity == "ERROR"
    assert event.message == "database request failed"
    assert event.trace_id == "abc123"


def test_otlp_span_parsing_derives_duration_and_status() -> None:
    event = OpenTelemetryAdapter().parse_span(
        {
            "traceId": "trace-1",
            "spanId": "span-1",
            "parentSpanId": "parent-1",
            "name": "checkout.submit",
            "startTimeUnixNano": str(START_NS),
            "endTimeUnixNano": str(START_NS + 25_000_000),
            "status": {"code": "STATUS_CODE_ERROR"},
        },
        RESOURCE,
    )

    assert event.trace_id == "trace-1"
    assert event.parent_span_id == "parent-1"
    assert event.duration_ms == 25.0
    assert event.status == "ERROR"
    assert event.arrival_timestamp > event.start_timestamp


@pytest.mark.parametrize(
    "kind,record",
    [
        ("metric", {"name": "cpu", "value": 0.5}),
        ("log", {"timeUnixNano": str(START_NS)}),
        (
            "span",
            {
                "spanId": "s1",
                "name": "work",
                "startTimeUnixNano": str(START_NS),
                "duration_ms": 1,
            },
        ),
    ],
)
def test_malformed_otlp_required_fields_fail_clearly(
    kind: str, record: dict[str, object]
) -> None:
    adapter = OpenTelemetryAdapter()
    parser = {
        "metric": adapter.parse_metric,
        "log": adapter.parse_log,
        "span": adapter.parse_span,
    }[kind]

    with pytest.raises(OpenTelemetryParseError):
        parser(record, RESOURCE)


def test_otlp_export_batch_enters_existing_query_boundary() -> None:
    payload = {
        "resourceMetrics": [
            {
                "resource": RESOURCE,
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "cpu_utilization",
                                "unit": "ratio",
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "timeUnixNano": str(START_NS),
                                            "asDouble": 0.6,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ],
            }
        ],
        "resourceLogs": [
            {
                "resource": RESOURCE,
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "timeUnixNano": str(START_NS),
                                "severityText": "INFO",
                                "body": {"stringValue": "ready"},
                            }
                        ]
                    }
                ],
            }
        ],
        "resourceSpans": [
            {
                "resource": RESOURCE,
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "s1",
                                "name": "checkout.work",
                                "startTimeUnixNano": str(START_NS),
                                "endTimeUnixNano": str(START_NS + 1_000_000),
                            }
                        ]
                    }
                ],
            }
        ],
    }
    batch = OpenTelemetryAdapter().parse_export(payload)
    api = TelemetryQueryAPI(
        TelemetryStore(batch.metrics, batch.logs, batch.spans), total_budget=3
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert api.query_metrics("checkout-api", start, start)[0].value == 0.6
    assert api.query_logs("checkout-api", start, start)[0].message == "ready"
    assert api.query_traces("checkout-api", start, start)[0].span_id == "s1"


def test_opentelemetry_adapter_has_no_simulation_dependency() -> None:
    source = (Path(__file__).parents[1] / "src" / "adapters" / "opentelemetry.py").read_text()

    assert "src.simulation" not in source
    assert "except Exception" not in source
