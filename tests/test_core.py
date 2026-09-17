from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import (
    BudgetExceededError,
    LogEvent,
    MetricEvent,
    QueryCosts,
    SpanEvent,
    TelemetryQueryAPI,
    TelemetryStore,
)
from src.core.normalization import TelemetryNormalizer


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _store() -> TelemetryStore:
    return TelemetryStore(
        [MetricEvent(NOW, NOW, "api", "request_rate", 12, "requests/s")],
        [LogEvent(NOW, NOW, "api", "INFO", "ready")],
        [SpanEvent("t1", "s1", None, "api", "GET /", NOW, NOW, 4, "OK")],
    )


def test_normalization_preserves_and_normalizes_event_and_arrival_timestamps() -> None:
    naive = datetime(2025, 1, 1)
    naive_arrival = naive + timedelta(seconds=7)
    metric = MetricEvent(
        naive, naive_arrival, "Checkout API", "Latency", 0.25, "s"
    )
    log = LogEvent(
        naive, naive_arrival, "Checkout API", "warn", "  slow request  "
    )
    span = SpanEvent(
        "t1",
        "s1",
        None,
        "Checkout API",
        "GET /checkout",
        naive,
        naive_arrival,
        250,
        "ok",
    )
    normalizer = TelemetryNormalizer()

    metrics = normalizer.normalize_metrics([metric, metric])
    logs = normalizer.normalize_logs([log, log])
    spans = normalizer.normalize_spans([span, span])

    assert len(metrics) == len(logs) == len(spans) == 1
    assert metrics[0].service == "checkout-api"
    assert metrics[0].metric_name == "latency"
    assert metrics[0].value == 250
    assert metrics[0].unit == "ms"
    assert metrics[0].event_timestamp == naive.replace(tzinfo=timezone.utc)
    assert metrics[0].arrival_timestamp == naive_arrival.replace(tzinfo=timezone.utc)
    assert logs[0].event_timestamp == naive.replace(tzinfo=timezone.utc)
    assert logs[0].arrival_timestamp == naive_arrival.replace(tzinfo=timezone.utc)
    assert spans[0].start_timestamp == naive.replace(tzinfo=timezone.utc)
    assert spans[0].arrival_timestamp == naive_arrival.replace(tzinfo=timezone.utc)
    assert logs[0].severity == "WARNING"
    assert logs[0].message == "slow request"


def test_semantic_deduplication_prefers_earliest_arrival() -> None:
    late = NOW + timedelta(seconds=8)
    early = NOW + timedelta(seconds=2)
    normalizer = TelemetryNormalizer()

    metrics = normalizer.normalize_metrics(
        [
            MetricEvent(NOW, late, "api", "cpu", 0.5, "ratio"),
            MetricEvent(NOW, early, "api", "cpu", 0.5, "ratio"),
        ]
    )
    logs = normalizer.normalize_logs(
        [
            LogEvent(NOW, late, "api", "ERROR", "failed", "trace-1"),
            LogEvent(NOW, early, "api", "ERROR", "failed", "trace-1"),
        ]
    )
    spans = normalizer.normalize_spans(
        [
            SpanEvent("trace-1", "span-1", None, "api", "work", NOW, late, 5, "OK"),
            SpanEvent("trace-1", "span-1", None, "api", "work", NOW, early, 5, "OK"),
        ]
    )

    assert len(metrics) == len(logs) == len(spans) == 1
    assert metrics[0].arrival_timestamp == early
    assert logs[0].arrival_timestamp == early
    assert spans[0].arrival_timestamp == early


def test_semantic_deduplication_keeps_distinct_events() -> None:
    normalizer = TelemetryNormalizer()

    metrics = normalizer.normalize_metrics(
        [
            MetricEvent(NOW, NOW, "api", "cpu", 0.5, "ratio"),
            MetricEvent(NOW, NOW, "api", "cpu", 0.6, "ratio"),
        ]
    )
    logs = normalizer.normalize_logs(
        [
            LogEvent(NOW, NOW, "api", "ERROR", "failure one", "trace-1"),
            LogEvent(NOW, NOW, "api", "ERROR", "failure two", "trace-1"),
        ]
    )

    assert len(metrics) == 2
    assert len(logs) == 2


def test_budget_costs_cache_and_history() -> None:
    api = TelemetryQueryAPI(
        _store(), total_budget=6, costs=QueryCosts(metrics=1, logs=2, traces=3)
    )
    end = NOW + timedelta(minutes=1)

    first = api.query_logs("api", NOW, end)
    second = api.query_logs("api", NOW, end)
    api.query_metrics("api", NOW, end)

    assert first is second
    assert api.spent_budget == 3
    assert api.remaining_budget == 3
    assert [entry.cost for entry in api.query_history] == [2, 0, 1]
    assert api.query_history[1].served_from_cache


def test_budget_exceeded_raises_specific_error_without_recording_query() -> None:
    api = TelemetryQueryAPI(_store(), total_budget=2, costs=QueryCosts(traces=3))

    with pytest.raises(BudgetExceededError):
        api.query_traces("api", NOW, NOW + timedelta(minutes=1))

    assert api.spent_budget == 0
    assert api.query_history == ()


def test_trace_query_returns_complete_available_cross_service_tree() -> None:
    spans = [
        SpanEvent(
            "trace-tree",
            "front-span",
            None,
            "front",
            "front.request",
            NOW - timedelta(seconds=1),
            NOW,
            100,
            "OK",
        ),
        SpanEvent(
            "trace-tree",
            "worker-span",
            "front-span",
            "worker",
            "worker.request",
            NOW,
            NOW,
            80,
            "OK",
        ),
        SpanEvent(
            "trace-tree",
            "datastore-span",
            "worker-span",
            "datastore",
            "datastore.query",
            NOW + timedelta(seconds=1),
            NOW + timedelta(seconds=1),
            70,
            "OK",
        ),
        SpanEvent(
            "unrelated-trace",
            "other-span",
            None,
            "other",
            "other.request",
            NOW,
            NOW,
            5,
            "OK",
        ),
    ]
    api = TelemetryQueryAPI(TelemetryStore((), (), spans), total_budget=1)

    result = api.query_traces("worker", NOW, NOW)

    assert [(span.service, span.span_id) for span in result] == [
        ("front", "front-span"),
        ("worker", "worker-span"),
        ("datastore", "datastore-span"),
    ]


def test_repeated_complete_trace_query_is_cached_at_zero_cost() -> None:
    end = NOW + timedelta(minutes=1)
    api = TelemetryQueryAPI(
        _store(), total_budget=3, costs=QueryCosts(traces=3)
    )

    first = api.query_traces("api", NOW, end)
    second = api.query_traces("api", NOW, end)

    assert first is second
    assert api.spent_budget == 3
    assert [entry.cost for entry in api.query_history] == [3, 0]
    assert api.query_history[1].served_from_cache


def test_service_summary_has_one_configured_cost() -> None:
    api = TelemetryQueryAPI(
        _store(), total_budget=4, costs=QueryCosts(service_summary=4)
    )
    summary = api.query_service_summary("api", NOW, NOW + timedelta(minutes=1))

    assert summary.metric_means == {"request_rate": 12}
    assert summary.log_count == 1
    assert summary.span_count == 1
    assert api.remaining_budget == 0
