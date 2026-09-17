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


def test_service_summary_has_one_configured_cost() -> None:
    api = TelemetryQueryAPI(
        _store(), total_budget=4, costs=QueryCosts(service_summary=4)
    )
    summary = api.query_service_summary("api", NOW, NOW + timedelta(minutes=1))

    assert summary.metric_means == {"request_rate": 12}
    assert summary.log_count == 1
    assert summary.span_count == 1
    assert api.remaining_budget == 0
