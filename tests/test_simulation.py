from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import MetricEvent, TelemetryQueryAPI
from src.simulation import (
    FaultRequest,
    FragmentationConfig,
    IncidentGenerator,
    ServiceConfig,
    SimulationConfig,
    benchmark_config,
)
from src.simulation.simulator import MicroserviceSimulator, ServiceState


NO_FRAGMENTATION = FragmentationConfig(
    clock_skew_range_seconds=0,
    missing_observation_probability=0,
    delayed_observation_probability=0,
    metric_noise_fraction=0,
    include_decoy_anomaly=False,
)


def _query_all_metrics(incident, service: str):
    api = TelemetryQueryAPI(incident.telemetry, 1)
    return api.query_metrics(
        service,
        incident.observed_start_time - timedelta(minutes=1),
        incident.observed_end_time + timedelta(minutes=1),
    )


def test_same_seed_and_configuration_are_deterministic() -> None:
    generator = IncidentGenerator(benchmark_config(NO_FRAGMENTATION))
    requested = FaultRequest("postgres", "database_slowdown")

    first = generator.generate(11, requested)
    second = generator.generate(11, requested)

    assert first.ground_truth == second.ground_truth
    assert _query_all_metrics(first, "gateway") == _query_all_metrics(second, "gateway")
    assert first.telemetry.log_count == second.telemetry.log_count
    assert first.telemetry.span_count == second.telemetry.span_count


def test_arbitrary_configured_service_names_and_all_telemetry_types_work() -> None:
    config = SimulationConfig(
        services=(
            ServiceConfig("Edge V2", 80, 5, 40, ("Ledger_X",)),
            ServiceConfig("Ledger_X", 70, 8),
        ),
        baseline_steps=3,
        incident_steps=4,
        fragmentation=NO_FRAGMENTATION,
    )
    incident = IncidentGenerator(config).generate(
        5, FaultRequest("Ledger_X", "network_latency")
    )

    assert incident.baseline.known_services == {"edge-v2", "ledger-x"}
    assert incident.ground_truth.root_service == "ledger-x"
    assert incident.telemetry.metric_count > 0
    assert incident.telemetry.log_count > 0
    assert incident.telemetry.span_count > 0


def test_trace_parent_child_relationships_are_valid() -> None:
    incident = IncidentGenerator(benchmark_config(NO_FRAGMENTATION)).generate(
        9, FaultRequest("payment", "deployment_regression")
    )
    api = TelemetryQueryAPI(incident.telemetry, 20)
    spans = []
    for service in incident.baseline.known_services:
        spans.extend(
            api.query_traces(
                service,
                incident.observed_start_time - timedelta(seconds=1),
                incident.observed_end_time + timedelta(seconds=1),
            )
        )
    by_trace: dict[str, list] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    assert by_trace
    for trace in by_trace.values():
        by_id = {span.span_id: span for span in trace}
        span_ids = {span.span_id for span in trace}
        roots = [span for span in trace if span.parent_span_id is None]
        assert len(roots) == 1
        assert all(
            span.parent_span_id is None or span.parent_span_id in span_ids for span in trace
        )
        assert all(span.duration_ms >= 0 for span in trace)
        children_by_parent: dict[str, list] = {}
        for span in trace:
            if span.parent_span_id is None:
                continue
            parent = by_id[span.parent_span_id]
            child_end = span.start_timestamp + timedelta(milliseconds=span.duration_ms)
            parent_end = parent.start_timestamp + timedelta(milliseconds=parent.duration_ms)
            assert parent.start_timestamp <= span.start_timestamp
            assert child_end <= parent_end
            children_by_parent.setdefault(parent.span_id, []).append(span)
        for siblings in children_by_parent.values():
            ordered = sorted(siblings, key=lambda span: span.start_timestamp)
            assert all(
                left.start_timestamp + timedelta(milliseconds=left.duration_ms)
                <= right.start_timestamp
                for left, right in zip(ordered, ordered[1:])
            )


def test_fault_changes_state_and_cascades_upstream() -> None:
    config = benchmark_config(NO_FRAGMENTATION)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    simulator = MicroserviceSimulator(config, seed=3)
    output = simulator.run(
        start,
        config.baseline_steps,
        "postgres",
        "database_slowdown",
        {"latency_factor": 10.0},
    )
    before = output.state_history[config.baseline_steps - 1]
    after = output.state_history[config.baseline_steps]

    assert after["postgres"].latency_ms > before["postgres"].latency_ms * 5
    assert after["orders"].latency_ms > before["orders"].latency_ms
    assert after["gateway"].latency_ms > before["gateway"].latency_ms
    assert after["gateway"].error_rate > before["gateway"].error_rate


def test_clock_skew_changes_event_timestamp_independently_of_delivery_delay() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        clock_offsets_seconds={"gateway": 13.0},
        missing_observation_probability=0,
        delayed_observation_probability=1,
        max_delay_seconds=2,
        metric_noise_fraction=0,
        include_decoy_anomaly=False,
    )
    config = benchmark_config(fragmentation)
    incident = IncidentGenerator(config).generate(
        2, FaultRequest("auth", "cpu_saturation")
    )
    first_gateway_metric = _query_all_metrics(incident, "gateway")[0]
    true_start = incident.ground_truth.true_fault_start_time - timedelta(
        seconds=config.baseline_steps * config.step_seconds
    )

    assert first_gateway_metric.event_timestamp == true_start + timedelta(seconds=13)
    assert true_start < first_gateway_metric.arrival_timestamp <= true_start + timedelta(
        seconds=2
    )
    assert first_gateway_metric.arrival_timestamp != first_gateway_metric.event_timestamp
    assert incident.ground_truth.clock_offsets_seconds["gateway"] == 13


def test_missing_observations_can_be_generated() -> None:
    clean = IncidentGenerator(benchmark_config(NO_FRAGMENTATION)).generate(
        21, FaultRequest("auth", "network_latency")
    )
    fragmented_config = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0.35,
        delayed_observation_probability=0,
        metric_noise_fraction=0,
        include_decoy_anomaly=False,
    )
    fragmented = IncidentGenerator(benchmark_config(fragmented_config)).generate(
        21, FaultRequest("auth", "network_latency")
    )

    assert fragmented.telemetry.metric_count < clean.telemetry.metric_count


def test_delivery_delay_changes_arrival_but_not_event_timestamps() -> None:
    clean = IncidentGenerator(benchmark_config(NO_FRAGMENTATION)).generate(
        21, FaultRequest("auth", "network_latency")
    )
    delayed_config = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=1,
        max_delay_seconds=20,
        metric_noise_fraction=0,
        include_decoy_anomaly=False,
    )
    delayed = IncidentGenerator(benchmark_config(delayed_config)).generate(
        21, FaultRequest("auth", "network_latency")
    )
    clean_metric = _query_all_metrics(clean, "gateway")[0]
    delayed_metric = _query_all_metrics(delayed, "gateway")[0]

    assert delayed_metric.event_timestamp == clean_metric.event_timestamp
    assert delayed_metric.arrival_timestamp > clean_metric.arrival_timestamp


def test_decoy_anomaly_is_recorded_and_visible_in_metrics() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0,
        include_decoy_anomaly=True,
    )
    incident = IncidentGenerator(benchmark_config(fragmentation)).generate(
        8, FaultRequest("postgres", "database_slowdown")
    )
    decoy = incident.ground_truth.decoy_services[0]
    cpu_events = [
        event
        for event in _query_all_metrics(incident, decoy)
        if event.metric_name == "cpu_utilization"
        and event.event_timestamp >= incident.ground_truth.true_fault_start_time
    ]
    baseline_cpu = incident.baseline.metric_summaries[(decoy, "cpu_utilization")].mean

    assert decoy != incident.ground_truth.root_service
    assert decoy not in incident.ground_truth.affected_services
    assert max(event.value for event in cpu_events) > baseline_cpu + 0.25


def test_decoy_is_omitted_when_every_service_is_structurally_affected() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0,
        include_decoy_anomaly=True,
    )
    config = SimulationConfig(
        services=(
            ServiceConfig("entry", 80, 5, 40, ("store",)),
            ServiceConfig("store", 70, 8),
        ),
        baseline_steps=3,
        incident_steps=4,
        fragmentation=fragmentation,
    )

    incident = IncidentGenerator(config).generate(
        5, FaultRequest("store", "network_latency")
    )

    assert incident.ground_truth.affected_services == {"entry", "store"}
    assert incident.ground_truth.decoy_services == ()


def test_ground_truth_records_effective_default_fault_parameters() -> None:
    incident = IncidentGenerator(benchmark_config(NO_FRAGMENTATION)).generate(
        6, FaultRequest("postgres", "database_slowdown")
    )

    assert incident.ground_truth.injected_fault_parameters == {"latency_factor": 7.0}


def test_baseline_selection_uses_time_boundary_not_metric_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_metric_events = MicroserviceSimulator._metric_events

    def emit_extra_metric(
        self: MicroserviceSimulator,
        event_time: datetime,
        states: dict[str, ServiceState],
    ) -> list[MetricEvent]:
        events = original_metric_events(self, event_time, states)
        events.extend(
            MetricEvent(
                event_time,
                event_time,
                service,
                "worker_count",
                4,
                "workers",
            )
            for service in states
        )
        return events

    monkeypatch.setattr(MicroserviceSimulator, "_metric_events", emit_extra_metric)
    config = benchmark_config(NO_FRAGMENTATION)
    incident = IncidentGenerator(config).generate(
        12, FaultRequest("postgres", "database_slowdown")
    )

    for service in incident.baseline.known_services:
        summary = incident.baseline.metric_summaries[(service, "worker_count")]
        assert summary.count == config.baseline_steps
        assert summary.mean == 4


def test_ground_truth_is_not_reachable_through_query_api() -> None:
    incident = IncidentGenerator(benchmark_config(NO_FRAGMENTATION)).generate(4)
    api = TelemetryQueryAPI(incident.telemetry, 2)

    assert not hasattr(api, "ground_truth")
    assert not hasattr(incident.telemetry, "ground_truth")
