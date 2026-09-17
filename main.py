"""Demonstrate the implemented telemetry/simulation foundation without diagnosis."""

from __future__ import annotations

from src.core import QueryCosts, TelemetryQueryAPI
from src.simulation import FaultRequest, IncidentGenerator, benchmark_config


def main() -> None:
    incident = IncidentGenerator(benchmark_config()).generate(
        seed=17,
        fault=FaultRequest(service="postgres", failure_mode="database_slowdown"),
    )
    store = incident.telemetry
    print(f"Incident: {incident.incident_id}")
    print(f"Known services: {', '.join(sorted(incident.baseline.known_services))}")
    print(
        "Telemetry counts: "
        f"metrics={store.metric_count}, logs={store.log_count}, spans={store.span_count}"
    )

    api = TelemetryQueryAPI(store, total_budget=4, costs=QueryCosts())
    start, end = incident.observed_start_time, incident.observed_end_time
    metrics = api.query_metrics("gateway", start, end)
    logs = api.query_logs("gateway", start, end)
    spans = api.query_traces("gateway", start, end)
    api.query_metrics("gateway", start, end)
    print(f"Queried gateway: metrics={len(metrics)}, logs={len(logs)}, spans={len(spans)}")
    print(
        f"Budget: spent={api.spent_budget}, remaining={api.remaining_budget}; "
        f"repeat metric query cached={api.query_history[-1].served_from_cache}"
    )

    print("Agent-visible data: canonical telemetry plus historical baseline only")
    truth = incident.ground_truth
    print(
        "Evaluation-only ground truth (kept outside TelemetryQueryAPI): "
        f"root=({truth.root_service}, {truth.failure_mode}), "
        f"affected={len(truth.affected_services)}, decoys={list(truth.decoy_services)}"
    )


if __name__ == "__main__":
    main()
