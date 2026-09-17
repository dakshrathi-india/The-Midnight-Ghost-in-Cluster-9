"""Demonstrate telemetry collection and observability intelligence without RCA."""

from __future__ import annotations

from datetime import timedelta

from src.core import QueryCosts, TelemetryQueryAPI
from src.rca import (
    CUSUMDetector,
    CandidateGenerator,
    CandidateStrength,
    IsolationForestDetector,
    LogEvidenceExtractor,
    MADDetector,
    TraceGraphAnalyzer,
)
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

    services = sorted(incident.baseline.known_services)
    api = TelemetryQueryAPI(
        store, total_budget=len(services) * 3, costs=QueryCosts()
    )
    start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    end = incident.observed_end_time
    metrics_by_service = {
        service: api.query_metrics(service, start, end) for service in services
    }
    logs = tuple(
        event
        for service in services
        for event in api.query_logs(service, start, end)
    )
    spans = tuple(
        event
        for service in services
        for event in api.query_traces(service, start, end)
    )
    api.query_metrics(services[0], start, end)
    print(
        f"Budget: spent={api.spent_budget}, remaining={api.remaining_budget}; "
        f"repeat metric query cached={api.query_history[-1].served_from_cache}"
    )

    graph = TraceGraphAnalyzer().reconstruct(spans, incident.baseline)
    mad_results = {
        service: MADDetector().analyze(
            service, metrics_by_service[service], incident.baseline
        )
        for service in services
    }
    change_results = {
        service: CUSUMDetector().analyze(
            service, metrics_by_service[service], incident.baseline
        )
        for service in services
    }
    isolation_results = {
        service: IsolationForestDetector().analyze(
            service, metrics_by_service[service], incident.baseline
        )
        for service in services
    }
    log_evidence = LogEvidenceExtractor().extract(logs)
    candidates = CandidateGenerator().generate(
        services,
        mad_results,
        change_results,
        isolation_results,
        graph.service_evidence,
        log_evidence,
    )

    edge_text = ", ".join(
        f"{edge.caller_service}->{edge.callee_service}({edge.support_count})"
        for edge in graph.edges
    )
    print(f"Trace-derived edges: {edge_text}")
    for service in services:
        fired = [
            name
            for name, result in (
                ("MAD", mad_results[service]),
                ("CUSUM", change_results[service]),
                ("IF", isolation_results[service]),
            )
            if result.flagged
        ]
        if fired:
            print(f"Detector flags: {service}={'+'.join(fired)}")
    visible_candidates = [
        candidate
        for candidate in candidates
        if candidate.strength is not CandidateStrength.NOT_CANDIDATE
    ]
    print(
        "Candidates: "
        + ", ".join(
            f"{candidate.service}:{candidate.strength.value}"
            for candidate in visible_candidates
        )
    )

    print("Analysis input: queried canonical telemetry plus historical baseline only")
    truth = incident.ground_truth
    print(
        "Evaluation-only ground truth (kept outside TelemetryQueryAPI): "
        f"root=({truth.root_service}, {truth.failure_mode}), "
        f"affected={len(truth.affected_services)}, decoys={list(truth.decoy_services)}"
    )


if __name__ == "__main__":
    main()
