"""Run the deterministic budget-aware diagnosis demonstration."""

from __future__ import annotations

from datetime import timedelta

from src.core import QueryCosts, TelemetryQueryAPI
from src.rca import CandidateStrength, DiagnosisAgent
from src.simulation import FaultRequest, IncidentGenerator, benchmark_config


def main() -> None:
    incident = IncidentGenerator(benchmark_config()).generate(
        seed=17,
        fault=FaultRequest(service="postgres", failure_mode="database_slowdown"),
    )
    analysis_start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    api = TelemetryQueryAPI(
        incident.telemetry,
        total_budget=17,
        costs=QueryCosts(metrics=1, logs=1, traces=1),
    )

    result = DiagnosisAgent().diagnose(
        api,
        incident.baseline,
        analysis_start,
        incident.observed_end_time,
    )

    candidates = [
        candidate
        for candidate in result.candidates
        if candidate.strength is not CandidateStrength.NOT_CANDIDATE
    ]
    print(f"Incident: {incident.incident_id}")
    print(
        "Candidates: "
        + ", ".join(
            f"{candidate.service}:{candidate.strength.value}"
            for candidate in candidates
        )
    )
    print(
        "Query sequence: "
        + " -> ".join(
            f"{entry.query_type}({entry.service})[{entry.cost}]"
            for entry in result.queries_executed
        )
    )
    print(
        f"Diagnosis status: {result.status.value}; "
        f"budget={result.budget_spent}/{api.total_budget}"
    )
    if result.ranked_hypotheses:
        best = result.ranked_hypotheses[0]
        hypothesis = best.hypothesis
        print(f"Best hypothesis: ({hypothesis.service}, {hypothesis.failure_mode})")
        print(
            "Supporting evidence: "
            + "; ".join(
                f"{item.category.value}:{item.observation}"
                for item in best.supporting_evidence
            )
        )
        print(
            "Contradictions: "
            + (
                "; ".join(item.explanation for item in best.contradictions)
                if best.contradictions
                else "none"
            )
        )

    truth = incident.ground_truth
    print(
        "Evaluation-only ground truth: "
        f"({truth.root_service}, {truth.failure_mode}); "
        f"decoys={list(truth.decoy_services)}"
    )


if __name__ == "__main__":
    main()
