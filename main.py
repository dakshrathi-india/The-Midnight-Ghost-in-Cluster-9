"""Run the deterministic diagnosis, remediation, and recovery demonstration."""

from __future__ import annotations

from datetime import timedelta

from src.remediation import RecoveryController
from src.simulation import (
    FaultRequest,
    IncidentGenerator,
    SimulatorRemediationExecutor,
    benchmark_config,
)


def main() -> None:
    session = IncidentGenerator(benchmark_config()).start_session(
        seed=17,
        fault=FaultRequest(service="postgres", failure_mode="database_slowdown"),
    )
    incident = session.initial_incident
    analysis_start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    api = session.create_query_api(total_budget=17)
    result = RecoveryController().run(
        api,
        incident.baseline,
        analysis_start,
        incident.observed_end_time,
        SimulatorRemediationExecutor(session.simulator),
        session,
    )
    print(f"Incident: {incident.incident_id}")
    hypothesis = result.diagnosis.best_hypothesis
    pair = (
        f"({hypothesis.service}, {hypothesis.failure_mode})"
        if hypothesis is not None
        else "none"
    )
    print(
        f"Diagnosis: {result.diagnosis.status.value} {pair}; "
        f"budget={result.diagnosis.budget_spent}/{api.total_budget}"
    )
    action = result.planning.action
    print(f"Planned action: {action.action_type.value if action else 'none'}")
    print(
        "Execution: "
        + (result.execution.status.value if result.execution is not None else "not run")
    )
    if result.verification is not None:
        print(
            "Verification services: "
            + ", ".join(result.verification.checked_services)
        )
        print(f"Recovery verification: {result.verification.status.value}")
    else:
        print("Recovery verification: not run")
    print(f"Total global budget: {api.spent_budget}/{api.total_budget}")

    truth = incident.ground_truth
    print(
        "EVALUATION-ONLY ground truth: "
        f"({truth.root_service}, {truth.failure_mode}); "
        f"decoys={list(truth.decoy_services)}"
    )


if __name__ == "__main__":
    main()
