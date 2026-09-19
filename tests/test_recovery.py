from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core import BaselineStore, MetricEvent, TelemetryQueryAPI, TelemetryStore
from src.rca import (
    CausalEvidence,
    CandidateStrength,
    DiagnosisAgent,
    DiagnosisResult,
    DiagnosisStatus,
    EvidenceCategory,
    EvidenceStatus,
    Hypothesis,
    HypothesisEvaluation,
    RankingComponents,
    ServiceCandidate,
)
from src.remediation import (
    ExecutionReceipt,
    ExecutionStatus,
    FreshObservationWindow,
    IncidentPhase,
    IncidentTransition,
    PlanningStatus,
    RecoveryController,
    RecoveryControllerConfig,
    RecoveryStatus,
    RecoveryVerifier,
    RemediationAction,
    RemediationActionType,
    SafeRemediationPlanner,
    validate_incident_transitions,
)
from src.simulation import (
    FaultRequest,
    FragmentationConfig,
    IncidentGenerator,
    SimulatorRemediationExecutor,
    benchmark_config,
)


START = datetime(2025, 4, 1, tzinfo=timezone.utc)
NO_FRAGMENTATION = FragmentationConfig(
    clock_skew_range_seconds=0,
    missing_observation_probability=0,
    delayed_observation_probability=0,
    metric_noise_fraction=0,
    include_decoy_anomaly=False,
)


def _metric_batch(
    service: str,
    *,
    latency: float = 10.0,
    cpu: float = 0.2,
    request_rate: float = 100.0,
    count: int = 5,
) -> list[MetricEvent]:
    result: list[MetricEvent] = []
    for index in range(count):
        timestamp = START + timedelta(minutes=1, seconds=index)
        result.extend(
            (
                MetricEvent(timestamp, timestamp, service, "request_latency_ms", latency, "ms"),
                MetricEvent(timestamp, timestamp, service, "cpu_utilization", cpu, "ratio"),
                MetricEvent(timestamp, timestamp, service, "memory_utilization", 0.3, "ratio"),
                MetricEvent(timestamp, timestamp, service, "error_rate", 0.0, "ratio"),
                MetricEvent(timestamp, timestamp, service, "request_rate", request_rate, "requests/s"),
                MetricEvent(timestamp, timestamp, service, "queue_length", 0.0, "requests"),
            )
        )
    return result


def _baseline(services: set[str]) -> BaselineStore:
    metrics: list[MetricEvent] = []
    for service in services:
        for index in range(8):
            timestamp = START + timedelta(seconds=index)
            variation = (index % 3 - 1) * 0.01
            metrics.extend(
                (
                    MetricEvent(timestamp, timestamp, service, "request_latency_ms", 10.0 + variation, "ms"),
                    MetricEvent(timestamp, timestamp, service, "cpu_utilization", 0.2 + variation, "ratio"),
                    MetricEvent(timestamp, timestamp, service, "memory_utilization", 0.3 + variation, "ratio"),
                    MetricEvent(timestamp, timestamp, service, "error_rate", 0.0, "ratio"),
                    MetricEvent(timestamp, timestamp, service, "request_rate", 100.0 + variation, "requests/s"),
                    MetricEvent(timestamp, timestamp, service, "queue_length", 0.0, "requests"),
                )
            )
    return BaselineStore.from_metrics(services, metrics)


def _diagnosis(
    *,
    explained: tuple[str, ...] = ("root", "upstream"),
    strong_services: tuple[str, ...] = ("root", "upstream"),
    status: DiagnosisStatus = DiagnosisStatus.RESOLVED,
) -> DiagnosisResult:
    hypothesis = Hypothesis("root", "database_slowdown", CandidateStrength.STRONG)
    evidence = (
        CausalEvidence(
            EvidenceCategory.METRIC_PATTERN,
            "root",
            "database_slowdown",
            "required latency is high",
            EvidenceStatus.SUPPORT,
            "The required metric pattern is present at the root.",
        ),
        CausalEvidence(
            EvidenceCategory.TRACE_LOCALIZATION,
            "root",
            "database_slowdown",
            "local latency",
            EvidenceStatus.SUPPORT,
            "Direct root trace localization.",
        ),
    )
    evaluation = HypothesisEvaluation(
        hypothesis,
        evidence,
        explained,
        (),
        RankingComponents(0, 0, len(explained), 2, 2),
    )
    candidates = tuple(
        ServiceCandidate(
            service,
            CandidateStrength.STRONG if service in strong_services else CandidateStrength.NOT_CANDIDATE,
            ("mad", "change_point") if service in strong_services else (),
            2 if service in strong_services else 0,
            False,
            (),
        )
        for service in ("decoy", "healthy", "root", "upstream")
    )
    return DiagnosisResult(status, hypothesis, (evaluation,), candidates, (), (), 3, 10)


def _action() -> RemediationAction:
    return RemediationAction(
        "root",
        "database_slowdown",
        RemediationActionType.FAILOVER,
        "test",
        frozenset({"root"}),
        True,
        True,
    )


def _receipt(status: ExecutionStatus = ExecutionStatus.APPLIED) -> ExecutionReceipt:
    return ExecutionReceipt("root", RemediationActionType.FAILOVER, status, START, "test")


def _verify(
    metrics: list[MetricEvent],
    *,
    diagnosis: DiagnosisResult | None = None,
    budget: int = 4,
    receipt: ExecutionReceipt | None = None,
):
    baseline = _baseline({"root", "upstream", "healthy", "decoy"})
    api = TelemetryQueryAPI(TelemetryStore(metrics, (), ()), budget)
    window = FreshObservationWindow(
        START + timedelta(minutes=1),
        START + timedelta(minutes=2),
        5,
    )
    result = RecoveryVerifier().verify(
        api,
        baseline,
        diagnosis or _diagnosis(),
        _action(),
        receipt or _receipt(),
        window,
    )
    return result, api


def test_verification_is_inconclusive_when_budget_cannot_cover_checked_services() -> None:
    metrics = sum((_metric_batch(service) for service in ("root", "upstream", "healthy")), [])

    result, api = _verify(metrics, budget=1)

    assert result.status is RecoveryStatus.INCONCLUSIVE
    assert api.spent_budget == 1


@pytest.mark.parametrize("count", [0, 1])
def test_missing_or_too_few_required_root_samples_is_inconclusive(count: int) -> None:
    root = [
        event
        for event in _metric_batch("root", count=max(count, 1))
        if event.metric_name != "request_latency_ms" or count > 0
    ]
    if count == 0:
        root = [event for event in root if event.metric_name != "request_latency_ms"]
    metrics = root + _metric_batch("upstream") + _metric_batch("healthy")

    result, _ = _verify(metrics)

    assert result.status is RecoveryStatus.INCONCLUSIVE


def test_persistent_required_direction_and_traffic_collapse_are_failed() -> None:
    common = _metric_batch("upstream") + _metric_batch("healthy")
    persistent, _ = _verify(_metric_batch("root", latency=100.0) + common)
    collapsed, _ = _verify(_metric_batch("root", request_rate=0.0) + common)

    assert persistent.status is RecoveryStatus.FAILED
    assert "remains HIGH" in persistent.reason
    assert collapsed.status is RecoveryStatus.FAILED
    assert "traffic collapsed" in collapsed.reason


def test_downstream_strong_is_inconclusive_but_new_sentinel_strong_is_failed() -> None:
    healthy_root = _metric_batch("root")
    diagnosis = _diagnosis(strong_services=("root", "upstream", "decoy"))
    downstream_strong, _ = _verify(
        healthy_root
        + _metric_batch("upstream", latency=100.0, cpu=0.9)
        + _metric_batch("healthy"),
        diagnosis=diagnosis,
    )
    sentinel_strong, _ = _verify(
        healthy_root
        + _metric_batch("upstream")
        + _metric_batch("healthy", latency=100.0, cpu=0.9),
        diagnosis=diagnosis,
    )

    assert downstream_strong.status is RecoveryStatus.INCONCLUSIVE
    assert sentinel_strong.status is RecoveryStatus.FAILED


def test_preexisting_strong_decoy_is_not_selected_as_sentinel() -> None:
    diagnosis = _diagnosis(strong_services=("root", "upstream", "decoy"))
    metrics = sum((_metric_batch(service) for service in ("root", "upstream", "healthy")), [])

    result, _ = _verify(metrics, diagnosis=diagnosis)

    assert result.status is RecoveryStatus.VERIFIED
    assert result.sentinel_service == "healthy"


def test_rejected_execution_is_failed_without_telemetry_query() -> None:
    result, api = _verify([], receipt=_receipt(ExecutionStatus.REJECTED))

    assert result.status is RecoveryStatus.FAILED
    assert result.queries_executed == ()
    assert api.spent_budget == 0


def _analysis_start(incident) -> datetime:
    return max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)


@pytest.mark.parametrize(
    ("service", "failure_mode", "action", "steps", "maximum_budget"),
    [
        ("postgres", "database_slowdown", RemediationActionType.FAILOVER, 8, 17),
        ("auth", "process_crash", RemediationActionType.RESTART, 100, 17),
        ("catalog", "cpu_saturation", RemediationActionType.SCALE_UP, 150, 17),
    ],
)
def test_correct_interventions_verify_from_fresh_session_telemetry(
    service: str,
    failure_mode: str,
    action: RemediationActionType,
    steps: int,
    maximum_budget: int,
) -> None:
    session = IncidentGenerator(benchmark_config()).start_session(
        17, FaultRequest(service, failure_mode)
    )
    incident = session.initial_incident
    initial_metric_count = session.telemetry.metric_count
    api = session.create_query_api(maximum_budget)
    controller = RecoveryController(
        config=RecoveryControllerConfig(steps, 1)
    )

    result = controller.run(
        api,
        incident.baseline,
        _analysis_start(incident),
        incident.observed_end_time,
        SimulatorRemediationExecutor(session.simulator),
        session,
    )

    assert result.diagnosis.status is DiagnosisStatus.RESOLVED
    assert result.diagnosis.best_hypothesis == Hypothesis(
        service, failure_mode, CandidateStrength.STRONG
    )
    assert result.planning.action is not None
    assert result.planning.action.action_type is action
    assert result.execution is not None
    assert result.execution.status is ExecutionStatus.APPLIED
    assert result.verification is not None
    assert result.verification.status is RecoveryStatus.VERIFIED
    assert session.telemetry.metric_count > initial_metric_count
    assert all(
        finding.observed_request_rate is not None
        and finding.observed_request_rate > 0
        for finding in result.verification.traffic_findings
    )
    assert api.spent_budget <= maximum_budget
    assert result.verification.budget_spent == len(result.verification.checked_services)
    assert tuple(item.to_phase for item in result.state_history) == (
        IncidentPhase.DETECTED,
        IncidentPhase.CANDIDATE,
        IncidentPhase.RESOLVED,
        IncidentPhase.ACTION_ELIGIBLE,
        IncidentPhase.APPLIED,
        IncidentPhase.VERIFIED,
    )


def test_wrong_applied_intervention_fails_adds_exact_contradiction_and_reinvestigates() -> None:
    session = IncidentGenerator(benchmark_config()).start_session(
        17, FaultRequest("auth", "process_crash")
    )
    incident = session.initial_incident
    api = session.create_query_api(60)
    agent = DiagnosisAgent()
    diagnosis = agent.diagnose(
        api, incident.baseline, _analysis_start(incident), incident.observed_end_time
    )
    wrong = next(
        evaluation
        for evaluation in diagnosis.ranked_hypotheses
        if evaluation.hypothesis.service == "auth"
        and evaluation.hypothesis.failure_mode == "database_slowdown"
    )
    direct_support = CausalEvidence(
        EvidenceCategory.TRACE_LOCALIZATION,
        "auth",
        "database_slowdown",
        "deliberate test hypothesis",
        EvidenceStatus.SUPPORT,
        "Direct evidence supplied by the test scenario.",
    )
    wrong = replace(wrong, evidence=(*wrong.evidence, direct_support))
    attempted = replace(
        diagnosis,
        status=DiagnosisStatus.RESOLVED,
        best_hypothesis=wrong.hypothesis,
        ranked_hypotheses=(
            wrong,
            *(item for item in diagnosis.ranked_hypotheses if item.hypothesis != wrong.hypothesis),
        ),
    )

    result = RecoveryController(diagnosis_agent=agent).run(
        api,
        incident.baseline,
        _analysis_start(incident),
        incident.observed_end_time,
        SimulatorRemediationExecutor(session.simulator),
        session,
        attempted,
    )

    assert result.execution is not None
    assert result.execution.status is ExecutionStatus.APPLIED
    assert "mismatch" not in result.execution.message.lower()
    assert "process_crash" not in result.execution.message
    assert result.verification is not None
    assert result.verification.status is RecoveryStatus.FAILED
    assert result.follow_up_diagnosis is not None
    assert result.follow_up_diagnosis.best_hypothesis == Hypothesis(
        "auth", "process_crash", CandidateStrength.STRONG
    )
    attempted_follow_up = next(
        item
        for item in result.follow_up_diagnosis.ranked_hypotheses
        if item.hypothesis.service == "auth"
        and item.hypothesis.failure_mode == "database_slowdown"
    )
    intervention = [
        evidence
        for evidence in attempted_follow_up.evidence
        if evidence.category is EvidenceCategory.INTERVENTION_OUTCOME
    ]
    assert len(intervention) == 1
    assert intervention[0].status is EvidenceStatus.CONTRADICTION
    assert all(
        not any(
            evidence.category is EvidenceCategory.INTERVENTION_OUTCOME
            for evidence in item.evidence
        )
        for item in result.follow_up_diagnosis.ranked_hypotheses
        if item.hypothesis != attempted_follow_up.hypothesis
    )


def test_blocked_plan_runs_neither_execution_nor_verification() -> None:
    diagnosis = replace(_diagnosis(), status=DiagnosisStatus.AMBIGUOUS)
    baseline = _baseline({"root", "upstream", "healthy", "decoy"})
    api = TelemetryQueryAPI(TelemetryStore((), (), ()), 5)

    class NeverExecutor:
        def execute(self, action: RemediationAction) -> ExecutionReceipt:
            raise AssertionError("blocked action must not execute")

    class NeverObserver:
        def observe(self, api: TelemetryQueryAPI, step_count: int) -> FreshObservationWindow:
            raise AssertionError("blocked action must not be observed")

    result = RecoveryController().run(
        api,
        baseline,
        START,
        START + timedelta(minutes=1),
        NeverExecutor(),
        NeverObserver(),
        diagnosis,
    )

    assert result.planning.status is PlanningStatus.BLOCKED
    assert result.execution is None
    assert result.verification is None
    assert result.intervention_feedback == {}
    assert result.state_history[-1].to_phase is IncidentPhase.ACTION_BLOCKED


@pytest.mark.parametrize(
    "history",
    (
        (
            IncidentTransition(None, IncidentPhase.DETECTED, "start"),
            IncidentTransition(
                IncidentPhase.DETECTED, IncidentPhase.NO_CANDIDATES, "healthy"
            ),
            IncidentTransition(
                IncidentPhase.NO_CANDIDATES, IncidentPhase.APPLIED, "impossible"
            ),
        ),
        (
            IncidentTransition(None, IncidentPhase.DETECTED, "start"),
            IncidentTransition(
                IncidentPhase.DETECTED, IncidentPhase.CANDIDATE, "candidate"
            ),
            IncidentTransition(
                IncidentPhase.CANDIDATE, IncidentPhase.UNSUPPORTED, "unknown"
            ),
            IncidentTransition(
                IncidentPhase.UNSUPPORTED, IncidentPhase.APPLIED, "impossible"
            ),
        ),
    ),
)
def test_impossible_incident_state_transitions_are_rejected(
    history: tuple[IncidentTransition, ...],
) -> None:
    with pytest.raises(ValueError, match="illegal incident transition"):
        validate_incident_transitions(history)


def test_inconclusive_verification_adds_no_feedback_or_second_execution() -> None:
    diagnosis = _diagnosis()
    baseline = _baseline({"root", "upstream", "healthy", "decoy"})
    metrics = sum((_metric_batch(service) for service in ("root", "upstream", "decoy")), [])
    api = TelemetryQueryAPI(TelemetryStore(metrics, (), ()), 1)
    window = FreshObservationWindow(
        START + timedelta(minutes=1),
        START + timedelta(minutes=2),
        5,
    )

    class CountingExecutor:
        calls = 0

        def execute(self, action: RemediationAction) -> ExecutionReceipt:
            self.calls += 1
            return _receipt()

    class StaticObserver:
        def observe(self, api: TelemetryQueryAPI, step_count: int) -> FreshObservationWindow:
            return window

    executor = CountingExecutor()
    result = RecoveryController(
        config=RecoveryControllerConfig(5, 1)
    ).run(
        api,
        baseline,
        START,
        window.end_time,
        executor,
        StaticObserver(),
        diagnosis,
    )

    assert executor.calls == 1
    assert result.verification is not None
    assert result.verification.status is RecoveryStatus.INCONCLUSIVE
    assert result.intervention_feedback == {}
    assert result.follow_up_diagnosis is None


def test_controller_uses_a_bounded_second_window_without_reexecuting() -> None:
    diagnosis = _diagnosis()
    baseline = _baseline({"root", "upstream", "healthy", "decoy"})
    api = TelemetryQueryAPI(TelemetryStore((), (), ()), 6)

    class CountingExecutor:
        calls = 0

        def execute(self, action: RemediationAction) -> ExecutionReceipt:
            self.calls += 1
            return _receipt()

    class SettlingObserver:
        calls = 0

        def observe(self, api: TelemetryQueryAPI, step_count: int) -> FreshObservationWindow:
            self.calls += 1
            offset = timedelta(minutes=self.calls)
            source = (
                _metric_batch("root")
                + _metric_batch(
                    "upstream",
                    latency=100.0 if self.calls == 1 else 10.0,
                    cpu=0.9 if self.calls == 1 else 0.2,
                )
                + _metric_batch("decoy")
            )
            shifted = [
                replace(
                    event,
                    event_timestamp=event.event_timestamp + offset,
                    arrival_timestamp=event.arrival_timestamp + offset,
                )
                for event in source
            ]
            api.ingest(metrics=shifted)
            return FreshObservationWindow(
                START + timedelta(minutes=1) + offset,
                START + timedelta(minutes=2) + offset,
                step_count,
            )

    executor = CountingExecutor()
    observer = SettlingObserver()
    result = RecoveryController(
        config=RecoveryControllerConfig(5, 2)
    ).run(
        api,
        baseline,
        START,
        START + timedelta(minutes=1),
        executor,
        observer,
        diagnosis,
    )

    assert executor.calls == 1
    assert observer.calls == 2
    assert len(result.observed_windows) == 2
    assert result.verification is not None
    assert result.verification.status is RecoveryStatus.VERIFIED


def test_architecture_boundaries_exclude_simulation_remediation_and_ground_truth() -> None:
    root = Path(__file__).parents[1] / "src"
    rca_text = "\n".join(path.read_text() for path in (root / "rca").glob("*.py"))
    remediation_text = "\n".join(
        path.read_text() for path in (root / "remediation").glob("*.py")
    )

    assert "src.simulation" not in rca_text
    assert "src.remediation" not in rca_text
    assert "ground_truth" not in rca_text
    assert "src.simulation" not in remediation_text
    assert "GroundTruth" not in remediation_text
    assert "except Exception" not in rca_text + remediation_text
