from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core import BaselineStore, MetricEvent
from src.rca import (
    CausalEvidence,
    CandidateStrength,
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
    FAILURE_MODE_ACTIONS,
    PlanningStatus,
    RemediationAction,
    RemediationActionType,
    SafeRemediationPlanner,
    execute_if_planned,
)
from src.simulation import ServiceConfig, SimulationConfig, SimulatorRemediationExecutor
from src.simulation.faults import resolve_parameters
from src.simulation.simulator import MicroserviceSimulator, SimulationOutput


START = datetime(2025, 3, 1, tzinfo=timezone.utc)


def _baseline(services: set[str] | None = None) -> BaselineStore:
    known_services = services or {"worker"}
    metrics = [
        MetricEvent(
            START + timedelta(seconds=index),
            START + timedelta(seconds=index),
            service,
            "request_latency_ms",
            10.0,
            "ms",
        )
        for service in known_services
        for index in range(4)
    ]
    return BaselineStore.from_metrics(known_services, metrics)


def _diagnosis(
    *,
    status: DiagnosisStatus = DiagnosisStatus.RESOLVED,
    service: str = "worker",
    failure_mode: str = "process_crash",
    strength: CandidateStrength = CandidateStrength.STRONG,
    directly_confirmed: bool = True,
    direct_category: EvidenceCategory = EvidenceCategory.LOG_SEMANTIC,
    contradictory: bool = False,
) -> DiagnosisResult:
    hypothesis = Hypothesis(service, failure_mode, strength)
    evidence = []
    if directly_confirmed:
        evidence.append(
            CausalEvidence(
                direct_category,
                service,
                failure_mode,
                "matching semantic log",
                EvidenceStatus.SUPPORT,
                "The target has direct semantic confirmation.",
                START,
            )
        )
    if contradictory:
        evidence.append(
            CausalEvidence(
                EvidenceCategory.METRIC_PATTERN,
                service,
                failure_mode,
                "required metric remained normal",
                EvidenceStatus.CONTRADICTION,
                "A required metric contradicts the hypothesis.",
            )
        )
    evaluation = HypothesisEvaluation(
        hypothesis,
        tuple(evidence),
        (service,),
        (),
        RankingComponents(int(contradictory), 0, 1, 1, 1),
    )
    candidate = ServiceCandidate(service, strength, ("mad", "change_point"), 2, False, ())
    return DiagnosisResult(
        status,
        hypothesis,
        (evaluation,),
        (candidate,),
        (),
        (),
        3,
        14,
    )


def test_failure_modes_have_exact_declarative_remediation_actions() -> None:
    assert dict(FAILURE_MODE_ACTIONS) == {
        "cpu_saturation": RemediationActionType.SCALE_UP,
        "deployment_regression": RemediationActionType.ROLLBACK,
        "database_slowdown": RemediationActionType.FAILOVER,
        "connection_exhaustion": RemediationActionType.RECYCLE_CONNECTIONS,
        "network_latency": RemediationActionType.REROUTE,
        "process_crash": RemediationActionType.RESTART,
    }


@pytest.mark.parametrize(
    "direct_category",
    (EvidenceCategory.LOG_SEMANTIC, EvidenceCategory.TRACE_LOCALIZATION),
)
def test_resolved_strong_directly_confirmed_diagnosis_is_planned(
    direct_category: EvidenceCategory,
) -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(direct_category=direct_category),
        _baseline(),
    )

    assert outcome.status is PlanningStatus.PLANNED
    assert outcome.action is not None
    assert outcome.action.target_service == "worker"
    assert outcome.action.action_type is RemediationActionType.RESTART
    assert outcome.action.blast_radius_services == {"worker"}
    assert outcome.action.safely_bounded


@pytest.mark.parametrize(
    "status",
    (
        DiagnosisStatus.AMBIGUOUS,
        DiagnosisStatus.INCOMPLETE_BUDGET,
        DiagnosisStatus.NO_CANDIDATES,
    ),
)
def test_non_resolved_diagnosis_is_blocked(status: DiagnosisStatus) -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(status=status),
        _baseline(),
    )

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


@pytest.mark.parametrize(
    ("strength", "directly_confirmed"),
    (
        (CandidateStrength.WEAK, True),
        (CandidateStrength.STRONG, False),
    ),
)
def test_weak_or_unconfirmed_hypothesis_is_blocked(
    strength: CandidateStrength,
    directly_confirmed: bool,
) -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(
            strength=strength,
            directly_confirmed=directly_confirmed,
        ),
        _baseline(),
    )

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


def test_unknown_target_service_is_blocked() -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(service="unknown"),
        _baseline({"worker"}),
    )

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


def test_resolved_diagnosis_without_best_hypothesis_is_blocked() -> None:
    diagnosis = replace(_diagnosis(), best_hypothesis=None)

    outcome = SafeRemediationPlanner().plan(diagnosis, _baseline())

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


def test_failure_mode_without_allowlisted_action_is_blocked() -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(failure_mode="unknown_failure"),
        _baseline(),
    )

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


def test_contradictory_hypothesis_is_blocked() -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(contradictory=True),
        _baseline(),
    )

    assert outcome.status is PlanningStatus.BLOCKED
    assert outcome.action is None


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[RemediationAction] = []

    def execute(self, action: RemediationAction) -> ExecutionReceipt:
        self.calls.append(action)
        return ExecutionReceipt(
            action.target_service,
            action.action_type,
            ExecutionStatus.APPLIED,
            START,
            "applied",
        )


def test_executor_is_not_called_for_blocked_plan() -> None:
    outcome = SafeRemediationPlanner().plan(
        _diagnosis(status=DiagnosisStatus.AMBIGUOUS),
        _baseline(),
    )
    executor = _RecordingExecutor()

    receipt = execute_if_planned(outcome, executor)

    assert receipt is None
    assert executor.calls == []


def _simulation_config() -> SimulationConfig:
    return SimulationConfig(
        services=(
            ServiceConfig("target", 100, 10, external_request_rate=80),
            ServiceConfig("healthy", 100, 8, external_request_rate=40),
        ),
        baseline_steps=2,
        incident_steps=6,
        step_seconds=30,
    )


def _start_fault(
    failure_mode: str,
    *,
    seed: int = 19,
) -> MicroserviceSimulator:
    simulator = MicroserviceSimulator(_simulation_config(), seed)
    simulator.start_run(
        START,
        2,
        "target",
        failure_mode,
        resolve_parameters(failure_mode, {}),
    )
    return simulator


def _action(
    action_type: RemediationActionType,
    *,
    target: str = "target",
    diagnosed_failure_mode: str = "test_mode",
) -> RemediationAction:
    return RemediationAction(
        target,
        diagnosed_failure_mode,
        action_type,
        "test action",
        frozenset({target}),
        True,
        True,
    )


def _combined(left: SimulationOutput, right: SimulationOutput) -> SimulationOutput:
    return SimulationOutput(
        left.metrics + right.metrics,
        left.logs + right.logs,
        left.spans + right.spans,
        left.state_history + right.state_history,
    )


def test_run_matches_split_continuation_on_same_state_and_rng_trajectory() -> None:
    config = _simulation_config()
    parameters = resolve_parameters("database_slowdown", {})
    one_shot = MicroserviceSimulator(config, 23).run(
        START,
        2,
        "target",
        "database_slowdown",
        parameters,
    )
    continued = MicroserviceSimulator(config, 23)
    continued.start_run(START, 2, "target", "database_slowdown", parameters)

    first = continued.continue_run(3)
    second = continued.continue_run(5)

    assert _combined(first, second) == one_shot


def test_scale_up_removes_cpu_fault_without_resetting_queue_or_traffic() -> None:
    simulator = _start_fault("cpu_saturation")
    before = simulator.continue_run(6).state_history[-1]["target"]

    receipt = SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.SCALE_UP)
    )
    after = simulator.continue_run(3).state_history

    assert receipt.status is ExecutionStatus.APPLIED
    assert before.capacity == pytest.approx(35)
    assert before.cpu_utilization >= 0.96
    assert after[0]["target"].capacity == pytest.approx(100)
    assert after[0]["target"].cpu_utilization < 0.96
    assert 0 < after[0]["target"].queue_length < before.queue_length
    assert after[0]["target"].request_rate > 0


def test_restart_restores_crashed_process_without_resetting_queue_or_traffic() -> None:
    simulator = _start_fault("process_crash")
    before = simulator.continue_run(5).state_history[-1]["target"]

    receipt = SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.RESTART)
    )
    after = simulator.continue_run(1).state_history[0]["target"]

    assert receipt.status is ExecutionStatus.APPLIED
    assert not before.available
    assert before.processed_rate == 0
    assert after.available
    assert after.processed_rate > 0
    assert after.error_rate < before.error_rate
    assert after.queue_length > 0
    assert after.request_rate > 0


def test_rollback_removes_deployment_regression_effects() -> None:
    simulator = _start_fault("deployment_regression")
    before = simulator.continue_run(3).state_history[-1]["target"]

    SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.ROLLBACK)
    )
    after = simulator.continue_run(1).state_history[0]["target"]

    assert before.latency_ms > after.latency_ms
    assert before.error_rate > after.error_rate
    assert after.request_rate > 0


@pytest.mark.parametrize(
    ("failure_mode", "action_type"),
    (
        ("database_slowdown", RemediationActionType.FAILOVER),
        ("connection_exhaustion", RemediationActionType.RECYCLE_CONNECTIONS),
        ("network_latency", RemediationActionType.REROUTE),
    ),
)
def test_remaining_actions_remove_corresponding_fault_effects(
    failure_mode: str,
    action_type: RemediationActionType,
) -> None:
    simulator = _start_fault(failure_mode)
    before = simulator.continue_run(3).state_history[-1]["target"]

    receipt = SimulatorRemediationExecutor(simulator).execute(_action(action_type))
    after = simulator.continue_run(1).state_history[0]["target"]

    assert receipt.status is ExecutionStatus.APPLIED
    if failure_mode == "database_slowdown":
        assert after.latency_ms < before.latency_ms
    elif failure_mode == "connection_exhaustion":
        assert after.capacity > before.capacity
        assert after.error_rate < before.error_rate
    else:
        assert after.latency_ms < before.latency_ms
    assert after.request_rate > 0


def test_wrong_target_does_not_clear_actual_fault() -> None:
    simulator = _start_fault("database_slowdown")
    before = simulator.continue_run(3).state_history[-1]["target"]

    receipt = SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.FAILOVER, target="healthy")
    )
    after = simulator.continue_run(1).state_history[0]["target"]

    assert receipt.status is ExecutionStatus.APPLIED
    assert after.latency_ms == pytest.approx(before.latency_ms)
    assert "mismatch" not in receipt.message.lower()


def test_incompatible_action_does_not_clear_actual_fault() -> None:
    simulator = _start_fault("database_slowdown")
    before = simulator.continue_run(3).state_history[-1]["target"]

    receipt = SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.RESTART)
    )
    after = simulator.continue_run(1).state_history[0]["target"]

    assert receipt.status is ExecutionStatus.APPLIED
    assert after.latency_ms == pytest.approx(before.latency_ms)
    assert "mismatch" not in receipt.message.lower()


def test_action_does_not_directly_mutate_unrelated_service_or_configuration() -> None:
    simulator = _start_fault("database_slowdown")
    configuration = simulator.config
    healthy_before = simulator.continue_run(3).state_history[-1]["healthy"]

    SimulatorRemediationExecutor(simulator).execute(
        _action(RemediationActionType.FAILOVER)
    )
    states = simulator.continue_run(1).state_history[0]

    assert simulator.config is configuration
    assert states["target"].latency_ms < 70
    assert states["healthy"].capacity == healthy_before.capacity == 100
    assert states["healthy"].available and healthy_before.available
    assert states["healthy"].latency_ms == healthy_before.latency_ms == 8


def test_remediation_package_has_no_simulation_or_ground_truth_dependency() -> None:
    remediation_root = Path(__file__).parents[1] / "src" / "remediation"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in remediation_root.glob("*.py")
    ).lower()

    assert "src.simulation" not in source
    assert "ground_truth" not in source
