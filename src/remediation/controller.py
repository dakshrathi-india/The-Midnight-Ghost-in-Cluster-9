"""Generic diagnose-plan-execute-observe-verify coordination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from src.core.baseline import BaselineStore
from src.core.telemetry import TelemetryQueryAPI
from src.rca.agent import DiagnosisAgent, DiagnosisResult, DiagnosisStatus
from src.rca.candidates import CandidateStrength

from .domain import (
    ExecutionReceipt,
    ExecutionStatus,
    FreshObservationWindow,
    IncidentPhase,
    IncidentTransition,
    PlanningOutcome,
    PlanningStatus,
    PostActionObserver,
    RemediationExecutor,
    validate_incident_transitions,
)
from .planner import SafeRemediationPlanner
from .recovery import RecoveryStatus, RecoveryVerification, RecoveryVerifier


@dataclass(frozen=True, slots=True)
class RecoveryControllerConfig:
    steps_per_observation_window: int = 8
    maximum_observation_windows: int = 2

    def __post_init__(self) -> None:
        if self.steps_per_observation_window < 1:
            raise ValueError("steps per observation window must be positive")
        if self.maximum_observation_windows < 1:
            raise ValueError("maximum observation windows must be positive")


@dataclass(frozen=True, slots=True)
class RecoveryRunResult:
    diagnosis: DiagnosisResult
    planning: PlanningOutcome
    execution: ExecutionReceipt | None
    verification: RecoveryVerification | None
    observed_windows: tuple[FreshObservationWindow, ...]
    intervention_feedback: Mapping[tuple[str, str], str]
    follow_up_diagnosis: DiagnosisResult | None
    state_history: tuple[IncidentTransition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_feedback",
            MappingProxyType(dict(self.intervention_feedback)),
        )
        object.__setattr__(
            self,
            "state_history",
            validate_incident_transitions(self.state_history),
        )


class RecoveryController:
    def __init__(
        self,
        diagnosis_agent: DiagnosisAgent | None = None,
        planner: SafeRemediationPlanner | None = None,
        verifier: RecoveryVerifier | None = None,
        config: RecoveryControllerConfig | None = None,
    ) -> None:
        self._diagnosis_agent = diagnosis_agent or DiagnosisAgent()
        self._planner = planner or SafeRemediationPlanner()
        self._verifier = verifier or RecoveryVerifier()
        self.config = config or RecoveryControllerConfig()

    def run(
        self,
        api: TelemetryQueryAPI,
        baseline: BaselineStore,
        analysis_start: datetime,
        analysis_end: datetime,
        executor: RemediationExecutor,
        observer: PostActionObserver,
        initial_diagnosis: DiagnosisResult | None = None,
    ) -> RecoveryRunResult:
        diagnosis = initial_diagnosis or self._diagnosis_agent.diagnose(
            api, baseline, analysis_start, analysis_end
        )
        planning = self._planner.plan(diagnosis, baseline)
        if (
            planning.status is PlanningStatus.BLOCKED
            and planning.reason
            == "insufficient orthogonal evidence for autonomous remediation"
        ):
            diagnosis = self._diagnosis_agent.confirm_for_action(
                api,
                baseline,
                analysis_start,
                analysis_end,
                diagnosis,
            )
            planning = self._planner.plan(diagnosis, baseline)
        if planning.status is PlanningStatus.BLOCKED:
            return RecoveryRunResult(
                diagnosis,
                planning,
                None,
                None,
                (),
                {},
                None,
                _build_state_history(diagnosis, planning),
            )

        assert planning.action is not None
        receipt = executor.execute(planning.action)
        if receipt.status is not ExecutionStatus.APPLIED:
            verification = self._verifier.verify(
                api,
                baseline,
                diagnosis,
                planning.action,
                receipt,
                None,
            )
            return RecoveryRunResult(
                diagnosis,
                planning,
                receipt,
                verification,
                (),
                {},
                None,
                _build_state_history(
                    diagnosis, planning, receipt, verification
                ),
            )

        windows: list[FreshObservationWindow] = []
        verification: RecoveryVerification | None = None
        for _ in range(self.config.maximum_observation_windows):
            window = observer.observe(
                api, self.config.steps_per_observation_window
            )
            windows.append(window)
            verification = self._verifier.verify(
                api,
                baseline,
                diagnosis,
                planning.action,
                receipt,
                window,
            )
            if verification.status is not RecoveryStatus.INCONCLUSIVE:
                break

        assert verification is not None
        feedback: dict[tuple[str, str], str] = {}
        follow_up: DiagnosisResult | None = None
        if verification.status is RecoveryStatus.FAILED:
            key = (
                planning.action.target_service,
                planning.action.diagnosed_failure_mode,
            )
            feedback[key] = (
                "The intervention was applied, but fresh post-action telemetry "
                f"decisively failed recovery verification: {verification.reason}"
            )
            follow_up = self._diagnosis_agent.diagnose(
                api,
                baseline,
                analysis_start,
                windows[-1].end_time,
                intervention_contradictions=feedback,
            )

        return RecoveryRunResult(
            diagnosis,
            planning,
            receipt,
            verification,
            tuple(windows),
            feedback,
            follow_up,
            _build_state_history(
                diagnosis,
                planning,
                receipt,
                verification,
                follow_up,
            ),
        )


def _build_state_history(
    diagnosis: DiagnosisResult,
    planning: PlanningOutcome,
    receipt: ExecutionReceipt | None = None,
    verification: RecoveryVerification | None = None,
    follow_up: DiagnosisResult | None = None,
) -> tuple[IncidentTransition, ...]:
    transitions: list[IncidentTransition] = []

    def advance(phase: IncidentPhase, reason: str) -> None:
        previous = transitions[-1].to_phase if transitions else None
        transitions.append(IncidentTransition(previous, phase, reason))

    advance(IncidentPhase.DETECTED, "incident investigation started")
    _append_diagnosis_phases(transitions, diagnosis)
    if planning.status is PlanningStatus.PLANNED:
        advance(IncidentPhase.ACTION_ELIGIBLE, planning.reason)
    else:
        advance(IncidentPhase.ACTION_BLOCKED, planning.reason)
        return validate_incident_transitions(transitions)

    if receipt is None:
        raise ValueError("planned remediation requires an execution receipt")
    if receipt.status is ExecutionStatus.APPLIED:
        advance(IncidentPhase.APPLIED, receipt.message)
    else:
        advance(IncidentPhase.EXECUTION_FAILED, receipt.message)

    if verification is None:
        advance(IncidentPhase.INCONCLUSIVE, "recovery was not verified")
    elif verification.status is RecoveryStatus.VERIFIED:
        advance(IncidentPhase.VERIFIED, verification.reason)
    elif verification.status is RecoveryStatus.FAILED:
        advance(IncidentPhase.RECOVERY_FAILED, verification.reason)
    else:
        advance(IncidentPhase.INCONCLUSIVE, verification.reason)

    if follow_up is not None:
        advance(IncidentPhase.DETECTED, "follow-up investigation started")
        _append_diagnosis_phases(transitions, follow_up)
    return validate_incident_transitions(transitions)


def _append_diagnosis_phases(
    transitions: list[IncidentTransition],
    diagnosis: DiagnosisResult,
) -> None:
    def advance(phase: IncidentPhase, reason: str) -> None:
        previous = transitions[-1].to_phase if transitions else None
        transitions.append(IncidentTransition(previous, phase, reason))

    has_candidate = any(
        candidate.strength is not CandidateStrength.NOT_CANDIDATE
        for candidate in diagnosis.candidates
    )
    if has_candidate:
        advance(IncidentPhase.CANDIDATE, "anomaly candidates were identified")
    phase_by_status = {
        DiagnosisStatus.RESOLVED: IncidentPhase.RESOLVED,
        DiagnosisStatus.AMBIGUOUS: IncidentPhase.AMBIGUOUS,
        DiagnosisStatus.UNSUPPORTED: IncidentPhase.UNSUPPORTED,
        DiagnosisStatus.INCOMPLETE_BUDGET: IncidentPhase.INCOMPLETE_BUDGET,
        DiagnosisStatus.NO_CANDIDATES: IncidentPhase.NO_CANDIDATES,
    }
    advance(
        phase_by_status[diagnosis.status],
        f"diagnosis status is {diagnosis.status.value}",
    )
