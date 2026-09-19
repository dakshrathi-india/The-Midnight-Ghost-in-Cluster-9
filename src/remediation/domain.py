"""Simulator-independent remediation domain objects and execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from src.core.telemetry import TelemetryQueryAPI


class RemediationActionType(str, Enum):
    SCALE_UP = "SCALE_UP"
    ROLLBACK = "ROLLBACK"
    FAILOVER = "FAILOVER"
    RECYCLE_CONNECTIONS = "RECYCLE_CONNECTIONS"
    REROUTE = "REROUTE"
    RESTART = "RESTART"


class PlanningStatus(str, Enum):
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"


class ExecutionStatus(str, Enum):
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class IncidentPhase(str, Enum):
    DETECTED = "DETECTED"
    CANDIDATE = "CANDIDATE"
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE_BUDGET = "INCOMPLETE_BUDGET"
    NO_CANDIDATES = "NO_CANDIDATES"
    ACTION_ELIGIBLE = "ACTION_ELIGIBLE"
    ACTION_BLOCKED = "ACTION_BLOCKED"
    APPLIED = "APPLIED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VERIFIED = "VERIFIED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class IncidentTransition:
    from_phase: IncidentPhase | None
    to_phase: IncidentPhase
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("incident transition reason must be non-empty")


_LEGAL_INCIDENT_TRANSITIONS = frozenset(
    {
        (None, IncidentPhase.DETECTED),
        (IncidentPhase.DETECTED, IncidentPhase.CANDIDATE),
        (IncidentPhase.DETECTED, IncidentPhase.NO_CANDIDATES),
        (IncidentPhase.DETECTED, IncidentPhase.INCOMPLETE_BUDGET),
        (IncidentPhase.CANDIDATE, IncidentPhase.RESOLVED),
        (IncidentPhase.CANDIDATE, IncidentPhase.AMBIGUOUS),
        (IncidentPhase.CANDIDATE, IncidentPhase.UNSUPPORTED),
        (IncidentPhase.CANDIDATE, IncidentPhase.INCOMPLETE_BUDGET),
        (IncidentPhase.RESOLVED, IncidentPhase.ACTION_ELIGIBLE),
        (IncidentPhase.RESOLVED, IncidentPhase.ACTION_BLOCKED),
        (IncidentPhase.AMBIGUOUS, IncidentPhase.ACTION_BLOCKED),
        (IncidentPhase.UNSUPPORTED, IncidentPhase.ACTION_BLOCKED),
        (IncidentPhase.INCOMPLETE_BUDGET, IncidentPhase.ACTION_BLOCKED),
        (IncidentPhase.NO_CANDIDATES, IncidentPhase.ACTION_BLOCKED),
        (IncidentPhase.ACTION_ELIGIBLE, IncidentPhase.APPLIED),
        (IncidentPhase.ACTION_ELIGIBLE, IncidentPhase.EXECUTION_FAILED),
        (IncidentPhase.APPLIED, IncidentPhase.VERIFIED),
        (IncidentPhase.APPLIED, IncidentPhase.RECOVERY_FAILED),
        (IncidentPhase.APPLIED, IncidentPhase.INCONCLUSIVE),
        (IncidentPhase.EXECUTION_FAILED, IncidentPhase.RECOVERY_FAILED),
        (IncidentPhase.EXECUTION_FAILED, IncidentPhase.INCONCLUSIVE),
        (IncidentPhase.RECOVERY_FAILED, IncidentPhase.DETECTED),
    }
)


def validate_incident_transitions(
    transitions: tuple[IncidentTransition, ...] | list[IncidentTransition],
) -> tuple[IncidentTransition, ...]:
    history = tuple(transitions)
    previous: IncidentPhase | None = None
    for transition in history:
        if transition.from_phase is not previous:
            raise ValueError("incident state history is not contiguous")
        if (transition.from_phase, transition.to_phase) not in _LEGAL_INCIDENT_TRANSITIONS:
            raise ValueError(
                "illegal incident transition: "
                f"{transition.from_phase} -> {transition.to_phase}"
            )
        previous = transition.to_phase
    return history


@dataclass(frozen=True, slots=True)
class RemediationAction:
    target_service: str
    diagnosed_failure_mode: str
    action_type: RemediationActionType
    reason: str
    blast_radius_services: frozenset[str]
    reversible: bool
    safely_bounded: bool


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    status: PlanningStatus
    reason: str
    action: RemediationAction | None = None

    def __post_init__(self) -> None:
        if self.status is PlanningStatus.PLANNED and self.action is None:
            raise ValueError("a planned outcome requires an action")
        if self.status is PlanningStatus.BLOCKED and self.action is not None:
            raise ValueError("a blocked outcome cannot contain an action")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    target_service: str
    requested_action: RemediationActionType
    status: ExecutionStatus
    executed_at: datetime | None
    message: str


@dataclass(frozen=True, slots=True)
class FreshObservationWindow:
    start_time: datetime
    end_time: datetime
    step_count: int

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("observation window end must not precede its start")
        if self.step_count < 1:
            raise ValueError("observation window must contain at least one step")


class RemediationExecutor(Protocol):
    def execute(self, action: RemediationAction) -> ExecutionReceipt: ...


class PostActionObserver(Protocol):
    def observe(
        self, api: TelemetryQueryAPI, step_count: int
    ) -> FreshObservationWindow: ...


def execute_if_planned(
    outcome: PlanningOutcome,
    executor: RemediationExecutor,
) -> ExecutionReceipt | None:
    """Execute only an action that passed the safety planner."""
    if outcome.status is PlanningStatus.BLOCKED:
        return None
    assert outcome.action is not None
    return executor.execute(outcome.action)
