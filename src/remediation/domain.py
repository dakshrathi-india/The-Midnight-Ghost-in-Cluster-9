"""Simulator-independent remediation domain objects and execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol


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


class RemediationExecutor(Protocol):
    def execute(self, action: RemediationAction) -> ExecutionReceipt: ...


def execute_if_planned(
    outcome: PlanningOutcome,
    executor: RemediationExecutor,
) -> ExecutionReceipt | None:
    """Execute only an action that passed the safety planner."""
    if outcome.status is PlanningStatus.BLOCKED:
        return None
    assert outcome.action is not None
    return executor.execute(outcome.action)
