"""Stateful remediation execution for the synthetic simulator backend."""

from __future__ import annotations

from src.remediation import (
    ExecutionReceipt,
    ExecutionStatus,
    FAILURE_MODE_ACTIONS,
    RemediationAction,
    RemediationExecutor,
)

from .simulator import MicroserviceSimulator


_FAILURE_MODE_BY_ACTION = {
    action_type: failure_mode
    for failure_mode, action_type in FAILURE_MODE_ACTIONS.items()
}


class SimulatorRemediationExecutor(RemediationExecutor):
    def __init__(self, simulator: MicroserviceSimulator) -> None:
        self._simulator = simulator

    def execute(self, action: RemediationAction) -> ExecutionReceipt:
        if action.target_service not in self._simulator.services:
            return ExecutionReceipt(
                action.target_service,
                action.action_type,
                ExecutionStatus.REJECTED,
                self._simulator.last_step_time,
                f"Service {action.target_service!r} is not managed by this simulator.",
            )

        compatible_failure_mode = _FAILURE_MODE_BY_ACTION[action.action_type]
        self._simulator.mitigate_fault(
            action.target_service,
            compatible_failure_mode,
        )
        return ExecutionReceipt(
            action.target_service,
            action.action_type,
            ExecutionStatus.APPLIED,
            self._simulator.last_step_time,
            f"{action.action_type.value} command applied to {action.target_service}.",
        )
