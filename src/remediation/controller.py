"""Generic diagnose-plan-execute-observe-verify coordination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from src.core.baseline import BaselineStore
from src.core.telemetry import TelemetryQueryAPI
from src.rca.agent import DiagnosisAgent, DiagnosisResult

from .domain import (
    ExecutionReceipt,
    ExecutionStatus,
    FreshObservationWindow,
    PlanningOutcome,
    PlanningStatus,
    PostActionObserver,
    RemediationExecutor,
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_feedback",
            MappingProxyType(dict(self.intervention_feedback)),
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
        if planning.status is PlanningStatus.BLOCKED:
            return RecoveryRunResult(
                diagnosis, planning, None, None, (), {}, None
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
                diagnosis, planning, receipt, verification, (), {}, None
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
        )
