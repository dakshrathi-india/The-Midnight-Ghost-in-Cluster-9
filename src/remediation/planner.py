"""Conservative remediation policy and diagnosis safety gates."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from src.core.baseline import BaselineStore
from src.rca.agent import DiagnosisResult, DiagnosisStatus
from src.rca.candidates import CandidateStrength
from src.rca.diagnosis import EvidenceCategory, EvidenceStatus

from .domain import (
    PlanningOutcome,
    PlanningStatus,
    RemediationAction,
    RemediationActionType,
)


FAILURE_MODE_ACTIONS: Mapping[str, RemediationActionType] = MappingProxyType(
    {
        "cpu_saturation": RemediationActionType.SCALE_UP,
        "deployment_regression": RemediationActionType.ROLLBACK,
        "database_slowdown": RemediationActionType.FAILOVER,
        "connection_exhaustion": RemediationActionType.RECYCLE_CONNECTIONS,
        "network_latency": RemediationActionType.REROUTE,
        "process_crash": RemediationActionType.RESTART,
    }
)

_REVERSIBLE_ACTIONS = frozenset(
    {
        RemediationActionType.SCALE_UP,
        RemediationActionType.ROLLBACK,
        RemediationActionType.FAILOVER,
        RemediationActionType.REROUTE,
    }
)


class SafeRemediationPlanner:
    def plan(
        self,
        diagnosis: DiagnosisResult,
        baseline: BaselineStore,
    ) -> PlanningOutcome:
        if diagnosis.status is not DiagnosisStatus.RESOLVED:
            return self._blocked(
                f"diagnosis status {diagnosis.status.value} is not RESOLVED"
            )
        hypothesis = diagnosis.best_hypothesis
        if hypothesis is None:
            return self._blocked("resolved diagnosis has no best hypothesis")
        if hypothesis.service not in baseline.known_services:
            return self._blocked(
                f"diagnosed target {hypothesis.service!r} is not a known service"
            )

        candidate = next(
            (
                item
                for item in diagnosis.candidates
                if item.service == hypothesis.service
            ),
            None,
        )
        if (
            hypothesis.candidate_strength is not CandidateStrength.STRONG
            or candidate is None
            or candidate.strength is not CandidateStrength.STRONG
        ):
            return self._blocked("diagnosed target is not a STRONG candidate")

        evaluation = next(
            (
                item
                for item in diagnosis.ranked_hypotheses
                if item.hypothesis == hypothesis
            ),
            None,
        )
        if evaluation is None:
            return self._blocked("best hypothesis has no causal evaluation")
        if evaluation.ranking.contradiction_count != 0 or evaluation.contradictions:
            return self._blocked("best hypothesis contains contradictory evidence")

        directly_confirmed = any(
            evidence.service == hypothesis.service
            and evidence.failure_mode == hypothesis.failure_mode
            and evidence.status is EvidenceStatus.SUPPORT
            and evidence.category
            in {
                EvidenceCategory.LOG_SEMANTIC,
                EvidenceCategory.TRACE_LOCALIZATION,
            }
            for evidence in evaluation.evidence
        )
        if not directly_confirmed:
            return self._blocked(
                "best hypothesis lacks direct root log or trace confirmation"
            )

        action_type = FAILURE_MODE_ACTIONS.get(hypothesis.failure_mode)
        if action_type is None:
            return self._blocked(
                f"failure mode {hypothesis.failure_mode!r} has no allowlisted action"
            )

        action = RemediationAction(
            target_service=hypothesis.service,
            diagnosed_failure_mode=hypothesis.failure_mode,
            action_type=action_type,
            reason=(
                f"Resolved {hypothesis.failure_mode} on {hypothesis.service} has "
                "zero contradictions and direct confirming root evidence."
            ),
            blast_radius_services=frozenset({hypothesis.service}),
            reversible=action_type in _REVERSIBLE_ACTIONS,
            safely_bounded=True,
        )
        if action.blast_radius_services != frozenset({hypothesis.service}):
            return self._blocked("remediation blast radius is not exactly the target")
        return PlanningOutcome(
            PlanningStatus.PLANNED,
            f"{action_type.value} is allowlisted for {hypothesis.failure_mode}",
            action,
        )

    @staticmethod
    def _blocked(reason: str) -> PlanningOutcome:
        return PlanningOutcome(PlanningStatus.BLOCKED, reason)
