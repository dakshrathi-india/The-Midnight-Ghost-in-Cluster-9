"""Budgeted post-action recovery verification over fresh telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Mapping, Sequence

from src.core.baseline import BaselineStore
from src.core.budget import QueryHistoryEntry
from src.core.models import MetricEvent
from src.core.telemetry import TelemetryQueryAPI
from src.rca.agent import DiagnosisResult
from src.rca.anomaly import CUSUMDetector, IsolationForestDetector, MADDetector
from src.rca.candidates import CandidateStrength
from src.rca.diagnosis import HypothesisEvaluator, MetricPatternDirection
from src.rca.signatures import FailureSignatureLibrary, MetricDirection

from .domain import (
    ExecutionReceipt,
    ExecutionStatus,
    FreshObservationWindow,
    RemediationAction,
)


class RecoveryStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    minimum_samples_per_required_metric: int = 3
    tail_samples_per_metric: int = 5
    minimum_request_rate_ratio: float = 0.5
    require_healthy_sentinel: bool = True

    def __post_init__(self) -> None:
        if self.minimum_samples_per_required_metric < 1:
            raise ValueError("minimum required-metric samples must be positive")
        if self.tail_samples_per_metric < self.minimum_samples_per_required_metric:
            raise ValueError("tail sample count cannot be smaller than the minimum")
        if not 0.0 <= self.minimum_request_rate_ratio <= 1.0:
            raise ValueError("minimum request-rate ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MetricRecoveryFinding:
    service: str
    metric_name: str
    sample_count: int
    observed_direction: MetricPatternDirection
    expected_fault_direction: MetricDirection | None


@dataclass(frozen=True, slots=True)
class TrafficContinuityFinding:
    service: str
    baseline_request_rate: float
    observed_request_rate: float | None
    ratio_to_baseline: float | None
    continuous: bool | None


@dataclass(frozen=True, slots=True)
class ServiceRecoveryFinding:
    service: str
    detector_flags: tuple[str, ...]
    candidate_strength: CandidateStrength


@dataclass(frozen=True, slots=True)
class RecoveryVerification:
    status: RecoveryStatus
    target_service: str
    failure_mode: str
    checked_services: tuple[str, ...]
    fresh_window: FreshObservationWindow | None
    queries_executed: tuple[QueryHistoryEntry, ...]
    budget_spent: int
    root_metric_findings: tuple[MetricRecoveryFinding, ...]
    traffic_findings: tuple[TrafficContinuityFinding, ...]
    service_findings: tuple[ServiceRecoveryFinding, ...]
    sentinel_service: str | None
    reason: str


class RecoveryVerifier:
    def __init__(
        self,
        signatures: FailureSignatureLibrary | None = None,
        config: RecoveryConfig | None = None,
    ) -> None:
        self.signatures = signatures or FailureSignatureLibrary()
        self.config = config or RecoveryConfig()
        self._evaluator = HypothesisEvaluator(self.signatures)
        self._mad = MADDetector()
        self._cusum = CUSUMDetector()
        self._isolation = IsolationForestDetector()

    def verify(
        self,
        api: TelemetryQueryAPI,
        baseline: BaselineStore,
        diagnosis: DiagnosisResult,
        action: RemediationAction,
        receipt: ExecutionReceipt,
        fresh_window: FreshObservationWindow | None,
    ) -> RecoveryVerification:
        history_start = len(api.query_history)
        budget_start = api.spent_budget
        if receipt.status is not ExecutionStatus.APPLIED:
            return self._result(
                RecoveryStatus.FAILED,
                action,
                (),
                fresh_window,
                api,
                history_start,
                budget_start,
                reason=f"intervention execution {receipt.status.value.lower()}: {receipt.message}",
            )
        if fresh_window is None:
            return self._result(
                RecoveryStatus.INCONCLUSIVE,
                action,
                (),
                None,
                api,
                history_start,
                budget_start,
                reason="no fresh post-action observation window was supplied",
            )

        evaluation = next(
            (
                item
                for item in diagnosis.ranked_hypotheses
                if item.hypothesis.service == action.target_service
                and item.hypothesis.failure_mode == action.diagnosed_failure_mode
            ),
            None,
        )
        if evaluation is None:
            return self._result(
                RecoveryStatus.INCONCLUSIVE,
                action,
                (),
                fresh_window,
                api,
                history_start,
                budget_start,
                reason="the applied hypothesis is absent from the diagnosis evidence",
            )

        initial_strong = {
            candidate.service
            for candidate in diagnosis.candidates
            if candidate.strength is CandidateStrength.STRONG
        }
        explained = tuple(
            sorted(
                set(evaluation.explained_strong_candidates)
                - {action.target_service}
            )
        )
        sentinel = next(
            (
                service
                for service in sorted(baseline.known_services)
                if service != action.target_service
                and service not in explained
                and service not in initial_strong
            ),
            None,
        )
        checked = (action.target_service, *explained)
        if sentinel is not None:
            checked = (*checked, sentinel)
        elif self.config.require_healthy_sentinel:
            return self._result(
                RecoveryStatus.INCONCLUSIVE,
                action,
                checked,
                fresh_window,
                api,
                history_start,
                budget_start,
                sentinel=None,
                reason="no valid previously healthy sentinel service is available",
            )

        metrics_by_service: dict[str, tuple[MetricEvent, ...]] = {}
        for service in checked:
            if not api.can_afford(
                "metrics", service, fresh_window.start_time, fresh_window.end_time
            ):
                return self._result(
                    RecoveryStatus.INCONCLUSIVE,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    sentinel=sentinel,
                    reason=f"insufficient query budget for fresh {service} metrics",
                )
            metrics_by_service[service] = api.query_metrics(
                service, fresh_window.start_time, fresh_window.end_time
            )

        tail_by_service = {
            service: _tail_metrics(events, self.config.tail_samples_per_metric)
            for service, events in metrics_by_service.items()
        }
        signature = self.signatures.get(action.diagnosed_failure_mode)
        patterns = self._evaluator.metric_patterns(
            action.target_service,
            tail_by_service[action.target_service],
            baseline,
        )
        root_findings: list[MetricRecoveryFinding] = []
        for expectation in signature.metric_expectations:
            if not expectation.required:
                continue
            samples = [
                event
                for event in tail_by_service[action.target_service]
                if event.metric_name == expectation.metric_name
            ]
            pattern = patterns.get(expectation.metric_name)
            direction = (
                pattern.direction
                if pattern is not None
                else MetricPatternDirection.UNKNOWN
            )
            root_findings.append(
                MetricRecoveryFinding(
                    action.target_service,
                    expectation.metric_name,
                    len(samples),
                    direction,
                    expectation.direction,
                )
            )
            if (
                len(samples) < self.config.minimum_samples_per_required_metric
                or direction is MetricPatternDirection.UNKNOWN
            ):
                return self._result(
                    RecoveryStatus.INCONCLUSIVE,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    sentinel=sentinel,
                    reason=f"insufficient fresh {expectation.metric_name} evidence at the target",
                )
            expected = (
                MetricPatternDirection.HIGH
                if expectation.direction is MetricDirection.HIGH
                else MetricPatternDirection.LOW
            )
            if direction is expected:
                return self._result(
                    RecoveryStatus.FAILED,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    sentinel=sentinel,
                    reason=f"required target signal {expectation.metric_name} remains {direction.value}",
                )
            if direction is not MetricPatternDirection.NORMAL:
                return self._result(
                    RecoveryStatus.INCONCLUSIVE,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    sentinel=sentinel,
                    reason=f"target signal {expectation.metric_name} changed but did not normalize",
                )

        traffic_findings = self._traffic_findings(checked, tail_by_service, baseline)
        for finding in traffic_findings:
            if finding.continuous is None:
                return self._result(
                    RecoveryStatus.INCONCLUSIVE,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    traffic_findings,
                    sentinel=sentinel,
                    reason=f"insufficient request-rate evidence for {finding.service}",
                )
            if not finding.continuous:
                return self._result(
                    RecoveryStatus.FAILED,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    traffic_findings,
                    sentinel=sentinel,
                    reason=f"request traffic collapsed at {finding.service}",
                )

        service_findings = tuple(
            self._service_finding(service, tail_by_service[service], baseline)
            for service in checked
        )
        if sentinel is not None:
            sentinel_finding = next(
                finding for finding in service_findings if finding.service == sentinel
            )
            if sentinel_finding.candidate_strength is CandidateStrength.STRONG:
                return self._result(
                    RecoveryStatus.FAILED,
                    action,
                    checked,
                    fresh_window,
                    api,
                    history_start,
                    budget_start,
                    root_findings,
                    traffic_findings,
                    service_findings,
                    sentinel,
                    f"healthy sentinel {sentinel} became a strong anomaly candidate",
                )

        residual = [
            finding.service
            for finding in service_findings
            if finding.service != sentinel
            and finding.candidate_strength is CandidateStrength.STRONG
        ]
        if residual:
            return self._result(
                RecoveryStatus.INCONCLUSIVE,
                action,
                checked,
                fresh_window,
                api,
                history_start,
                budget_start,
                root_findings,
                traffic_findings,
                service_findings,
                sentinel,
                "previously affected services still show strong metric anomalies: "
                + ", ".join(residual),
            )

        return self._result(
            RecoveryStatus.VERIFIED,
            action,
            checked,
            fresh_window,
            api,
            history_start,
            budget_start,
            root_findings,
            traffic_findings,
            service_findings,
            sentinel,
            "required target metrics normalized, traffic continued, and no strong fresh anomaly remained",
        )

    def _traffic_findings(
        self,
        services: Sequence[str],
        metrics_by_service: Mapping[str, Sequence[MetricEvent]],
        baseline: BaselineStore,
    ) -> tuple[TrafficContinuityFinding, ...]:
        findings: list[TrafficContinuityFinding] = []
        for service in services:
            summary = baseline.metric_summaries.get((service, "request_rate"))
            if summary is None or summary.mean <= 0:
                continue
            samples = [
                event.value
                for event in metrics_by_service[service]
                if event.metric_name == "request_rate"
            ]
            if len(samples) < self.config.minimum_samples_per_required_metric:
                findings.append(
                    TrafficContinuityFinding(service, summary.mean, None, None, None)
                )
                continue
            observed = median(samples)
            ratio = observed / summary.mean
            findings.append(
                TrafficContinuityFinding(
                    service,
                    summary.mean,
                    observed,
                    ratio,
                    ratio >= self.config.minimum_request_rate_ratio,
                )
            )
        return tuple(findings)

    def _service_finding(
        self,
        service: str,
        metrics: Sequence[MetricEvent],
        baseline: BaselineStore,
    ) -> ServiceRecoveryFinding:
        results = (
            ("mad", self._mad.analyze(service, metrics, baseline)),
            ("change_point", self._cusum.analyze(service, metrics, baseline)),
            (
                "isolation_forest",
                self._isolation.analyze(service, metrics, baseline),
            ),
        )
        flags = tuple(name for name, result in results if result.flagged)
        strength = (
            CandidateStrength.STRONG
            if len(flags) >= 2
            else CandidateStrength.WEAK
            if len(flags) == 1
            else CandidateStrength.NOT_CANDIDATE
        )
        return ServiceRecoveryFinding(service, flags, strength)

    @staticmethod
    def _result(
        status: RecoveryStatus,
        action: RemediationAction,
        checked_services: Sequence[str],
        fresh_window: FreshObservationWindow | None,
        api: TelemetryQueryAPI,
        history_start: int,
        budget_start: int,
        root_findings: Sequence[MetricRecoveryFinding] = (),
        traffic_findings: Sequence[TrafficContinuityFinding] = (),
        service_findings: Sequence[ServiceRecoveryFinding] = (),
        sentinel: str | None = None,
        reason: str = "",
    ) -> RecoveryVerification:
        return RecoveryVerification(
            status,
            action.target_service,
            action.diagnosed_failure_mode,
            tuple(checked_services),
            fresh_window,
            api.query_history[history_start:],
            api.spent_budget - budget_start,
            tuple(root_findings),
            tuple(traffic_findings),
            tuple(service_findings),
            sentinel,
            reason,
        )


def _tail_metrics(
    events: Sequence[MetricEvent], sample_count: int
) -> tuple[MetricEvent, ...]:
    grouped: dict[str, list[MetricEvent]] = {}
    for event in sorted(events, key=lambda item: (item.event_timestamp, item.metric_name)):
        grouped.setdefault(event.metric_name, []).append(event)
    return tuple(
        event
        for metric_name in sorted(grouped)
        for event in grouped[metric_name][-sample_count:]
    )
