"""Typed records and aggregate metrics for controlled benchmark evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import fmean, median
from typing import Sequence

from src.rca import DiagnosisStatus
from src.remediation import RecoveryStatus


@dataclass(frozen=True, slots=True)
class BenchmarkIncidentResult:
    profile: str
    ablation: str
    seed: int
    true_root_service: str
    true_failure_mode: str
    diagnosis_status: str
    predicted_service: str | None
    predicted_failure_mode: str | None
    exact_pair_correct: bool
    root_service_correct: bool
    failure_mode_correct: bool
    diagnosis_query_budget_spent: int
    total_query_budget_spent: int
    planned_remediation_action: str | None
    recovery_status: str | None
    remediation_target_correct: bool
    verification_window_count: int
    follow_up_diagnosis_status: str | None
    localized_service: str | None = None
    action_eligible: bool = False
    action_blocked_for_evidence: bool = False
    confirmation_query_count: int = 0

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    profile: str
    ablation: str
    incident_count: int
    exact_pair_accuracy: float
    root_service_accuracy: float
    failure_mode_accuracy: float
    resolved_count: int
    resolved_rate: float
    ambiguous_count: int
    ambiguous_rate: float
    incomplete_budget_count: int
    incomplete_budget_rate: float
    no_candidates_count: int
    no_candidates_rate: float
    false_resolution_count: int
    false_resolution_rate: float
    mean_diagnosis_query_cost: float
    median_diagnosis_query_cost: float
    p95_diagnosis_query_cost: float
    safely_planned_count: int
    safely_planned_fraction: float
    verified_count: int
    verified_rate: float
    failed_count: int
    failed_rate: float
    inconclusive_count: int
    inconclusive_rate: float
    correct_remediation_target_count: int
    correct_remediation_target_rate: float
    mean_total_query_cost: float
    median_total_query_cost: float
    p95_total_query_cost: float
    blocked_unsafe_action_count: int
    wrong_action_verified_count: int
    unsupported_count: int
    unsupported_rate: float
    action_eligible_count: int
    action_blocked_for_evidence_count: int
    confirmation_query_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_results(
    results: Sequence[BenchmarkIncidentResult],
    *,
    profile: str,
    ablation: str,
) -> BenchmarkSummary:
    if not results:
        raise ValueError("cannot summarize an empty benchmark result set")
    count = len(results)
    status_counts = {
        status.value: sum(item.diagnosis_status == status.value for item in results)
        for status in DiagnosisStatus
    }
    recovery_counts = {
        status.value: sum(item.recovery_status == status.value for item in results)
        for status in RecoveryStatus
    }
    resolved_count = status_counts[DiagnosisStatus.RESOLVED.value]
    false_resolutions = sum(
        item.diagnosis_status == DiagnosisStatus.RESOLVED.value
        and not item.exact_pair_correct
        for item in results
    )
    planned = [item for item in results if item.planned_remediation_action is not None]
    correct_planned = sum(item.remediation_target_correct for item in planned)
    diagnosis_costs = [item.diagnosis_query_budget_spent for item in results]
    total_costs = [item.total_query_budget_spent for item in results]
    return BenchmarkSummary(
        profile=profile,
        ablation=ablation,
        incident_count=count,
        exact_pair_accuracy=_rate(sum(item.exact_pair_correct for item in results), count),
        root_service_accuracy=_rate(sum(item.root_service_correct for item in results), count),
        failure_mode_accuracy=_rate(sum(item.failure_mode_correct for item in results), count),
        resolved_count=resolved_count,
        resolved_rate=_rate(resolved_count, count),
        ambiguous_count=status_counts[DiagnosisStatus.AMBIGUOUS.value],
        ambiguous_rate=_rate(status_counts[DiagnosisStatus.AMBIGUOUS.value], count),
        incomplete_budget_count=status_counts[DiagnosisStatus.INCOMPLETE_BUDGET.value],
        incomplete_budget_rate=_rate(
            status_counts[DiagnosisStatus.INCOMPLETE_BUDGET.value], count
        ),
        no_candidates_count=status_counts[DiagnosisStatus.NO_CANDIDATES.value],
        no_candidates_rate=_rate(status_counts[DiagnosisStatus.NO_CANDIDATES.value], count),
        false_resolution_count=false_resolutions,
        false_resolution_rate=_rate(false_resolutions, resolved_count),
        mean_diagnosis_query_cost=fmean(diagnosis_costs),
        median_diagnosis_query_cost=median(diagnosis_costs),
        p95_diagnosis_query_cost=_p95(diagnosis_costs),
        safely_planned_count=len(planned),
        safely_planned_fraction=_rate(len(planned), count),
        verified_count=recovery_counts[RecoveryStatus.VERIFIED.value],
        verified_rate=_rate(recovery_counts[RecoveryStatus.VERIFIED.value], count),
        failed_count=recovery_counts[RecoveryStatus.FAILED.value],
        failed_rate=_rate(recovery_counts[RecoveryStatus.FAILED.value], count),
        inconclusive_count=recovery_counts[RecoveryStatus.INCONCLUSIVE.value],
        inconclusive_rate=_rate(
            recovery_counts[RecoveryStatus.INCONCLUSIVE.value], count
        ),
        correct_remediation_target_count=correct_planned,
        correct_remediation_target_rate=_rate(correct_planned, len(planned)),
        mean_total_query_cost=fmean(total_costs),
        median_total_query_cost=median(total_costs),
        p95_total_query_cost=_p95(total_costs),
        blocked_unsafe_action_count=count - len(planned),
        wrong_action_verified_count=sum(
            item.planned_remediation_action is not None
            and not item.remediation_target_correct
            and item.recovery_status == RecoveryStatus.VERIFIED.value
            for item in results
        ),
        unsupported_count=status_counts[DiagnosisStatus.UNSUPPORTED.value],
        unsupported_rate=_rate(
            status_counts[DiagnosisStatus.UNSUPPORTED.value], count
        ),
        action_eligible_count=sum(item.action_eligible for item in results),
        action_blocked_for_evidence_count=sum(
            item.action_blocked_for_evidence for item in results
        ),
        confirmation_query_count=sum(
            item.confirmation_query_count for item in results
        ),
    )


@dataclass(frozen=True, slots=True)
class HealthyControlResult:
    profile: str
    seed: int
    diagnosis_status: str
    diagnosis_query_budget_spent: int
    total_query_budget_spent: int
    action_count: int
    preserved: bool

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HealthyControlSummary:
    profile: str
    healthy_case_count: int
    healthy_no_candidate_count: int
    healthy_no_candidate_rate: float
    healthy_action_count: int
    healthy_action_rate: float
    healthy_preserved_count: int
    healthy_preserved_rate: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_healthy_results(
    results: Sequence[HealthyControlResult],
    *,
    profile: str,
) -> HealthyControlSummary:
    if not results:
        raise ValueError("cannot summarize an empty healthy-control result set")
    count = len(results)
    no_candidate_count = sum(
        item.diagnosis_status == DiagnosisStatus.NO_CANDIDATES.value
        for item in results
    )
    action_count = sum(item.action_count for item in results)
    preserved_count = sum(item.preserved for item in results)
    return HealthyControlSummary(
        profile,
        count,
        no_candidate_count,
        _rate(no_candidate_count, count),
        action_count,
        _rate(action_count, count),
        preserved_count,
        _rate(preserved_count, count),
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _p95(values: Sequence[int]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, ceil(0.95 * len(ordered)) - 1)])
