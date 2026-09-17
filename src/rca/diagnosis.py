"""Structured hypothesis generation, causal evaluation, and deterministic ranking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from statistics import fmean, median
from typing import Mapping, Sequence

from src.core.baseline import BaselineStore, MetricSummary
from src.core.models import MetricEvent, SpanEvent

from .candidates import CandidateStrength, ServiceCandidate
from .logs import LogEvidence
from .signatures import (
    FailureSignature,
    FailureSignatureLibrary,
    MetricDirection,
    TraceExpectation,
)


class EvidenceStatus(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


class EvidenceCategory(str, Enum):
    METRIC_PATTERN = "METRIC_PATTERN"
    LOG_SEMANTIC = "LOG_SEMANTIC"
    TRACE_LOCALIZATION = "TRACE_LOCALIZATION"
    TEMPORAL_PRECEDENCE = "TEMPORAL_PRECEDENCE"
    GRAPH_PROPAGATION = "GRAPH_PROPAGATION"
    CANDIDATE_STRENGTH = "CANDIDATE_STRENGTH"


@dataclass(frozen=True, slots=True)
class CausalEvidence:
    category: EvidenceCategory
    service: str
    failure_mode: str | None
    observation: str
    status: EvidenceStatus
    explanation: str
    event_timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class Hypothesis:
    service: str
    failure_mode: str
    candidate_strength: CandidateStrength


class MetricPatternDirection(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class MetricPattern:
    service: str
    metric_name: str
    direction: MetricPatternDirection
    baseline_median: float | None
    incident_median: float | None
    modified_z: float | None
    ratio_to_baseline: float | None
    onset_time: datetime | None


class TraceLocalizationKind(str, Enum):
    LOCAL = "LOCAL"
    DEPENDENCY = "DEPENDENCY"
    FAILURES = "FAILURES"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TraceLocalization:
    service: str
    kind: TraceLocalizationKind
    span_count: int
    complete_parent_child_count: int
    failed_span_fraction: float
    mean_span_duration_ms: float | None
    mean_dependency_duration_ms: float | None
    mean_local_duration_ms: float | None
    mean_dependency_fraction_lower_bound: float | None
    mean_dependency_fraction_upper_bound: float | None
    dependency_service: str | None


@dataclass(frozen=True, slots=True)
class RankingComponents:
    contradiction_count: int
    unexplained_strong_count: int
    explained_strong_count: int
    supporting_category_count: int
    direct_support_count: int
    direct_support_observation_count: int = 0

    def core_key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.contradiction_count,
            self.unexplained_strong_count,
            -self.explained_strong_count,
            -self.supporting_category_count,
            -self.direct_support_count,
            -self.direct_support_observation_count,
        )


@dataclass(frozen=True, slots=True)
class HypothesisEvaluation:
    hypothesis: Hypothesis
    evidence: tuple[CausalEvidence, ...]
    explained_strong_candidates: tuple[str, ...]
    unexplained_strong_candidates: tuple[str, ...]
    ranking: RankingComponents

    @property
    def supporting_evidence(self) -> tuple[CausalEvidence, ...]:
        return tuple(item for item in self.evidence if item.status is EvidenceStatus.SUPPORT)

    @property
    def contradictions(self) -> tuple[CausalEvidence, ...]:
        return tuple(
            item for item in self.evidence if item.status is EvidenceStatus.CONTRADICTION
        )

    def sort_key(self) -> tuple[int, int, int, int, int, int, str, str]:
        return (*self.ranking.core_key(), self.hypothesis.service, self.hypothesis.failure_mode)


@dataclass(frozen=True, slots=True)
class DiagnosisConfig:
    temporal_tolerance: timedelta = timedelta(seconds=8)
    direction_modified_z_threshold: float = 2.5
    onset_modified_z_threshold: float = 3.5
    extreme_onset_modified_z_threshold: float = 8.0
    onset_consecutive_samples: int = 3
    trace_latency_ratio_threshold: float = 2.0
    trace_dependency_fraction_threshold: float = 0.6
    trace_failure_fraction_threshold: float = 0.2
    dispersion_epsilon: float = 1e-9


class HypothesisGenerator:
    def __init__(self, signatures: FailureSignatureLibrary | None = None) -> None:
        self.signatures = signatures or FailureSignatureLibrary()

    def generate(self, candidates: Sequence[ServiceCandidate]) -> tuple[Hypothesis, ...]:
        return tuple(
            Hypothesis(candidate.service, failure_mode, candidate.strength)
            for candidate in sorted(candidates, key=lambda item: item.service)
            if candidate.strength is not CandidateStrength.NOT_CANDIDATE
            for failure_mode in self.signatures.failure_modes
        )


class TraceLocalizer:
    def __init__(self, config: DiagnosisConfig | None = None) -> None:
        self.config = config or DiagnosisConfig()

    def localize(
        self,
        service: str,
        spans: Sequence[SpanEvent],
        dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
        baseline: BaselineStore,
    ) -> TraceLocalization:
        service_spans = [span for span in spans if span.service == service]
        if not service_spans:
            return TraceLocalization(
                service,
                TraceLocalizationKind.UNKNOWN,
                0,
                0,
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        children_by_parent: dict[tuple[str, str], list[SpanEvent]] = {}
        for span in spans:
            if span.parent_span_id is not None:
                children_by_parent.setdefault((span.trace_id, span.parent_span_id), []).append(span)

        dependency_durations: list[float] = []
        local_durations: list[float] = []
        lower_fractions: list[float] = []
        upper_fractions: list[float] = []
        child_services: list[str] = []
        for parent in service_spans:
            children = children_by_parent.get((parent.trace_id, parent.span_id), [])
            if not children:
                continue
            duration = max(parent.duration_ms, self.config.dispersion_epsilon)
            durations = [min(child.duration_ms, duration) for child in children]
            dependency_lower = max(durations)
            dependency_upper = min(duration, sum(durations))
            dependency_durations.append(dependency_upper)
            local_durations.append(max(0.0, duration - dependency_upper))
            lower_fractions.append(dependency_lower / duration)
            upper_fractions.append(dependency_upper / duration)
            child_services.extend(child.service for child in children)

        failed_fraction = sum(span.status == "ERROR" for span in service_spans) / len(
            service_spans
        )
        mean_duration = fmean(span.duration_ms for span in service_spans)
        typical = baseline.typical_service_latencies_ms.get(service)
        latency_ratio = mean_duration / typical if typical is not None and typical > 0 else None
        known_dependencies = {
            dependency for caller, dependency in dependency_edges if caller == service
        }

        complete_count = len(dependency_durations)
        complete_evidence = complete_count == len(service_spans)
        if failed_fraction >= self.config.trace_failure_fraction_threshold:
            kind = TraceLocalizationKind.FAILURES
        elif dependency_durations and complete_evidence:
            mean_lower = fmean(lower_fractions)
            mean_upper = fmean(upper_fractions)
            local_fraction_threshold = (
                1.0 - self.config.trace_dependency_fraction_threshold
            )
            if mean_lower >= self.config.trace_dependency_fraction_threshold:
                kind = TraceLocalizationKind.DEPENDENCY
            elif (
                mean_upper <= local_fraction_threshold
                and latency_ratio is not None
                and latency_ratio >= self.config.trace_latency_ratio_threshold
            ):
                kind = TraceLocalizationKind.LOCAL
            else:
                kind = TraceLocalizationKind.MIXED
        elif (
            not known_dependencies
            and latency_ratio is not None
            and latency_ratio >= self.config.trace_latency_ratio_threshold
        ):
            kind = TraceLocalizationKind.LOCAL
        else:
            kind = TraceLocalizationKind.UNKNOWN

        dependency_service = None
        if child_services:
            dependency_service = sorted(
                set(child_services),
                key=lambda item: (-child_services.count(item), item),
            )[0]
        return TraceLocalization(
            service,
            kind,
            len(service_spans),
            complete_count,
            failed_fraction,
            mean_duration,
            fmean(dependency_durations) if dependency_durations else None,
            fmean(local_durations) if local_durations else None,
            fmean(lower_fractions) if lower_fractions else None,
            fmean(upper_fractions) if upper_fractions else None,
            dependency_service,
        )


class HypothesisEvaluator:
    def __init__(
        self,
        signatures: FailureSignatureLibrary | None = None,
        config: DiagnosisConfig | None = None,
    ) -> None:
        self.signatures = signatures or FailureSignatureLibrary()
        self.config = config or DiagnosisConfig()
        self._trace_localizer = TraceLocalizer(self.config)

    def evaluate_all(
        self,
        hypotheses: Sequence[Hypothesis],
        candidates: Sequence[ServiceCandidate],
        metrics_by_service: Mapping[str, Sequence[MetricEvent]],
        logs: Sequence[LogEvidence],
        spans: Sequence[SpanEvent],
        dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
        baseline: BaselineStore,
    ) -> tuple[HypothesisEvaluation, ...]:
        patterns = {
            service: self.metric_patterns(service, events, baseline)
            for service, events in metrics_by_service.items()
        }
        onsets = {
            service: min(
                (
                    pattern.onset_time
                    for pattern in service_patterns.values()
                    if pattern.direction
                    in {MetricPatternDirection.HIGH, MetricPatternDirection.LOW}
                    and pattern.onset_time is not None
                ),
                default=None,
            )
            for service, service_patterns in patterns.items()
        }
        localizations = {
            service: self._trace_localizer.localize(
                service, spans, dependency_edges, baseline
            )
            for service in sorted({hypothesis.service for hypothesis in hypotheses})
        }
        trace_dependency_edges = _trace_dependency_edges(spans)
        return tuple(
            self._evaluate(
                hypothesis,
                candidates,
                patterns.get(hypothesis.service, {}),
                onsets,
                logs,
                localizations[hypothesis.service],
                dependency_edges,
                trace_dependency_edges,
            )
            for hypothesis in hypotheses
        )

    def metric_patterns(
        self,
        service: str,
        metrics: Sequence[MetricEvent],
        baseline: BaselineStore,
    ) -> dict[str, MetricPattern]:
        grouped: dict[str, list[MetricEvent]] = {}
        for event in sorted(metrics, key=lambda item: (item.event_timestamp, item.metric_name)):
            if event.service == service:
                grouped.setdefault(event.metric_name, []).append(event)

        patterns: dict[str, MetricPattern] = {}
        metric_names = {
            metric_name
            for baseline_service, metric_name in baseline.metric_summaries
            if baseline_service == service
        } | set(grouped)
        for metric_name in sorted(metric_names):
            events = grouped.get(metric_name, [])
            summary = baseline.metric_summaries.get((service, metric_name))
            if summary is None or not events:
                patterns[metric_name] = MetricPattern(
                    service,
                    metric_name,
                    MetricPatternDirection.UNKNOWN,
                    summary.median if summary else None,
                    None,
                    None,
                    None,
                    None,
                )
                continue
            incident_median = median(event.value for event in events)
            dispersion = _dispersion(summary, self.config.dispersion_epsilon)
            modified_z = 0.67448975 * (incident_median - summary.median) / dispersion
            if modified_z >= self.config.direction_modified_z_threshold:
                direction = MetricPatternDirection.HIGH
            elif modified_z <= -self.config.direction_modified_z_threshold:
                direction = MetricPatternDirection.LOW
            else:
                direction = MetricPatternDirection.NORMAL
            ratio = incident_median / summary.median if summary.median != 0 else None
            onset_time = (
                self._metric_onset(events, summary)
                if direction
                in {MetricPatternDirection.HIGH, MetricPatternDirection.LOW}
                else None
            )
            patterns[metric_name] = MetricPattern(
                service,
                metric_name,
                direction,
                summary.median,
                incident_median,
                modified_z,
                ratio,
                onset_time,
            )
        return patterns

    def _metric_onset(
        self, events: Sequence[MetricEvent], summary: MetricSummary
    ) -> datetime | None:
        dispersion = _dispersion(summary, self.config.dispersion_epsilon)
        run_direction = 0
        run_start: datetime | None = None
        run_length = 0
        for event in events:
            standardized_delta = (
                0.67448975 * (event.value - summary.median) / dispersion
            )
            if abs(standardized_delta) >= self.config.extreme_onset_modified_z_threshold:
                return event.event_timestamp
            if standardized_delta >= self.config.onset_modified_z_threshold:
                direction = 1
            elif standardized_delta <= -self.config.onset_modified_z_threshold:
                direction = -1
            else:
                direction = 0
            if direction == 0:
                run_direction = run_length = 0
                run_start = None
                continue
            if direction != run_direction:
                run_direction = direction
                run_length = 1
                run_start = event.event_timestamp
            else:
                run_length += 1
            if run_length >= self.config.onset_consecutive_samples:
                return run_start
        return None

    def _evaluate(
        self,
        hypothesis: Hypothesis,
        candidates: Sequence[ServiceCandidate],
        patterns: Mapping[str, MetricPattern],
        onsets: Mapping[str, datetime | None],
        logs: Sequence[LogEvidence],
        localization: TraceLocalization,
        dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
        trace_dependency_edges: set[tuple[str, str]],
    ) -> HypothesisEvaluation:
        signature = self.signatures.get(hypothesis.failure_mode)
        evidence: list[CausalEvidence] = [self._candidate_evidence(hypothesis)]
        evidence.extend(self._metric_evidence(hypothesis, signature, patterns))
        evidence.extend(self._log_evidence(hypothesis, signature, logs))
        evidence.append(self._trace_evidence(hypothesis, signature, localization))

        strong_candidates = sorted(
            candidate.service
            for candidate in candidates
            if candidate.strength is CandidateStrength.STRONG
        )
        explained: list[str] = []
        unexplained: list[str] = []
        for service in strong_candidates:
            path = dependency_path(service, hypothesis.service, dependency_edges)
            if path is None:
                unexplained.append(service)
                evidence.append(
                    CausalEvidence(
                        EvidenceCategory.GRAPH_PROPAGATION,
                        service,
                        hypothesis.failure_mode,
                        "no dependency path",
                        EvidenceStatus.NEUTRAL,
                        f"{service} has no dependency path to {hypothesis.service}.",
                    )
                )
                continue
            explained.append(service)
            evidence.append(
                CausalEvidence(
                    EvidenceCategory.GRAPH_PROPAGATION,
                    service,
                    hypothesis.failure_mode,
                    " -> ".join(path),
                    EvidenceStatus.SUPPORT,
                    f"{service} depends on {hypothesis.service} through {' -> '.join(path)}.",
                )
            )
            if service != hypothesis.service:
                trace_path = dependency_path(
                    service, hypothesis.service, trace_dependency_edges
                )
                if trace_path is not None:
                    evidence.append(
                        CausalEvidence(
                            EvidenceCategory.TEMPORAL_PRECEDENCE,
                            service,
                            hypothesis.failure_mode,
                            f"trace path={' -> '.join(trace_path)}",
                            EvidenceStatus.SUPPORT,
                            "Trace parent/child identities establish the causal path "
                            f"{' -> '.join(trace_path)} without relying on synchronized clocks.",
                        )
                    )
                else:
                    evidence.append(
                        self._temporal_evidence(
                            hypothesis,
                            service,
                            onsets.get(hypothesis.service),
                            onsets.get(service),
                        )
                    )

        contradictions = sum(
            item.status is EvidenceStatus.CONTRADICTION for item in evidence
        )
        supporting_categories = {
            item.category for item in evidence if item.status is EvidenceStatus.SUPPORT
        }
        direct_categories = {
            item.category
            for item in evidence
            if item.status is EvidenceStatus.SUPPORT
            and item.service == hypothesis.service
            and item.category
            in {
                EvidenceCategory.METRIC_PATTERN,
                EvidenceCategory.LOG_SEMANTIC,
                EvidenceCategory.TRACE_LOCALIZATION,
                EvidenceCategory.CANDIDATE_STRENGTH,
            }
        }
        direct_observations = sum(
            item.status is EvidenceStatus.SUPPORT
            and item.service == hypothesis.service
            and item.category
            in {
                EvidenceCategory.METRIC_PATTERN,
                EvidenceCategory.LOG_SEMANTIC,
                EvidenceCategory.TRACE_LOCALIZATION,
                EvidenceCategory.CANDIDATE_STRENGTH,
            }
            for item in evidence
        )
        ranking = RankingComponents(
            contradictions,
            len(unexplained),
            len(explained),
            len(supporting_categories),
            len(direct_categories),
            direct_observations,
        )
        return HypothesisEvaluation(
            hypothesis,
            tuple(evidence),
            tuple(explained),
            tuple(unexplained),
            ranking,
        )

    @staticmethod
    def _candidate_evidence(hypothesis: Hypothesis) -> CausalEvidence:
        return CausalEvidence(
            EvidenceCategory.CANDIDATE_STRENGTH,
            hypothesis.service,
            hypothesis.failure_mode,
            hypothesis.candidate_strength.value,
            EvidenceStatus.SUPPORT,
            f"{hypothesis.service} is a {hypothesis.candidate_strength.value} anomaly candidate.",
        )

    def _metric_evidence(
        self,
        hypothesis: Hypothesis,
        signature: FailureSignature,
        patterns: Mapping[str, MetricPattern],
    ) -> list[CausalEvidence]:
        evidence: list[CausalEvidence] = []
        for expectation in signature.metric_expectations:
            pattern = patterns.get(expectation.metric_name)
            if pattern is None or pattern.direction is MetricPatternDirection.UNKNOWN:
                evidence.append(
                    CausalEvidence(
                        EvidenceCategory.METRIC_PATTERN,
                        hypothesis.service,
                        hypothesis.failure_mode,
                        f"{expectation.metric_name}=unknown",
                        EvidenceStatus.NEUTRAL,
                        f"{hypothesis.service} has no conclusive {expectation.metric_name} evidence.",
                    )
                )
                continue
            expected = (
                MetricPatternDirection.HIGH
                if expectation.direction is MetricDirection.HIGH
                else MetricPatternDirection.LOW
            )
            opposite = (
                MetricPatternDirection.LOW
                if expected is MetricPatternDirection.HIGH
                else MetricPatternDirection.HIGH
            )
            if pattern.direction is expected:
                status = EvidenceStatus.SUPPORT
            elif expectation.required and pattern.direction in {
                MetricPatternDirection.NORMAL,
                opposite,
            }:
                status = EvidenceStatus.CONTRADICTION
            else:
                status = EvidenceStatus.NEUTRAL
            ratio_text = (
                f"{pattern.ratio_to_baseline:.2f}x"
                if pattern.ratio_to_baseline is not None
                else f"modified-z {pattern.modified_z:.2f}"
            )
            evidence.append(
                CausalEvidence(
                    EvidenceCategory.METRIC_PATTERN,
                    hypothesis.service,
                    hypothesis.failure_mode,
                    f"{expectation.metric_name}={pattern.direction.value}; {ratio_text}",
                    status,
                    f"{hypothesis.service} {expectation.metric_name} is {pattern.direction.value.lower()} "
                    f"relative to baseline ({ratio_text}).",
                )
            )
        return evidence

    def _log_evidence(
        self,
        hypothesis: Hypothesis,
        signature: FailureSignature,
        logs: Sequence[LogEvidence],
    ) -> list[CausalEvidence]:
        matching = [
            item
            for item in logs
            if item.service == hypothesis.service
            and item.semantic_category in signature.log_categories
        ]
        if not matching:
            return [
                CausalEvidence(
                    EvidenceCategory.LOG_SEMANTIC,
                    hypothesis.service,
                    hypothesis.failure_mode,
                    "no matching semantic log",
                    EvidenceStatus.NEUTRAL,
                    f"No queried {hypothesis.service} log matches {hypothesis.failure_mode}.",
                )
            ]
        earliest_time = min(item.event_timestamp for item in matching)
        relevant = [
            item
            for item in matching
            if item.event_timestamp <= earliest_time + self.config.temporal_tolerance
        ]
        best = min(
            relevant,
            key=lambda item: (
                -item.similarity_score,
                item.event_timestamp,
                item.message,
            ),
        )
        return [
            CausalEvidence(
                EvidenceCategory.LOG_SEMANTIC,
                hypothesis.service,
                hypothesis.failure_mode,
                f"{best.semantic_category}; similarity={best.similarity_score:.3f}",
                EvidenceStatus.SUPPORT,
                f"{hypothesis.service} log evidence matches {best.semantic_category} "
                f"with similarity {best.similarity_score:.3f}.",
                best.event_timestamp,
            )
        ]

    @staticmethod
    def _trace_evidence(
        hypothesis: Hypothesis,
        signature: FailureSignature,
        localization: TraceLocalization,
    ) -> CausalEvidence:
        matching = (
            localization.kind is TraceLocalizationKind.LOCAL
            and TraceExpectation.LOCAL_LATENCY in signature.trace_expectations
        ) or (
            localization.kind is TraceLocalizationKind.DEPENDENCY
            and TraceExpectation.DEPENDENCY_LATENCY in signature.trace_expectations
        ) or (
            localization.kind is TraceLocalizationKind.FAILURES
            and bool(
                signature.trace_expectations
                & {TraceExpectation.FAILURES, TraceExpectation.UNAVAILABLE}
            )
        )
        contradictory = (
            localization.kind is TraceLocalizationKind.DEPENDENCY
            and signature.trace_expectations == frozenset({TraceExpectation.LOCAL_LATENCY})
        )
        status = (
            EvidenceStatus.SUPPORT
            if matching
            else EvidenceStatus.CONTRADICTION
            if contradictory
            else EvidenceStatus.NEUTRAL
        )
        detail = localization.kind.value
        if localization.mean_dependency_fraction_lower_bound is not None:
            detail += (
                "; dependency_fraction="
                f"{localization.mean_dependency_fraction_lower_bound:.3f}.."
                f"{localization.mean_dependency_fraction_upper_bound:.3f}; "
                f"local_ms={localization.mean_local_duration_ms:.3f}"
            )
        return CausalEvidence(
            EvidenceCategory.TRACE_LOCALIZATION,
            hypothesis.service,
            hypothesis.failure_mode,
            detail,
            status,
            f"Trace localization for {hypothesis.service} is {localization.kind.value.lower()} "
            f"from {localization.span_count} queried spans.",
        )

    def _temporal_evidence(
        self,
        hypothesis: Hypothesis,
        affected_service: str,
        root_onset: datetime | None,
        affected_onset: datetime | None,
    ) -> CausalEvidence:
        if root_onset is None or affected_onset is None:
            return CausalEvidence(
                EvidenceCategory.TEMPORAL_PRECEDENCE,
                affected_service,
                hypothesis.failure_mode,
                "onset unknown",
                EvidenceStatus.NEUTRAL,
                f"Temporal order between {hypothesis.service} and {affected_service} is unknown.",
            )
        if root_onset + self.config.temporal_tolerance < affected_onset:
            status = EvidenceStatus.SUPPORT
            explanation = (
                f"{hypothesis.service} became abnormal before {affected_service} beyond "
                "the clock-skew tolerance."
            )
        elif affected_onset + self.config.temporal_tolerance < root_onset:
            status = EvidenceStatus.CONTRADICTION
            explanation = (
                f"{hypothesis.service} became abnormal after {affected_service} beyond "
                "the clock-skew tolerance."
            )
        else:
            status = EvidenceStatus.NEUTRAL
            explanation = (
                f"{hypothesis.service} and {affected_service} onset times are within "
                "the clock-skew tolerance."
            )
        return CausalEvidence(
            EvidenceCategory.TEMPORAL_PRECEDENCE,
            affected_service,
            hypothesis.failure_mode,
            f"root={root_onset.isoformat()}, affected={affected_onset.isoformat()}",
            status,
            explanation,
        )


def dependency_path(
    caller: str,
    dependency: str,
    edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
) -> tuple[str, ...] | None:
    if caller == dependency:
        return (caller,)
    adjacency: dict[str, list[str]] = {}
    for source, target in sorted(edges):
        adjacency.setdefault(source, []).append(target)
    pending: deque[tuple[str, tuple[str, ...]]] = deque([(caller, (caller,))])
    visited = {caller}
    while pending:
        current, path = pending.popleft()
        for target in adjacency.get(current, []):
            if target == dependency:
                return (*path, target)
            if target not in visited:
                visited.add(target)
                pending.append((target, (*path, target)))
    return None


def _trace_dependency_edges(spans: Sequence[SpanEvent]) -> set[tuple[str, str]]:
    by_identity = {(span.trace_id, span.span_id): span for span in spans}
    return {
        (parent.service, span.service)
        for span in spans
        if span.parent_span_id is not None
        and (
            parent := by_identity.get((span.trace_id, span.parent_span_id))
        )
        is not None
        and parent.service != span.service
    }


def rank_evaluations(
    evaluations: Sequence[HypothesisEvaluation],
) -> tuple[HypothesisEvaluation, ...]:
    return tuple(sorted(evaluations, key=lambda evaluation: evaluation.sort_key()))


def _dispersion(summary: MetricSummary, epsilon: float) -> float:
    if summary.median_absolute_deviation > epsilon:
        return summary.median_absolute_deviation
    if summary.standard_deviation > epsilon:
        return summary.standard_deviation * 0.67448975
    return max(abs(summary.median) * 0.01, epsilon)
