"""Autonomous budget-aware diagnosis loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

from src.core.baseline import BaselineStore
from src.core.budget import QueryHistoryEntry
from src.core.models import LogEvent, MetricEvent, SpanEvent
from src.core.telemetry import TelemetryQueryAPI

from .anomaly import (
    CUSUMDetector,
    CUSUMServiceEvidence,
    IsolationForestDetector,
    IsolationForestEvidence,
    MADDetector,
    MADServiceEvidence,
)
from .candidates import CandidateGenerator, CandidateStrength, ServiceCandidate
from .diagnosis import (
    DiagnosisConfig,
    DiagnosisDecisionExplanation,
    EvidenceCategory,
    EvidenceStatus,
    Hypothesis,
    HypothesisEvaluation,
    HypothesisEvaluator,
    HypothesisGenerator,
    explain_ranked_decision,
    explain_unsupported_decision,
    rank_evaluations,
)
from .graph import TraceGraphAnalyzer
from .logs import LogEvidence, LogEvidenceExtractor
from .planner import ActiveQueryPlanner, PlannedQuery
from .signatures import FailureSignatureLibrary


class DiagnosisStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE_BUDGET = "INCOMPLETE_BUDGET"
    NO_CANDIDATES = "NO_CANDIDATES"


class BootstrapStrategy(str, Enum):
    EXHAUSTIVE_METRIC_BOOTSTRAP = "exhaustive_metric_bootstrap"
    TOPOLOGY_METRIC_SUBSET = "topology_metric_subset"


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    status: DiagnosisStatus
    best_hypothesis: Hypothesis | None
    ranked_hypotheses: tuple[HypothesisEvaluation, ...]
    candidates: tuple[ServiceCandidate, ...]
    planned_queries: tuple[PlannedQuery, ...]
    queries_executed: tuple[QueryHistoryEntry, ...]
    budget_spent: int
    budget_remaining: int
    localized_service: str | None = None
    decision_explanation: DiagnosisDecisionExplanation | None = None
    confirmation_query_count: int = 0


@dataclass(frozen=True, slots=True)
class DiagnosisAgentConfig:
    maximum_active_queries: int = 100
    bootstrap_strategy: BootstrapStrategy = (
        BootstrapStrategy.EXHAUSTIVE_METRIC_BOOTSTRAP
    )


class LogEvidenceProvider(Protocol):
    def extract(self, logs: Sequence[LogEvent]) -> tuple[LogEvidence, ...]: ...


MADAnalyzer = Callable[
    [str, Sequence[MetricEvent], BaselineStore], MADServiceEvidence | None
]
CUSUMAnalyzer = Callable[
    [str, Sequence[MetricEvent], BaselineStore], CUSUMServiceEvidence | None
]
IsolationAnalyzer = Callable[
    [str, Sequence[MetricEvent], BaselineStore], IsolationForestEvidence | None
]


class DiagnosisAgent:
    def __init__(
        self,
        signatures: FailureSignatureLibrary | None = None,
        diagnosis_config: DiagnosisConfig | None = None,
        agent_config: DiagnosisAgentConfig | None = None,
        log_extractor: LogEvidenceProvider | None = None,
        mad_analyzer: MADAnalyzer | None = None,
        cusum_analyzer: CUSUMAnalyzer | None = None,
        isolation_analyzer: IsolationAnalyzer | None = None,
        trace_analyzer: TraceGraphAnalyzer | None = None,
        hypothesis_evaluator: HypothesisEvaluator | None = None,
    ) -> None:
        self.signatures = signatures or FailureSignatureLibrary()
        self.diagnosis_config = diagnosis_config or DiagnosisConfig()
        self.agent_config = agent_config or DiagnosisAgentConfig()
        self._mad_analyze = mad_analyzer or MADDetector().analyze
        self._cusum_analyze = cusum_analyzer or CUSUMDetector().analyze
        self._isolation_analyze = (
            isolation_analyzer or IsolationForestDetector().analyze
        )
        self._candidate_generator = CandidateGenerator()
        self._hypothesis_generator = HypothesisGenerator(self.signatures)
        self._evaluator = hypothesis_evaluator or HypothesisEvaluator(
            self.signatures, self.diagnosis_config
        )
        self._planner = ActiveQueryPlanner(self.signatures)
        self._trace_analyzer = trace_analyzer or TraceGraphAnalyzer()
        self._log_extractor = log_extractor or LogEvidenceExtractor()

    def diagnose(
        self,
        api: TelemetryQueryAPI,
        baseline: BaselineStore,
        start_time: datetime,
        end_time: datetime,
        intervention_contradictions: Mapping[tuple[str, str], str] | None = None,
    ) -> DiagnosisResult:
        history_start = len(api.query_history)
        services = tuple(sorted(baseline.known_services))
        metrics_by_service: dict[str, tuple[MetricEvent, ...]] = {}
        logs_by_service: dict[str, tuple[LogEvent, ...]] = {}
        spans_by_service: dict[str, tuple[SpanEvent, ...]] = {}
        executed: set[tuple[str, str, datetime, datetime]] = set()
        planned_queries: list[PlannedQuery] = []

        bootstrap_order = self._bootstrap_order(
            services, baseline, api, start_time, end_time
        )
        for service in bootstrap_order:
            if not api.can_afford("metrics", service, start_time, end_time):
                continue
            metrics_by_service[service] = api.query_metrics(service, start_time, end_time)
            executed.add(("metrics", service, start_time, end_time))

        observed_services = tuple(sorted(metrics_by_service))
        if not observed_services:
            return self._result(
                DiagnosisStatus.INCOMPLETE_BUDGET,
                None,
                (),
                (),
                planned_queries,
                api,
                history_start,
            )

        candidates, evaluations = self._analyze(
            observed_services,
            metrics_by_service,
            logs_by_service,
            spans_by_service,
            baseline,
            intervention_contradictions,
        )
        if len(observed_services) < len(services):
            return self._result(
                DiagnosisStatus.INCOMPLETE_BUDGET,
                None,
                rank_evaluations(evaluations),
                candidates,
                planned_queries,
                api,
                history_start,
            )
        if not any(
            candidate.strength is not CandidateStrength.NOT_CANDIDATE
            for candidate in candidates
        ):
            return self._result(
                DiagnosisStatus.NO_CANDIDATES,
                None,
                (),
                candidates,
                planned_queries,
                api,
                history_start,
            )

        ranked = rank_evaluations(evaluations)
        for _ in range(self.agent_config.maximum_active_queries):
            if self._is_resolved(ranked):
                return self._result(
                    DiagnosisStatus.RESOLVED,
                    ranked[0].hypothesis,
                    ranked,
                    candidates,
                    planned_queries,
                    api,
                    history_start,
                )
            unsupported_service = self._unsupported_service(ranked, candidates)
            if unsupported_service is not None:
                return self._result(
                    DiagnosisStatus.UNSUPPORTED,
                    None,
                    ranked,
                    candidates,
                    planned_queries,
                    api,
                    history_start,
                    localized_service=unsupported_service,
                )

            cohort = self._planner_cohort(ranked)
            next_query = self._planner.choose_next(
                [evaluation.hypothesis for evaluation in cohort],
                {evaluation.hypothesis.service for evaluation in cohort},
                start_time,
                end_time,
                api,
                executed,
            )
            if next_query is None:
                discriminating_query_unavailable = (
                    self._planner.has_unexecuted_discriminating_query(
                        [evaluation.hypothesis for evaluation in cohort],
                        {evaluation.hypothesis.service for evaluation in cohort},
                        start_time,
                        end_time,
                        executed,
                    )
                )
                status = (
                    DiagnosisStatus.INCOMPLETE_BUDGET
                    if discriminating_query_unavailable
                    else DiagnosisStatus.AMBIGUOUS
                )
                return self._result(
                    status,
                    ranked[0].hypothesis if ranked else None,
                    ranked,
                    candidates,
                    planned_queries,
                    api,
                    history_start,
                )

            self._execute(next_query, api, metrics_by_service, logs_by_service, spans_by_service)
            executed.add(next_query.key)
            planned_queries.append(next_query)
            candidates, evaluations = self._analyze(
                services,
                metrics_by_service,
                logs_by_service,
                spans_by_service,
                baseline,
                intervention_contradictions,
            )
            ranked = rank_evaluations(evaluations)

        unsupported_service = self._unsupported_service(ranked, candidates)
        if unsupported_service is not None:
            return self._result(
                DiagnosisStatus.UNSUPPORTED,
                None,
                ranked,
                candidates,
                planned_queries,
                api,
                history_start,
                localized_service=unsupported_service,
            )
        return self._result(
            DiagnosisStatus.AMBIGUOUS,
            ranked[0].hypothesis if ranked else None,
            ranked,
            candidates,
            planned_queries,
            api,
            history_start,
        )

    def confirm_for_action(
        self,
        api: TelemetryQueryAPI,
        baseline: BaselineStore,
        start_time: datetime,
        end_time: datetime,
        diagnosis: DiagnosisResult,
    ) -> DiagnosisResult:
        if (
            diagnosis.status is not DiagnosisStatus.RESOLVED
            or diagnosis.best_hypothesis is None
        ):
            return diagnosis
        evaluation = next(
            (
                item
                for item in diagnosis.ranked_hypotheses
                if item.hypothesis == diagnosis.best_hypothesis
            ),
            None,
        )
        if evaluation is None:
            return diagnosis
        direct = {
            evidence.category
            for evidence in evaluation.supporting_evidence
            if evidence.service == evaluation.hypothesis.service
            and evidence.category
            in {
                EvidenceCategory.METRIC_PATTERN,
                EvidenceCategory.LOG_SEMANTIC,
                EvidenceCategory.TRACE_LOCALIZATION,
            }
        }
        if len(direct) != 1:
            return diagnosis
        if any(
            api.query_cost("metrics", service, start_time, end_time) != 0
            for service in baseline.known_services
        ):
            return diagnosis

        query_by_category = (
            (EvidenceCategory.LOG_SEMANTIC, "logs"),
            (EvidenceCategory.TRACE_LOCALIZATION, "traces"),
            (EvidenceCategory.METRIC_PATTERN, "metrics"),
        )
        service = diagnosis.best_hypothesis.service
        query_type = next(
            (
                candidate_query
                for category, candidate_query in query_by_category
                if category not in direct
                and api.query_cost(candidate_query, service, start_time, end_time) > 0
                and api.can_afford(candidate_query, service, start_time, end_time)
            ),
            None,
        )
        if query_type is None:
            return diagnosis

        history_start = len(api.query_history)
        cost = api.query_cost(query_type, service, start_time, end_time)
        confirmation = PlannedQuery(
            query_type,
            service,
            start_time,
            end_time,
            0,
            cost,
            0.0,
        )
        metrics_by_service: dict[str, tuple[MetricEvent, ...]] = {}
        logs_by_service: dict[str, tuple[LogEvent, ...]] = {}
        spans_by_service: dict[str, tuple[SpanEvent, ...]] = {}
        self._execute(
            confirmation,
            api,
            metrics_by_service,
            logs_by_service,
            spans_by_service,
        )
        for observed_service in sorted(baseline.known_services):
            for cached_type, target in (
                ("metrics", metrics_by_service),
                ("logs", logs_by_service),
                ("traces", spans_by_service),
            ):
                if api.query_cost(
                    cached_type, observed_service, start_time, end_time
                ) != 0:
                    continue
                cached_query = PlannedQuery(
                    cached_type,
                    observed_service,
                    start_time,
                    end_time,
                    0,
                    0,
                    0.0,
                )
                self._execute(
                    cached_query,
                    api,
                    metrics_by_service,
                    logs_by_service,
                    spans_by_service,
                )

        observed_services = tuple(sorted(metrics_by_service))
        candidates, evaluations = self._analyze(
            observed_services,
            metrics_by_service,
            logs_by_service,
            spans_by_service,
            baseline,
        )
        ranked = rank_evaluations(evaluations)
        unsupported = self._unsupported_service(ranked, candidates)
        if self._is_resolved(ranked):
            status = DiagnosisStatus.RESOLVED
            best = ranked[0].hypothesis
            localized_service = best.service
        elif unsupported is not None:
            status = DiagnosisStatus.UNSUPPORTED
            best = None
            localized_service = unsupported
        else:
            status = DiagnosisStatus.AMBIGUOUS
            best = ranked[0].hypothesis if ranked else None
            localized_service = best.service if best is not None else None
        result = self._result(
            status,
            best,
            ranked,
            candidates,
            (*diagnosis.planned_queries, confirmation),
            api,
            history_start,
            localized_service=localized_service,
            confirmation_query_count=diagnosis.confirmation_query_count + 1,
        )
        return replace(
            result,
            queries_executed=(
                *diagnosis.queries_executed,
                *result.queries_executed,
            ),
        )

    def _bootstrap_order(
        self,
        services: Sequence[str],
        baseline: BaselineStore,
        api: TelemetryQueryAPI,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[str, ...]:
        exhaustive_affordable = sum(
            api.query_cost("metrics", service, start_time, end_time)
            for service in services
        ) <= api.remaining_budget
        if (
            self.agent_config.bootstrap_strategy
            is BootstrapStrategy.EXHAUSTIVE_METRIC_BOOTSTRAP
            and exhaustive_affordable
        ):
            return tuple(sorted(services))
        return _topology_bootstrap_order(
            services, baseline.cached_dependency_edges
        )

    def _analyze(
        self,
        services: Sequence[str],
        metrics_by_service: Mapping[str, Sequence[MetricEvent]],
        logs_by_service: Mapping[str, Sequence[LogEvent]],
        spans_by_service: Mapping[str, Sequence[SpanEvent]],
        baseline: BaselineStore,
        intervention_contradictions: Mapping[tuple[str, str], str] | None = None,
    ) -> tuple[tuple[ServiceCandidate, ...], tuple[HypothesisEvaluation, ...]]:
        mad = {
            service: self._mad_analyze(service, metrics_by_service[service], baseline)
            for service in services
        }
        change = {
            service: self._cusum_analyze(service, metrics_by_service[service], baseline)
            for service in services
        }
        isolation = {
            service: self._isolation_analyze(
                service, metrics_by_service[service], baseline
            )
            for service in services
        }
        logs = tuple(event for values in logs_by_service.values() for event in values)
        spans_by_identity = {
            (span.trace_id, span.span_id): span
            for values in spans_by_service.values()
            for span in values
        }
        spans = tuple(
            sorted(
                spans_by_identity.values(),
                key=lambda span: (
                    span.trace_id,
                    span.start_timestamp,
                    span.span_id,
                ),
            )
        )
        log_evidence = self._log_extractor.extract(logs)
        trace_graph = self._trace_analyzer.reconstruct(spans, baseline)
        candidates = self._candidate_generator.generate(
            services,
            mad,
            change,
            isolation,
            trace_graph.service_evidence,
            log_evidence,
        )
        hypotheses = self._hypothesis_generator.generate(candidates)
        edges = set(baseline.cached_dependency_edges)
        edges.update(
            (edge.caller_service, edge.callee_service) for edge in trace_graph.edges
        )
        evaluations = self._evaluator.evaluate_all(
            hypotheses,
            candidates,
            metrics_by_service,
            log_evidence,
            spans,
            edges,
            baseline,
            intervention_contradictions,
        )
        return candidates, evaluations

    @staticmethod
    def _execute(
        query: PlannedQuery,
        api: TelemetryQueryAPI,
        metrics: dict[str, tuple[MetricEvent, ...]],
        logs: dict[str, tuple[LogEvent, ...]],
        spans: dict[str, tuple[SpanEvent, ...]],
    ) -> None:
        if query.query_type == "metrics":
            metrics[query.service] = api.query_metrics(
                query.service, query.start_time, query.end_time
            )
        elif query.query_type == "logs":
            logs[query.service] = api.query_logs(
                query.service, query.start_time, query.end_time
            )
        elif query.query_type == "traces":
            spans[query.service] = api.query_traces(
                query.service, query.start_time, query.end_time
            )
        else:
            raise ValueError(f"unsupported query type: {query.query_type}")

    @staticmethod
    def _planner_cohort(
        ranked: Sequence[HypothesisEvaluation],
    ) -> tuple[HypothesisEvaluation, ...]:
        if not ranked:
            return ()
        minimum_contradictions = ranked[0].ranking.contradiction_count
        eligible = [
            evaluation
            for evaluation in ranked
            if evaluation.ranking.contradiction_count == minimum_contradictions
        ]
        minimum_unexplained = min(
            evaluation.ranking.unexplained_strong_count for evaluation in eligible
        )
        cohort = tuple(
            evaluation
            for evaluation in eligible
            if evaluation.ranking.unexplained_strong_count == minimum_unexplained
        )
        if len(cohort) == 1 and len(ranked) > 1:
            return (cohort[0], ranked[1])
        return cohort

    @staticmethod
    def _is_resolved(ranked: Sequence[HypothesisEvaluation]) -> bool:
        if not ranked:
            return False
        best = ranked[0]
        if best.ranking.contradiction_count > 0:
            return False
        if len(ranked) > 1 and best.ranking.core_key() == ranked[1].ranking.core_key():
            return False
        direct_confirming_categories = {
            evidence.category
            for evidence in best.supporting_evidence
            if evidence.service == best.hypothesis.service
            and evidence.category
            in {EvidenceCategory.LOG_SEMANTIC, EvidenceCategory.TRACE_LOCALIZATION}
            and evidence.status is EvidenceStatus.SUPPORT
        }
        return bool(direct_confirming_categories)

    def _unsupported_service(
        self,
        ranked: Sequence[HypothesisEvaluation],
        candidates: Sequence[ServiceCandidate],
    ) -> str | None:
        strong_services = {
            candidate.service
            for candidate in candidates
            if candidate.strength is CandidateStrength.STRONG
        }
        if not strong_services:
            return None
        if any(
            evaluation.ranking.contradiction_count == 0
            for evaluation in ranked
        ):
            return None
        by_service = {
            service: tuple(
                evaluation
                for evaluation in ranked
                if evaluation.hypothesis.service == service
            )
            for service in strong_services
        }
        complete = {
            service: evaluations
            for service, evaluations in by_service.items()
            if {
                evaluation.hypothesis.failure_mode
                for evaluation in evaluations
            }
            == set(self.signatures.failure_modes)
        }
        if not complete:
            return None
        localization_keys = {
            service: (
                min(
                    evaluation.ranking.unexplained_strong_count
                    for evaluation in evaluations
                ),
                -max(
                    evaluation.ranking.explained_strong_count
                    for evaluation in evaluations
                ),
            )
            for service, evaluations in complete.items()
        }
        best_key = min(localization_keys.values())
        localized = tuple(
            sorted(
                service
                for service, key in localization_keys.items()
                if key == best_key
            )
        )
        if len(localized) != 1:
            return None
        service = localized[0]
        if all(
            evaluation.ranking.contradiction_count > 0
            for evaluation in complete[service]
        ):
            return service
        return None

    @staticmethod
    def _result(
        status: DiagnosisStatus,
        best: Hypothesis | None,
        ranked: Sequence[HypothesisEvaluation],
        candidates: Sequence[ServiceCandidate],
        planned: Sequence[PlannedQuery],
        api: TelemetryQueryAPI,
        history_start: int,
        localized_service: str | None = None,
        confirmation_query_count: int = 0,
    ) -> DiagnosisResult:
        ranked_tuple = tuple(ranked)
        explanation = (
            explain_unsupported_decision(localized_service, ranked_tuple)
            if status is DiagnosisStatus.UNSUPPORTED
            and localized_service is not None
            else explain_ranked_decision(ranked_tuple)
        )
        return DiagnosisResult(
            status,
            best,
            ranked_tuple,
            tuple(candidates),
            tuple(planned),
            api.query_history[history_start:],
            api.spent_budget,
            api.remaining_budget,
            localized_service or (best.service if best is not None else None),
            explanation,
            confirmation_query_count,
        )


def _topology_bootstrap_order(
    services: Sequence[str],
    dependency_edges: frozenset[tuple[str, str]],
) -> tuple[str, ...]:
    callers_by_dependency: dict[str, set[str]] = {
        service: set() for service in services
    }
    for caller, dependency in dependency_edges:
        callers_by_dependency.setdefault(dependency, set()).add(caller)

    def transitive_caller_count(service: str) -> int:
        visited: set[str] = set()
        pending = list(callers_by_dependency.get(service, ()))
        while pending:
            caller = pending.pop()
            if caller in visited:
                continue
            visited.add(caller)
            pending.extend(callers_by_dependency.get(caller, ()))
        return len(visited)

    return tuple(
        sorted(services, key=lambda service: (-transitive_caller_count(service), service))
    )
