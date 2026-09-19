"""Explicit dependency-injected evidence ablations for evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from src.core import BaselineStore, LogEvent, MetricEvent, SpanEvent
from src.rca import (
    CUSUMServiceEvidence,
    DependencyGraph,
    DiagnosisAgent,
    FailureSignatureLibrary,
    Hypothesis,
    HypothesisEvaluation,
    HypothesisEvaluator,
    IsolationForestEvidence,
    LogEvidence,
    MADServiceEvidence,
    ServiceCandidate,
    TraceGraphAnalyzer,
)


class Ablation(str, Enum):
    FULL_SYSTEM = "FULL_SYSTEM"
    NO_SEMANTIC_LOG_EVIDENCE = "NO_SEMANTIC_LOG_EVIDENCE"
    NO_TRACE_EVIDENCE = "NO_TRACE_EVIDENCE"
    NO_ISOLATION_FOREST = "NO_ISOLATION_FOREST"
    NO_CUSUM = "NO_CUSUM"
    NO_MAD = "NO_MAD"
    METRICS_ONLY = "METRICS_ONLY"


class _NoLogEvidence:
    def extract(self, logs: Sequence[LogEvent]) -> tuple[LogEvidence, ...]:
        return ()


class _NoTraceGraphAnalyzer(TraceGraphAnalyzer):
    def reconstruct(
        self,
        spans: Sequence[SpanEvent],
        baseline: BaselineStore | None = None,
    ) -> DependencyGraph:
        return DependencyGraph(frozenset(), (), {})


class _NoTraceHypothesisEvaluator(HypothesisEvaluator):
    def evaluate_all(
        self,
        hypotheses: Sequence[Hypothesis],
        candidates: Sequence[ServiceCandidate],
        metrics_by_service: Mapping[str, Sequence[MetricEvent]],
        logs: Sequence[LogEvidence],
        spans: Sequence[SpanEvent],
        dependency_edges: set[tuple[str, str]] | frozenset[tuple[str, str]],
        baseline: BaselineStore,
        intervention_contradictions: Mapping[tuple[str, str], str] | None = None,
    ) -> tuple[HypothesisEvaluation, ...]:
        return super().evaluate_all(
            hypotheses,
            candidates,
            metrics_by_service,
            logs,
            (),
            dependency_edges,
            baseline,
            intervention_contradictions,
        )


def _no_mad(
    service: str,
    metrics: Sequence[MetricEvent],
    baseline: BaselineStore,
) -> MADServiceEvidence | None:
    return None


def _no_cusum(
    service: str,
    metrics: Sequence[MetricEvent],
    baseline: BaselineStore,
) -> CUSUMServiceEvidence | None:
    return None


def _no_isolation(
    service: str,
    metrics: Sequence[MetricEvent],
    baseline: BaselineStore,
) -> IsolationForestEvidence | None:
    return None


def build_agent(ablation: Ablation) -> DiagnosisAgent:
    """Build an agent with explicit unavailable evidence sources for one ablation."""
    no_logs = ablation in {
        Ablation.NO_SEMANTIC_LOG_EVIDENCE,
        Ablation.METRICS_ONLY,
    }
    no_traces = ablation in {Ablation.NO_TRACE_EVIDENCE, Ablation.METRICS_ONLY}
    signatures = FailureSignatureLibrary()
    return DiagnosisAgent(
        signatures=signatures,
        log_extractor=_NoLogEvidence() if no_logs else None,
        mad_analyzer=_no_mad if ablation is Ablation.NO_MAD else None,
        cusum_analyzer=_no_cusum if ablation is Ablation.NO_CUSUM else None,
        isolation_analyzer=(
            _no_isolation if ablation is Ablation.NO_ISOLATION_FOREST else None
        ),
        trace_analyzer=_NoTraceGraphAnalyzer() if no_traces else None,
        hypothesis_evaluator=(
            _NoTraceHypothesisEvaluator(signatures) if no_traces else None
        ),
    )
