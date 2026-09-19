"""Exact detector-count candidate generation and evidence-based promotion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .anomaly import (
    CUSUMServiceEvidence,
    IsolationForestEvidence,
    MADServiceEvidence,
)
from .graph import TraceServiceEvidence
from .logs import LogEvidence


class CandidateStrength(str, Enum):
    STRONG = "STRONG"
    WEAK = "WEAK"
    NOT_CANDIDATE = "NOT_CANDIDATE"


@dataclass(frozen=True, slots=True)
class ServiceCandidate:
    service: str
    strength: CandidateStrength
    fired_detectors: tuple[str, ...]
    detector_count: int
    promoted: bool
    corroborating_reasons: tuple[str, ...]


class CandidateGenerator:
    def generate(
        self,
        services: Sequence[str] | frozenset[str] | set[str],
        mad_results: Mapping[str, MADServiceEvidence | None],
        change_results: Mapping[str, CUSUMServiceEvidence | None],
        isolation_results: Mapping[str, IsolationForestEvidence | None],
        trace_evidence: Mapping[str, TraceServiceEvidence],
        log_evidence: Sequence[LogEvidence],
    ) -> tuple[ServiceCandidate, ...]:
        high_severity_logs: dict[str, list[LogEvidence]] = {}
        for evidence in log_evidence:
            if evidence.high_severity_corroborating:
                high_severity_logs.setdefault(evidence.service, []).append(evidence)

        candidates: list[ServiceCandidate] = []
        for service in sorted(services):
            fired = tuple(
                name
                for name, result in (
                    ("mad", mad_results.get(service)),
                    ("change_point", change_results.get(service)),
                    ("isolation_forest", isolation_results.get(service)),
                )
                if result is not None and result.flagged
            )
            reasons = self._corroboration_reasons(
                service, trace_evidence.get(service), high_severity_logs.get(service, [])
            )
            detector_count = len(fired)
            promoted = detector_count == 1 and bool(reasons)
            if detector_count >= 2 or promoted:
                strength = CandidateStrength.STRONG
            elif detector_count == 1:
                strength = CandidateStrength.WEAK
            else:
                strength = CandidateStrength.NOT_CANDIDATE
            candidates.append(
                ServiceCandidate(
                    service,
                    strength,
                    fired,
                    detector_count,
                    promoted,
                    reasons,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _corroboration_reasons(
        service: str,
        trace: TraceServiceEvidence | None,
        logs: Sequence[LogEvidence],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if trace is not None and trace.corroborates_abnormality:
            reasons.append(
                "trace abnormality: "
                f"failed_fraction={trace.failed_span_fraction:.3f}, "
                f"latency_ratio={trace.latency_ratio}"
            )
        for log in logs:
            category = log.semantic_category or "uncategorized"
            reasons.append(
                f"high-severity log: {log.severity} {category} "
                f"similarity={log.similarity_score:.3f} message={log.message!r}"
            )
        return tuple(reasons)
