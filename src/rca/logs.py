"""Explainable structured and TF-IDF-based log evidence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.models import LogEvent


PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("resource_pressure", "cpu saturation high processor memory pressure resource exhausted queue overload"),
    ("timeout_network_delay", "request timeout timed out network latency slow connection delay unreachable"),
    ("connection_exhaustion", "connection pool exhausted too many connections socket unavailable"),
    ("process_unavailable", "process crash service unavailable stopped terminated health check failed"),
    ("deployment_regression", "deployment regression new release error exception request failure"),
    ("database_slowdown", "database storage query slow latency transaction lock disk"),
)


@dataclass(frozen=True, slots=True)
class LogEvidenceConfig:
    semantic_similarity_threshold: float = 0.12
    high_severities: frozenset[str] = frozenset({"ERROR", "CRITICAL"})


@dataclass(frozen=True, slots=True)
class LogEvidence:
    service: str
    severity: str
    message: str
    event_timestamp: datetime
    arrival_timestamp: datetime
    trace_id: str | None
    semantic_category: str | None
    similarity_score: float
    high_severity_corroborating: bool


class LogEvidenceExtractor:
    def __init__(self, config: LogEvidenceConfig | None = None) -> None:
        self.config = config or LogEvidenceConfig()
        self._categories = tuple(category for category, _ in PROTOTYPES)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self._prototype_matrix = self._vectorizer.fit_transform(
            [prototype for _, prototype in PROTOTYPES]
        )

    def extract(self, logs: Sequence[LogEvent]) -> tuple[LogEvidence, ...]:
        ordered_logs = sorted(
            logs, key=lambda log: (log.event_timestamp, log.service, log.message)
        )
        if not ordered_logs:
            return ()
        message_matrix = self._vectorizer.transform([log.message for log in ordered_logs])
        similarities = cosine_similarity(message_matrix, self._prototype_matrix)
        evidence: list[LogEvidence] = []
        for log, scores in zip(ordered_logs, similarities, strict=True):
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            category = (
                self._categories[best_index]
                if best_score >= self.config.semantic_similarity_threshold
                else None
            )
            evidence.append(
                LogEvidence(
                    service=log.service,
                    severity=log.severity,
                    message=log.message,
                    event_timestamp=log.event_timestamp,
                    arrival_timestamp=log.arrival_timestamp,
                    trace_id=log.trace_id,
                    semantic_category=category,
                    similarity_score=best_score,
                    high_severity_corroborating=log.severity
                    in self.config.high_severities,
                )
            )
        return tuple(evidence)
