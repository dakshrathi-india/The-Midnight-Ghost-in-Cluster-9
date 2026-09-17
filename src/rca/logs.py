"""Structured log facts and configurable semantic evidence extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.models import LogEvent


PROTOTYPES: dict[str, tuple[str, ...]] = {
    "resource_pressure": (
        "processor utilization is saturated and work is backing up",
        "compute resources are exhausted under heavy load",
        "memory or CPU pressure is limiting request processing",
    ),
    "timeout_network_delay": (
        "a remote call exceeded its deadline",
        "network communication is unusually slow or unreachable",
        "dependency responses are taking much longer than expected",
        "a request timed out while waiting for a remote dependency",
        "network round trip latency increased above normal",
    ),
    "connection_exhaustion": (
        "the connection pool has no free capacity",
        "all available client connections are in use",
        "new sockets cannot be allocated because the limit was reached",
    ),
    "process_unavailable": (
        "the service process stopped unexpectedly",
        "a worker crashed and is no longer available",
        "health checks fail because the application terminated",
    ),
    "deployment_regression": (
        "a recent release introduced exceptions and request failures",
        "errors increased after a deployment change",
        "new application code is failing during request handling",
    ),
    "database_slowdown": (
        "database queries are completing unusually slowly",
        "storage latency increased while transactions waited",
        "data access is delayed by locks or slow disk operations",
        "query execution is waiting behind a database transaction lock",
        "database operations exceed their normal response latency",
    ),
}

_NEGATED_FAILURE_PATTERN = re.compile(
    r"\b(?:"
    r"not\s+(?:exhausted|unavailable|failed|slow|saturated|crashed|terminated|timed\s+out)"
    r"|no\s+(?:errors?|failures?|timeouts?|latency|pressure)"
    r"|without\s+(?:errors?|failures?|timeouts?|delay)"
    r")\b",
    re.IGNORECASE,
)


class TextEmbedder(Protocol):
    """Minimal injectable interface for batch text embeddings."""

    @property
    def backend_name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazily load a real sentence-transformer inference backend."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None

    @property
    def backend_name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=max(1, min(64, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float)

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
            except OSError:
                model = SentenceTransformer(self.model_name)
            model.eval()
            self._model = model
        return self._model


@dataclass(frozen=True, slots=True)
class LogEvidenceConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_similarity_threshold: float = 0.42
    lexical_fallback_threshold: float = 0.20
    allow_lexical_fallback: bool = True
    high_severities: frozenset[str] = frozenset({"ERROR", "CRITICAL"})


@dataclass(frozen=True, slots=True)
class ExtractedLogFact:
    fact_type: str
    name: str
    value: float | int | str
    unit: str | None
    source_text: str


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
    extracted_facts: tuple[ExtractedLogFact, ...] = ()
    matched_prototype: str | None = None
    semantic_backend: str = "unknown"


class DeterministicLogFactExtractor:
    _duration_pattern = re.compile(
        r"\b(?P<name>timeout|took|duration|latency)\s*(?:=|:|was)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|milliseconds?|s|sec|seconds?)\b",
        re.IGNORECASE,
    )
    _percentage_pattern = re.compile(
        r"\b(?:(?P<name>cpu|memory|utilization|usage)\s*(?:=|:|was)?\s*)?"
        r"(?P<value>\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _count_pattern = re.compile(
        r"\b(?P<name>active|max|remaining|queue|backlog|retries|retry_count)"
        r"\s*[=:]\s*(?P<value>\d+)\b",
        re.IGNORECASE,
    )
    _code_pattern = re.compile(
        r"\b(?P<name>status|error|code)(?:\s+code)?\s*[=:]\s*"
        r"(?P<value>[A-Za-z]?\d{3,5}|[A-Z][A-Z0-9_-]+)\b"
    )

    def extract(self, message: str) -> tuple[ExtractedLogFact, ...]:
        positioned: list[tuple[int, ExtractedLogFact]] = []
        for match in self._duration_pattern.finditer(message):
            value = float(match.group("value"))
            source_unit = match.group("unit").lower()
            milliseconds = value * 1000.0 if source_unit.startswith("s") else value
            positioned.append(
                (
                    match.start(),
                    ExtractedLogFact(
                        "duration",
                        match.group("name").lower(),
                        _compact_number(milliseconds),
                        "ms",
                        match.group(0),
                    ),
                )
            )
        for match in self._percentage_pattern.finditer(message):
            positioned.append(
                (
                    match.start(),
                    ExtractedLogFact(
                        "percentage",
                        (match.group("name") or "percentage").lower(),
                        _compact_number(float(match.group("value"))),
                        "%",
                        match.group(0),
                    ),
                )
            )
        for match in self._count_pattern.finditer(message):
            positioned.append(
                (
                    match.start(),
                    ExtractedLogFact(
                        "count",
                        match.group("name").lower(),
                        int(match.group("value")),
                        "count",
                        match.group(0),
                    ),
                )
            )
        for match in self._code_pattern.finditer(message):
            positioned.append(
                (
                    match.start(),
                    ExtractedLogFact(
                        "code",
                        match.group("name").lower(),
                        match.group("value"),
                        None,
                        match.group(0),
                    ),
                )
            )
        return tuple(
            fact
            for _, fact in sorted(
                positioned,
                key=lambda item: (
                    item[0],
                    item[1].fact_type,
                    item[1].name,
                    str(item[1].value),
                ),
            )
        )


class LogEvidenceExtractor:
    def __init__(
        self,
        config: LogEvidenceConfig | None = None,
        embedder: TextEmbedder | None = None,
        fact_extractor: DeterministicLogFactExtractor | None = None,
    ) -> None:
        self.config = config or LogEvidenceConfig()
        self._embedder = embedder or SentenceTransformerEmbedder(
            self.config.embedding_model_name
        )
        self._fact_extractor = fact_extractor or DeterministicLogFactExtractor()
        self._prototype_categories = tuple(
            category for category, values in PROTOTYPES.items() for _ in values
        )
        self._prototype_texts = tuple(
            prototype for values in PROTOTYPES.values() for prototype in values
        )
        self._prototype_embeddings: np.ndarray | None = None
        self._lexical_vectorizer: TfidfVectorizer | None = None
        self._lexical_prototypes: object | None = None
        self._using_lexical_fallback = False

    def extract(self, logs: Sequence[LogEvent]) -> tuple[LogEvidence, ...]:
        ordered_logs = sorted(
            logs, key=lambda log: (log.event_timestamp, log.service, log.message)
        )
        if not ordered_logs:
            return ()
        messages = [log.message for log in ordered_logs]
        similarities, backend = self._similarities(messages)
        threshold = (
            self.config.lexical_fallback_threshold
            if self._using_lexical_fallback
            else self.config.semantic_similarity_threshold
        )
        evidence: list[LogEvidence] = []
        for log, scores in zip(ordered_logs, similarities, strict=True):
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            category = (
                self._prototype_categories[best_index]
                if best_score >= threshold
                and _NEGATED_FAILURE_PATTERN.search(log.message) is None
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
                    high_severity_corroborating=(
                        category is not None
                        and log.severity in self.config.high_severities
                    ),
                    extracted_facts=self._fact_extractor.extract(log.message),
                    matched_prototype=(
                        self._prototype_texts[best_index] if best_score > 0 else None
                    ),
                    semantic_backend=backend,
                )
            )
        return tuple(evidence)

    def _similarities(self, messages: Sequence[str]) -> tuple[np.ndarray, str]:
        if not self._using_lexical_fallback:
            try:
                if self._prototype_embeddings is None:
                    self._prototype_embeddings = _normalized(
                        self._embedder.encode(self._prototype_texts)
                    )
                message_embeddings = _normalized(self._embedder.encode(messages))
                return (
                    message_embeddings @ self._prototype_embeddings.T,
                    self._embedder.backend_name,
                )
            except (ImportError, OSError):
                if not self.config.allow_lexical_fallback:
                    raise
                self._using_lexical_fallback = True
        return self._lexical_similarities(messages), "tfidf-fallback"

    def _lexical_similarities(self, messages: Sequence[str]) -> np.ndarray:
        if self._lexical_vectorizer is None:
            self._lexical_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), lowercase=True
            )
            self._lexical_prototypes = self._lexical_vectorizer.fit_transform(
                self._prototype_texts
            )
        assert self._lexical_prototypes is not None
        message_matrix = self._lexical_vectorizer.transform(messages)
        return np.asarray(
            (message_matrix @ self._lexical_prototypes.T).toarray(), dtype=float
        )


def _normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("text embedder must return a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _compact_number(value: float) -> float | int:
    return int(value) if value.is_integer() else value
