from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from src.core import BaselineStore, LogEvent, MetricEvent, SpanEvent, TelemetryQueryAPI
from src.rca.anomaly import (
    CUSUMConfig,
    CUSUMDetector,
    CUSUMServiceEvidence,
    IsolationForestConfig,
    IsolationForestDetector,
    IsolationForestEvidence,
    MADDetector,
    MADServiceEvidence,
)
from src.rca.candidates import CandidateGenerator, CandidateStrength, ServiceCandidate
from src.rca.graph import TraceGraphAnalyzer, TraceServiceEvidence
from src.rca.logs import (
    LogEvidence,
    LogEvidenceConfig,
    LogEvidenceExtractor,
    SentenceTransformerEmbedder,
)
from src.simulation import (
    FaultRequest,
    FragmentationConfig,
    Incident,
    IncidentGenerator,
    benchmark_config,
)
from src.simulation.simulator import MicroserviceSimulator


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _metric(
    offset: int, service: str, name: str, value: float, unit: str = "ratio"
) -> MetricEvent:
    timestamp = START + timedelta(seconds=offset)
    return MetricEvent(timestamp, timestamp, service, name, value, unit)


def _span(
    trace_id: str,
    span_id: str,
    parent_id: str | None,
    service: str,
    duration_ms: float = 10,
    status: str = "OK",
) -> SpanEvent:
    return SpanEvent(
        trace_id,
        span_id,
        parent_id,
        service,
        f"{service}.request",
        START,
        START + timedelta(milliseconds=duration_ms),
        duration_ms,
        status,
    )


def _baseline(events: list[MetricEvent]) -> BaselineStore:
    services = {event.service for event in events}
    return BaselineStore.from_metrics(services, events)


def _candidates_from_evidence(
    services: Sequence[str],
    metrics_by_service: dict[str, Sequence[MetricEvent]],
    baseline: BaselineStore,
    spans: Sequence[SpanEvent],
    logs: Sequence[LogEvent],
) -> dict[str, ServiceCandidate]:
    mad = {
        service: MADDetector().analyze(service, metrics_by_service[service], baseline)
        for service in services
    }
    change = {
        service: CUSUMDetector().analyze(
            service, metrics_by_service[service], baseline
        )
        for service in services
    }
    isolation = {
        service: IsolationForestDetector().analyze(
            service, metrics_by_service[service], baseline
        )
        for service in services
    }
    graph = TraceGraphAnalyzer().reconstruct(spans, baseline)
    log_evidence = LogEvidenceExtractor().extract(logs)
    candidates = CandidateGenerator().generate(
        services, mad, change, isolation, graph.service_evidence, log_evidence
    )
    return {candidate.service: candidate for candidate in candidates}


def _analyze_incident(incident: Incident) -> dict[str, ServiceCandidate]:
    services = sorted(incident.baseline.known_services)
    analysis_start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    api = TelemetryQueryAPI(incident.telemetry, total_budget=len(services) * 3)
    metrics_by_service = {
        service: api.query_metrics(
            service, analysis_start, incident.observed_end_time
        )
        for service in services
    }
    spans_by_identity = {
        (span.trace_id, span.span_id): span
        for service in services
        for span in api.query_traces(
            service, analysis_start, incident.observed_end_time
        )
    }
    spans = tuple(
        sorted(
            spans_by_identity.values(),
            key=lambda span: (span.trace_id, span.start_timestamp, span.span_id),
        )
    )
    logs = tuple(
        log
        for service in services
        for log in api.query_logs(service, analysis_start, incident.observed_end_time)
    )
    return _candidates_from_evidence(
        services, metrics_by_service, incident.baseline, spans, logs
    )


def test_trace_graph_reconstructs_arbitrary_services_and_ignores_orphans() -> None:
    spans = (
        _span("trace-1", "root", None, "Public Edge"),
        _span("trace-1", "child", "root", "Ledger.V2"),
        _span("trace-1", "internal", "child", "Ledger.V2"),
        _span("trace-1", "orphan", "missing", "Detached Worker"),
        _span("trace-2", "root-2", None, "Public Edge"),
        _span("trace-2", "child-2", "root-2", "Ledger.V2"),
    )

    graph = TraceGraphAnalyzer().reconstruct(spans)

    assert graph.services == {"Public Edge", "Ledger.V2", "Detached Worker"}
    assert [
        (edge.caller_service, edge.callee_service, edge.support_count)
        for edge in graph.edges
    ] == [("Public Edge", "Ledger.V2", 2)]


def test_trace_evidence_preserves_failures_and_latency_corroboration() -> None:
    baseline = _baseline(
        [_metric(index, "worker", "request_latency_ms", 10, "ms") for index in range(6)]
    )
    spans = (
        _span("t1", "s1", None, "worker", 30, "ERROR"),
        _span("t2", "s2", None, "worker", 40, "OK"),
    )

    evidence = TraceGraphAnalyzer().reconstruct(spans, baseline).service_evidence["worker"]

    assert evidence.failed_span_count == 1
    assert evidence.failed_span_fraction == 0.5
    assert evidence.latency_ratio == 3.5
    assert evidence.corroborates_abnormality


def test_mad_detects_robust_anomaly_and_preserves_raw_evidence() -> None:
    baseline = _baseline(
        [_metric(index, "api", "latency", value) for index, value in enumerate([9, 10, 10, 11, 10])]
    )
    incident = [_metric(20, "api", "latency", 10), _metric(21, "api", "latency", 100)]

    result = MADDetector().analyze("api", incident, baseline)
    metric = result.metric_evidence["latency"]

    assert result.flagged
    assert metric.anomalous_count == 1
    assert len(metric.modified_z_scores) == 2
    assert metric.maximum_absolute_modified_z > 3.5
    assert metric.baseline_median == 10
    assert metric.dispersion_used > 0


def test_mad_zero_dispersion_fallback_is_deterministic() -> None:
    baseline = _baseline([_metric(index, "api", "errors", 0) for index in range(5)])
    incident = [_metric(10, "api", "errors", 1)]

    first = MADDetector().analyze("api", incident, baseline)
    second = MADDetector().analyze("api", incident, baseline)

    assert first == second
    assert first.flagged
    assert first.metric_evidence["errors"].dispersion_used > 0


def test_mad_does_not_promote_one_moderate_noisy_sample() -> None:
    baseline = _baseline(
        [_metric(index, "api", "latency", value) for index, value in enumerate([9, 10, 10, 11, 10])]
    )
    incident = [_metric(20 + index, "api", "latency", 10) for index in range(10)]
    incident[4] = _metric(24, "api", "latency", 14)

    result = MADDetector().analyze("api", incident, baseline)

    assert result.metric_evidence["latency"].anomalous_count == 1
    assert not result.metric_evidence["latency"].flagged
    assert not result.flagged


def test_cusum_detects_sustained_regime_shift_in_both_directions() -> None:
    healthy = [10, 11, 9, 10, 11, 9, 10, 10]
    baseline = _baseline(
        [_metric(index, "api", "load", value) for index, value in enumerate(healthy)]
        + [_metric(index, "api", "free_capacity", value) for index, value in enumerate(healthy)]
    )
    incident = [
        *[_metric(20 + index, "api", "load", 15) for index in range(6)],
        *[_metric(20 + index, "api", "free_capacity", 5) for index in range(6)],
    ]

    result = CUSUMDetector(CUSUMConfig(threshold=4, drift=0.25)).analyze(
        "api", incident, baseline
    )

    assert result.flagged
    assert result.metric_evidence["load"].direction == "upward"
    assert result.metric_evidence["free_capacity"].direction == "downward"
    assert result.metric_evidence["load"].maximum_positive_cusum >= 4
    assert result.metric_evidence["free_capacity"].maximum_negative_cusum >= 4
    assert len(result.metric_evidence["load"].positive_cusum_series) == 6
    assert len(result.metric_evidence["free_capacity"].negative_cusum_series) == 6


def test_cusum_does_not_flag_an_isolated_spike() -> None:
    baseline = _baseline(
        [_metric(index, "api", "load", value) for index, value in enumerate([9, 10, 11] * 4)]
    )
    incident = [_metric(20 + index, "api", "load", 10) for index in range(8)]
    incident[3] = _metric(23, "api", "load", 100)

    result = CUSUMDetector().analyze("api", incident, baseline)

    assert result.metric_evidence["load"].maximum_positive_cusum > 5
    assert result.metric_evidence["load"].longest_positive_run == 1
    assert not result.flagged


def test_isolation_forest_detects_multivariate_anomaly_deterministically() -> None:
    baseline_events: list[MetricEvent] = []
    for index in range(30):
        baseline_events.extend(
            (
                _metric(index, "api", "cpu_utilization", 0.30 + (index % 3) * 0.01),
                _metric(index, "api", "request_latency_ms", 20 + (index % 4), "ms"),
            )
        )
    baseline = _baseline(baseline_events)
    incident: list[MetricEvent] = []
    for index in range(8):
        incident.extend(
            (
                _metric(100 + index, "api", "cpu_utilization", 0.95),
                _metric(100 + index, "api", "request_latency_ms", 250, "ms"),
            )
        )
    config = IsolationForestConfig(random_state=23, minimum_anomalous_fraction=0.1)

    first = IsolationForestDetector(config).analyze("api", incident, baseline)
    second = IsolationForestDetector(config).analyze("api", incident, baseline)

    assert first == second
    assert first.flagged
    assert first.feature_names == ("cpu_utilization", "request_latency_ms")
    assert first.training_sample_count == 30
    assert first.incident_sample_count == 8
    assert first.anomalous_count > 0
    assert len(first.anomaly_scores) == 8
    assert first.maximum_anomaly_score == max(first.anomaly_scores)


def test_isolation_forest_fits_only_baseline_and_handles_incomplete_vectors() -> None:
    baseline_events: list[MetricEvent] = []
    for index in range(12):
        baseline_events.append(
            _metric(index, "svc", "cpu_utilization", 0.2 + index * 0.001)
        )
        if index % 2 == 0:
            baseline_events.append(
                _metric(index, "svc", "request_latency_ms", 10 + index, "ms")
            )
    baseline = _baseline(baseline_events)
    incident = [
        _metric(30, "svc", "cpu_utilization", 0.9),
        _metric(31, "svc", "request_latency_ms", 100, "ms"),
        _metric(32, "svc", "cpu_utilization", 0.95),
        _metric(32, "svc", "request_latency_ms", 120, "ms"),
    ]

    result = IsolationForestDetector().analyze("svc", incident, baseline)

    assert result.training_sample_count == 12
    assert result.incident_sample_count == 3
    assert result.feature_names == ("cpu_utilization", "request_latency_ms")


def test_isolation_forest_does_not_flag_service_for_small_anomalous_fraction() -> None:
    baseline_events: list[MetricEvent] = []
    incident: list[MetricEvent] = []
    for index in range(20):
        baseline_events.extend(
            (
                _metric(index, "api", "cpu_utilization", 0.3 + (index % 3) * 0.01),
                _metric(index, "api", "request_latency_ms", 20 + (index % 4), "ms"),
            )
        )
        incident.extend(
            (
                _metric(100 + index, "api", "cpu_utilization", 0.3 + (index % 3) * 0.01),
                _metric(100 + index, "api", "request_latency_ms", 20 + (index % 4), "ms"),
            )
        )
    incident[-2] = _metric(119, "api", "cpu_utilization", 0.99)
    incident[-1] = _metric(119, "api", "request_latency_ms", 500, "ms")

    result = IsolationForestDetector().analyze(
        "api", incident, _baseline(baseline_events)
    )

    assert result.anomalous_fraction < 0.5
    assert not result.flagged


def test_isolation_forest_ignores_unapproved_metrics() -> None:
    baseline_events = [
        event
        for index in range(8)
        for event in (
            _metric(index, "svc", "cpu_utilization", 0.3),
            _metric(index, "svc", "vendor_magic_score", 10 + index),
        )
    ]
    incident = [
        event
        for index in range(5)
        for event in (
            _metric(20 + index, "svc", "cpu_utilization", 0.8),
            _metric(20 + index, "svc", "vendor_magic_score", 1000),
        )
    ]

    result = IsolationForestDetector().analyze(
        "svc", incident, _baseline(baseline_events)
    )

    assert result.feature_names == ("cpu_utilization",)


def test_isolation_forest_optional_queue_feature_can_be_enabled() -> None:
    baseline_events = [
        event
        for index in range(8)
        for event in (
            _metric(index, "svc", "cpu_utilization", 0.3),
            _metric(index, "svc", "queue_length", index, "requests"),
        )
    ]
    incident = [
        event
        for index in range(5)
        for event in (
            _metric(20 + index, "svc", "cpu_utilization", 0.8),
            _metric(20 + index, "svc", "queue_length", 100, "requests"),
        )
    ]
    detector = IsolationForestDetector(
        IsolationForestConfig(optional_features=("queue_length",))
    )

    result = detector.analyze("svc", incident, _baseline(baseline_events))

    assert result.feature_names == ("cpu_utilization", "queue_length")


def test_isolation_forest_missing_optional_feature_is_safe() -> None:
    baseline_events = [
        event
        for index in range(8)
        for event in (
            _metric(index, "svc", "cpu_utilization", 0.3),
            _metric(index, "svc", "queue_length", index, "requests"),
        )
    ]
    incident = [
        _metric(20 + index, "svc", "cpu_utilization", 0.8)
        for index in range(5)
    ]
    detector = IsolationForestDetector(
        IsolationForestConfig(optional_features=("queue_length",))
    )

    result = detector.analyze("svc", incident, _baseline(baseline_events))

    assert result.feature_names == ("cpu_utilization",)
    assert result.incident_sample_count == 5


def _mad_signal(service: str, flagged: bool) -> MADServiceEvidence:
    return MADServiceEvidence(service, flagged, {})


def _change_signal(service: str, flagged: bool) -> CUSUMServiceEvidence:
    return CUSUMServiceEvidence(service, flagged, {})


def _isolation_signal(service: str, flagged: bool) -> IsolationForestEvidence:
    return IsolationForestEvidence(service, flagged, (), 0, 0, (), 0, 0.0, 0.0)


def _trace_signal(service: str, corroborates: bool) -> TraceServiceEvidence:
    return TraceServiceEvidence(service, 2, 1, 0.5, 20, 30, 10, 2, corroborates)


def _log_signal(service: str) -> LogEvidence:
    return LogEvidence(
        service,
        "ERROR",
        "request timed out",
        START,
        START,
        None,
        "timeout_network_delay",
        0.8,
        True,
    )


class _DeterministicTestEmbedder:
    backend_name = "deterministic-test-embedder"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vectors: list[list[float]] = []
        concept_terms = (
            {"processor", "compute", "cpu", "memory", "resources"},
            {
                "remote",
                "network",
                "communication",
                "dependency",
                "deadline",
                "backend",
                "far",
            },
            {"connection", "pool", "client", "sockets", "capacity", "slots", "occupied"},
            {
                "process",
                "worker",
                "crashed",
                "stopped",
                "terminated",
                "health",
                "unexpectedly",
            },
            {"release", "deployment", "code", "exceptions"},
            {"database", "storage", "queries", "transactions", "data", "locks", "disk"},
        )
        for text in texts:
            lowered = text.lower()
            words = set(re.findall(r"[a-z]+", lowered))
            vectors.append(
                [float(len(words & terms)) for terms in concept_terms]
            )
        return np.asarray(vectors, dtype=float)


class _UnavailableEmbedder:
    backend_name = "unavailable-test-embedder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise OSError("embedding backend is unavailable")


@pytest.mark.parametrize(
    ("flags", "trace", "logs", "expected_strength", "promoted"),
    [
        ((True, True, False), False, (), CandidateStrength.STRONG, False),
        ((True, False, False), False, (), CandidateStrength.WEAK, False),
        ((True, False, False), True, (), CandidateStrength.STRONG, True),
        ((True, False, False), False, ("log",), CandidateStrength.STRONG, True),
        ((False, False, False), True, ("log",), CandidateStrength.NOT_CANDIDATE, False),
    ],
)
def test_candidate_generation_exact_rule(
    flags: tuple[bool, bool, bool],
    trace: bool,
    logs: tuple[str, ...],
    expected_strength: CandidateStrength,
    promoted: bool,
) -> None:
    service = "arbitrary-service"
    result = CandidateGenerator().generate(
        {service},
        {service: _mad_signal(service, flags[0])},
        {service: _change_signal(service, flags[1])},
        {service: _isolation_signal(service, flags[2])},
        {service: _trace_signal(service, trace)},
        [_log_signal(service)] if logs else [],
    )[0]

    assert result.strength is expected_strength
    assert result.detector_count == sum(flags)
    assert result.promoted is promoted
    if promoted:
        assert result.corroborating_reasons


def test_semantic_log_evidence_is_structured_and_explainable() -> None:
    event = LogEvent(
        START,
        START + timedelta(seconds=2),
        "custom-db",
        "ERROR",
        "database query is slow due to transaction lock",
        "trace-9",
    )

    evidence = LogEvidenceExtractor(
        LogEvidenceConfig(semantic_similarity_threshold=0.8),
        embedder=_DeterministicTestEmbedder(),
    ).extract([event])[0]

    assert evidence.service == "custom-db"
    assert evidence.semantic_category == "database_slowdown"
    assert evidence.similarity_score > 0
    assert evidence.high_severity_corroborating
    assert evidence.message == event.message
    assert evidence.trace_id == "trace-9"
    assert evidence.matched_prototype is not None
    assert evidence.semantic_backend == "deterministic-test-embedder"


@pytest.mark.parametrize(
    ("message", "expected_category"),
    (
        ("all client slots are occupied", "connection_exhaustion"),
        ("backend call is taking far longer than normal", "timeout_network_delay"),
        ("worker terminated unexpectedly", "process_unavailable"),
    ),
)
def test_embedding_semantics_classify_unseen_paraphrases(
    message: str, expected_category: str
) -> None:
    event = LogEvent(START, START, "worker", "ERROR", message)
    extractor = LogEvidenceExtractor(
        LogEvidenceConfig(semantic_similarity_threshold=0.8),
        embedder=_DeterministicTestEmbedder(),
    )

    evidence = extractor.extract([event])[0]

    assert evidence.semantic_category == expected_category
    assert evidence.similarity_score >= 0.8


def test_embedding_semantics_abstain_when_evidence_is_misleading() -> None:
    event = LogEvent(
        START,
        START,
        "worker",
        "INFO",
        "connection pool is not exhausted and routine checks are healthy",
    )
    extractor = LogEvidenceExtractor(
        LogEvidenceConfig(semantic_similarity_threshold=0.8),
        embedder=_DeterministicTestEmbedder(),
    )

    evidence = extractor.extract([event])[0]

    assert evidence.semantic_category is None
    assert not evidence.high_severity_corroborating


def test_prototype_embeddings_are_cached_between_batches() -> None:
    embedder = _DeterministicTestEmbedder()
    extractor = LogEvidenceExtractor(
        LogEvidenceConfig(semantic_similarity_threshold=0.8), embedder=embedder
    )
    event = LogEvent(START, START, "worker", "ERROR", "worker terminated unexpectedly")

    extractor.extract([event])
    extractor.extract([event])

    assert embedder.calls == 3


def test_sentence_transformer_backend_is_configurable_and_lazy() -> None:
    embedder = SentenceTransformerEmbedder("organization/test-model")

    assert embedder.backend_name == "sentence-transformers:organization/test-model"
    assert not embedder.is_loaded


def test_tfidf_fallback_is_reachable_when_embedding_backend_is_unavailable() -> None:
    event = LogEvent(
        START,
        START,
        "worker",
        "ERROR",
        "connection pool has no free capacity",
    )
    extractor = LogEvidenceExtractor(embedder=_UnavailableEmbedder())

    evidence = extractor.extract([event])[0]

    assert evidence.semantic_backend == "tfidf-fallback"
    assert evidence.semantic_category == "connection_exhaustion"
    assert evidence.high_severity_corroborating


def test_log_fact_extraction_preserves_high_confidence_values() -> None:
    message = (
        "status=503 timeout=5000ms took 2.4s cpu=97% "
        "active=100 max=100 remaining=0 queue=42 retries=3"
    )
    event = LogEvent(START, START, "worker", "ERROR", message)
    extractor = LogEvidenceExtractor(
        LogEvidenceConfig(semantic_similarity_threshold=0.8),
        embedder=_DeterministicTestEmbedder(),
    )

    evidence = extractor.extract([event])[0]
    facts = {(fact.fact_type, fact.name, fact.value, fact.unit) for fact in evidence.extracted_facts}

    assert evidence.message == message
    assert ("code", "status", "503", None) in facts
    assert ("duration", "timeout", 5000, "ms") in facts
    assert ("duration", "took", 2400, "ms") in facts
    assert ("percentage", "cpu", 97, "%") in facts
    assert ("count", "active", 100, "count") in facts
    assert ("count", "remaining", 0, "count") in facts
    assert ("count", "queue", 42, "count") in facts
    assert ("count", "retries", 3, "count") in facts


def test_end_to_end_incident_produces_meaningful_candidates_through_query_api() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0,
        include_decoy_anomaly=False,
    )
    incident = IncidentGenerator(benchmark_config(fragmentation)).generate(
        31, FaultRequest("postgres", "database_slowdown")
    )
    candidates = _analyze_incident(incident)

    assert any(
        candidate.strength is CandidateStrength.STRONG
        for candidate in candidates.values()
    )
    assert any(candidate.detector_count > 0 for candidate in candidates.values())


def test_effectively_healthy_continuation_has_low_candidate_rate() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0,
        include_decoy_anomaly=False,
    )
    config = benchmark_config(fragmentation)
    raw = MicroserviceSimulator(config, seed=17).run(
        START,
        fault_start_step=10_000,
        root_service="postgres",
        failure_mode="database_slowdown",
        fault_parameters={},
    )
    split = START + timedelta(seconds=config.baseline_steps * config.step_seconds)
    baseline = BaselineStore.from_metrics(
        set(config.service_map),
        [event for event in raw.metrics if event.event_timestamp < split],
    )
    services = sorted(baseline.known_services)
    metrics_by_service = {
        service: tuple(
            event
            for event in raw.metrics
            if event.service == service and event.event_timestamp >= split
        )
        for service in services
    }
    candidates = _candidates_from_evidence(
        services,
        metrics_by_service,
        baseline,
        [span for span in raw.spans if span.start_timestamp >= split],
        [log for log in raw.logs if log.event_timestamp >= split],
    )

    strong = [
        candidate
        for candidate in candidates.values()
        if candidate.strength is CandidateStrength.STRONG
    ]
    assert not strong
    assert sum(
        candidate.strength is not CandidateStrength.NOT_CANDIDATE
        for candidate in candidates.values()
    ) <= 1


def test_postgres_slowdown_selects_root_and_upstream_without_branch_explosion() -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0.02,
        include_decoy_anomaly=False,
    )
    incident = IncidentGenerator(benchmark_config(fragmentation)).generate(
        17, FaultRequest("postgres", "database_slowdown")
    )
    candidates = _analyze_incident(incident)
    strong_services = {
        service
        for service, candidate in candidates.items()
        if candidate.strength is CandidateStrength.STRONG
    }

    assert candidates[incident.ground_truth.root_service].strength is CandidateStrength.STRONG
    assert len(strong_services & incident.ground_truth.affected_services) >= 3
    unrelated_branch = {"auth", "redis", "inventory", "catalog"}
    assert not (strong_services & unrelated_branch)


@pytest.mark.parametrize(
    ("service", "failure_mode"),
    (("catalog", "cpu_saturation"), ("auth", "process_crash")),
)
def test_clear_fault_marks_injected_service_strong(
    service: str, failure_mode: str
) -> None:
    fragmentation = FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0.02,
        include_decoy_anomaly=False,
    )
    incident = IncidentGenerator(benchmark_config(fragmentation)).generate(
        17, FaultRequest(service, failure_mode)
    )
    candidates = _analyze_incident(incident)

    assert candidates[incident.ground_truth.root_service].strength is CandidateStrength.STRONG


def test_rca_package_has_no_simulation_or_ground_truth_dependency() -> None:
    rca_root = Path(__file__).parents[1] / "src" / "rca"
    source = "\n".join(path.read_text(encoding="utf-8") for path in rca_root.glob("*.py"))

    assert "src.simulation" not in source
    assert "ground_truth" not in source
    for benchmark_service in (
        "gateway",
        "auth",
        "checkout",
        "payment",
        "orders",
        "inventory",
        "catalog",
        "redis",
        "postgres",
    ):
        assert benchmark_service not in source
