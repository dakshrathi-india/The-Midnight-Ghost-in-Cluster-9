from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core import (
    BaselineStore,
    MetricEvent,
    QueryCosts,
    SpanEvent,
    TelemetryQueryAPI,
    TelemetryStore,
)
from src.rca import (
    ActiveQueryPlanner,
    BootstrapStrategy,
    CandidateStrength,
    DiagnosisAgent,
    DiagnosisAgentConfig,
    DiagnosisConfig,
    DiagnosisStatus,
    EvidenceCategory,
    EvidenceStatus,
    FailureSignatureLibrary,
    Hypothesis,
    HypothesisEvaluation,
    HypothesisEvaluator,
    HypothesisGenerator,
    RankingComponents,
    ServiceCandidate,
    TraceLocalizationKind,
    TraceLocalizer,
    dependency_path,
    rank_evaluations,
)
from src.rca.logs import LogEvidence
from src.simulation import (
    FaultRequest,
    FragmentationConfig,
    IncidentGenerator,
    benchmark_config,
)


START = datetime(2025, 2, 1, tzinfo=timezone.utc)
MODES = {
    "cpu_saturation",
    "deployment_regression",
    "database_slowdown",
    "connection_exhaustion",
    "network_latency",
    "process_crash",
}


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
    duration_ms: float,
    status: str = "OK",
) -> SpanEvent:
    timestamp = START + timedelta(seconds=100)
    return SpanEvent(
        trace_id,
        span_id,
        parent_id,
        service,
        f"{service}.request",
        timestamp,
        timestamp,
        duration_ms,
        status,
    )


def _candidate(
    service: str, strength: CandidateStrength = CandidateStrength.STRONG
) -> ServiceCandidate:
    return ServiceCandidate(service, strength, ("mad", "change_point"), 2, False, ())


def _baseline(
    services: tuple[str, ...] = ("worker",),
    edges: set[tuple[str, str]] | None = None,
) -> BaselineStore:
    metrics = [
        _metric(index, service, metric_name, value, unit)
        for service in services
        for index, value in enumerate((9.0, 10.0, 11.0, 10.0, 9.5, 10.5))
        for metric_name, unit in (
            ("request_latency_ms", "ms"),
            ("cpu_utilization", "ratio"),
            ("error_rate", "ratio"),
            ("queue_length", "count"),
        )
    ]
    return BaselineStore.from_metrics(set(services), metrics, edges)


def _temporal_evaluation(root_offset: int, caller_offset: int):
    baseline = _baseline(("root-node", "caller-node"), {("caller-node", "root-node")})
    metrics = {
        "root-node": [
            _metric(root_offset + index, "root-node", "request_latency_ms", 100.0, "ms")
            for index in range(3)
        ],
        "caller-node": [
            _metric(caller_offset + index, "caller-node", "request_latency_ms", 90.0, "ms")
            for index in range(3)
        ],
    }
    evaluator = HypothesisEvaluator(
        config=DiagnosisConfig(
            temporal_tolerance=timedelta(seconds=5),
            onset_consecutive_samples=1,
        )
    )
    hypothesis = Hypothesis(
        "root-node", "database_slowdown", CandidateStrength.STRONG
    )
    return evaluator.evaluate_all(
        [hypothesis],
        [_candidate("root-node"), _candidate("caller-node")],
        metrics,
        (),
        (),
        {("caller-node", "root-node")},
        baseline,
    )[0]


def _temporal_item(evaluation: HypothesisEvaluation):
    return next(
        item
        for item in evaluation.evidence
        if item.category is EvidenceCategory.TEMPORAL_PRECEDENCE
    )


def _controlled_fragmentation() -> FragmentationConfig:
    return FragmentationConfig(
        clock_skew_range_seconds=0,
        missing_observation_probability=0,
        delayed_observation_probability=0,
        metric_noise_fraction=0.02,
        include_decoy_anomaly=False,
    )


def _run_incident(
    service: str,
    failure_mode: str,
    fragmentation: FragmentationConfig | None = None,
):
    incident = IncidentGenerator(benchmark_config(fragmentation)).generate(
        17, FaultRequest(service, failure_mode)
    )
    start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    api = TelemetryQueryAPI(incident.telemetry, total_budget=17)
    result = DiagnosisAgent().diagnose(
        api, incident.baseline, start, incident.observed_end_time
    )
    return incident, result


def test_signature_library_contains_all_generic_failure_modes() -> None:
    library = FailureSignatureLibrary()

    assert set(library.failure_modes) == MODES
    assert all(library.get(mode).metric_expectations for mode in MODES)


def test_hypothesis_generation_uses_arbitrary_candidate_names() -> None:
    hypotheses = HypothesisGenerator().generate(
        [_candidate("tenant-specific-worker"), _candidate("quiet", CandidateStrength.NOT_CANDIDATE)]
    )

    assert {(item.service, item.failure_mode) for item in hypotheses} == {
        ("tenant-specific-worker", mode) for mode in MODES
    }


def test_dependency_paths_follow_caller_to_dependency_direction() -> None:
    edges = {("edge", "middle"), ("middle", "storage")}

    assert dependency_path("edge", "storage", edges) == (
        "edge",
        "middle",
        "storage",
    )
    assert dependency_path("storage", "edge", edges) is None


def test_unrelated_strong_candidate_is_preserved_as_unexplained() -> None:
    baseline = _baseline(("root-node", "unrelated"))
    hypothesis = Hypothesis("root-node", "database_slowdown", CandidateStrength.STRONG)

    evaluation = HypothesisEvaluator().evaluate_all(
        [hypothesis],
        [_candidate("root-node"), _candidate("unrelated")],
        {"root-node": (), "unrelated": ()},
        (),
        (),
        frozenset(),
        baseline,
    )[0]

    assert evaluation.explained_strong_candidates == ("root-node",)
    assert evaluation.unexplained_strong_candidates == ("unrelated",)


@pytest.mark.parametrize(
    ("root_offset", "caller_offset", "expected"),
    (
        (100, 120, EvidenceStatus.SUPPORT),
        (120, 100, EvidenceStatus.CONTRADICTION),
        (100, 103, EvidenceStatus.NEUTRAL),
    ),
)
def test_temporal_precedence_uses_clock_skew_tolerance(
    root_offset: int, caller_offset: int, expected: EvidenceStatus
) -> None:
    assert _temporal_item(_temporal_evaluation(root_offset, caller_offset)).status is expected


def test_service_onset_ignores_transient_normal_metric() -> None:
    baseline = _baseline(
        ("root-node", "caller-node"), {("caller-node", "root-node")}
    )
    root_metrics = [
        _metric(100, "root-node", "cpu_utilization", 100.0),
        *[
            _metric(offset, "root-node", "cpu_utilization", 10.0)
            for offset in range(101, 106)
        ],
        *[
            _metric(offset, "root-node", "request_latency_ms", 100.0, "ms")
            for offset in range(200, 203)
        ],
    ]
    caller_metrics = [
        _metric(offset, "caller-node", "request_latency_ms", 90.0, "ms")
        for offset in range(220, 223)
    ]
    evaluator = HypothesisEvaluator()
    patterns = evaluator.metric_patterns("root-node", root_metrics, baseline)

    evaluation = evaluator.evaluate_all(
        [Hypothesis("root-node", "database_slowdown", CandidateStrength.STRONG)],
        [_candidate("root-node"), _candidate("caller-node")],
        {"root-node": root_metrics, "caller-node": caller_metrics},
        (),
        (),
        {("caller-node", "root-node")},
        baseline,
    )[0]
    temporal = _temporal_item(evaluation)

    assert patterns["cpu_utilization"].direction.value == "NORMAL"
    assert patterns["cpu_utilization"].onset_time is None
    assert temporal.status is EvidenceStatus.SUPPORT
    assert f"root={(START + timedelta(seconds=200)).isoformat()}" in temporal.observation


@pytest.mark.parametrize(
    ("incident_value", "expected"),
    ((40.0, EvidenceStatus.SUPPORT), (-20.0, EvidenceStatus.CONTRADICTION)),
)
def test_metric_direction_supports_or_contradicts_signature(
    incident_value: float, expected: EvidenceStatus
) -> None:
    baseline = _baseline()
    hypothesis = Hypothesis("worker", "cpu_saturation", CandidateStrength.STRONG)
    metrics = [
        _metric(100 + index, "worker", "cpu_utilization", incident_value)
        for index in range(3)
    ]

    evaluation = HypothesisEvaluator().evaluate_all(
        [hypothesis], [_candidate("worker")], {"worker": metrics}, (), (), set(), baseline
    )[0]
    cpu = next(
        item
        for item in evaluation.evidence
        if item.category is EvidenceCategory.METRIC_PATTERN
        and item.observation.startswith("cpu_utilization=")
    )

    assert cpu.status is expected


def test_missing_metric_evidence_is_neutral() -> None:
    baseline = _baseline()
    hypothesis = Hypothesis("worker", "cpu_saturation", CandidateStrength.STRONG)

    evaluation = HypothesisEvaluator().evaluate_all(
        [hypothesis], [_candidate("worker")], {"worker": ()}, (), (), set(), baseline
    )[0]
    cpu = next(
        item
        for item in evaluation.evidence
        if item.category is EvidenceCategory.METRIC_PATTERN
        and item.observation.startswith("cpu_utilization=")
    )

    assert cpu.status is EvidenceStatus.NEUTRAL


def test_matching_early_log_category_supports_failure_mode() -> None:
    baseline = _baseline()
    hypothesis = Hypothesis("worker", "cpu_saturation", CandidateStrength.STRONG)
    timestamp = START + timedelta(seconds=100)
    log = LogEvidence(
        "worker",
        "ERROR",
        "CPU resource pressure",
        timestamp,
        timestamp,
        None,
        "resource_pressure",
        0.8,
        True,
    )

    evaluation = HypothesisEvaluator().evaluate_all(
        [hypothesis], [_candidate("worker")], {"worker": ()}, [log], (), set(), baseline
    )[0]

    assert any(
        item.category is EvidenceCategory.LOG_SEMANTIC
        and item.status is EvidenceStatus.SUPPORT
        for item in evaluation.evidence
    )


def test_unrelated_early_log_does_not_suppress_later_matching_category() -> None:
    baseline = _baseline()
    hypothesis = Hypothesis(
        "worker", "database_slowdown", CandidateStrength.STRONG
    )
    early_time = START + timedelta(seconds=100)
    later_time = START + timedelta(seconds=140)
    logs = (
        LogEvidence(
            "worker",
            "ERROR",
            "CPU resource pressure",
            early_time,
            early_time,
            None,
            "resource_pressure",
            0.9,
            True,
        ),
        LogEvidence(
            "worker",
            "ERROR",
            "database operations exceeded normal latency",
            later_time,
            later_time,
            None,
            "database_slowdown",
            0.8,
            True,
        ),
    )

    evaluation = HypothesisEvaluator().evaluate_all(
        [hypothesis], [_candidate("worker")], {"worker": ()}, logs, (), set(), baseline
    )[0]
    evidence = next(
        item
        for item in evaluation.evidence
        if item.category is EvidenceCategory.LOG_SEMANTIC
    )

    assert evidence.status is EvidenceStatus.SUPPORT
    assert evidence.observation.startswith("database_slowdown;")
    assert evidence.event_timestamp == later_time


def test_trace_localization_distinguishes_local_and_dependency_latency() -> None:
    baseline = _baseline(("front", "storage"), {("front", "storage")})
    local = TraceLocalizer().localize(
        "storage", [_span("local", "root", None, "storage", 100)], set(), baseline
    )
    dependency = TraceLocalizer().localize(
        "front",
        [
            _span("nested", "parent", None, "front", 100),
            _span("nested", "child", "parent", "storage", 90),
        ],
        {("front", "storage")},
        baseline,
    )

    assert local.kind is TraceLocalizationKind.LOCAL
    assert dependency.kind is TraceLocalizationKind.DEPENDENCY
    assert dependency.dependency_service == "storage"
    assert dependency.mean_dependency_fraction_lower_bound == pytest.approx(0.9)
    assert dependency.mean_dependency_fraction_upper_bound == pytest.approx(0.9)
    assert dependency.mean_local_duration_ms == pytest.approx(10)


def test_service_seeded_trace_query_enables_dependency_localization() -> None:
    baseline = _baseline(
        ("front", "worker", "datastore"),
        {("front", "worker"), ("worker", "datastore")},
    )
    store = TelemetryStore(
        (),
        (),
        (
            _span("nested", "front-span", None, "front", 120),
            _span("nested", "worker-span", "front-span", "worker", 100),
            _span(
                "nested",
                "datastore-span",
                "worker-span",
                "datastore",
                90,
            ),
        ),
    )
    api = TelemetryQueryAPI(store, total_budget=1)

    spans = api.query_traces(
        "worker", START + timedelta(seconds=100), START + timedelta(seconds=100)
    )
    localization = TraceLocalizer().localize(
        "worker", spans, {("worker", "datastore")}, baseline
    )

    assert {span.service for span in spans} == {"front", "worker", "datastore"}
    assert localization.kind is TraceLocalizationKind.DEPENDENCY
    assert localization.dependency_service == "datastore"
    assert api.spent_budget == 1


def test_trace_localization_identifies_mostly_local_parent_time() -> None:
    baseline = _baseline(("front", "storage"), {("front", "storage")})
    localization = TraceLocalizer().localize(
        "front",
        [
            _span("local-heavy", "parent", None, "front", 220),
            _span("local-heavy", "child", "parent", "storage", 20),
        ],
        {("front", "storage")},
        baseline,
    )

    assert localization.kind is TraceLocalizationKind.LOCAL
    assert localization.mean_dependency_duration_ms == pytest.approx(20)
    assert localization.mean_local_duration_ms == pytest.approx(200)


def test_trace_localization_handles_overlapping_children_conservatively() -> None:
    baseline = _baseline(
        ("front", "left-store", "right-store"),
        {("front", "left-store"), ("front", "right-store")},
    )
    localization = TraceLocalizer().localize(
        "front",
        [
            _span("overlap", "parent", None, "front", 100),
            _span("overlap", "left", "parent", "left-store", 40),
            _span("overlap", "right", "parent", "right-store", 40),
        ],
        {("front", "left-store"), ("front", "right-store")},
        baseline,
    )

    assert localization.kind is TraceLocalizationKind.MIXED
    assert localization.mean_dependency_fraction_lower_bound == pytest.approx(0.4)
    assert localization.mean_dependency_fraction_upper_bound == pytest.approx(0.8)


def test_incomplete_trace_does_not_fabricate_localization() -> None:
    baseline = _baseline(("front", "storage"), {("front", "storage")})

    localization = TraceLocalizer().localize(
        "front",
        [_span("orphaned", "parent", None, "front", 100)],
        {("front", "storage")},
        baseline,
    )

    assert localization.kind is TraceLocalizationKind.UNKNOWN
    assert localization.complete_parent_child_count == 0


def test_trace_causality_survives_reversed_cross_service_timestamps() -> None:
    baseline = _baseline(("front", "storage"), {("front", "storage")})
    parent_start = START + timedelta(seconds=120)
    child_start = START + timedelta(seconds=110)
    spans = (
        SpanEvent(
            "skewed",
            "parent",
            None,
            "front",
            "front.request",
            parent_start,
            parent_start,
            220,
            "OK",
        ),
        SpanEvent(
            "skewed",
            "child",
            "parent",
            "storage",
            "storage.request",
            child_start,
            child_start,
            190,
            "OK",
        ),
    )
    metrics = {
        "front": [
            _metric(100 + index, "front", "request_latency_ms", 100, "ms")
            for index in range(3)
        ],
        "storage": [
            _metric(120 + index, "storage", "request_latency_ms", 100, "ms")
            for index in range(3)
        ],
    }
    hypothesis = Hypothesis(
        "storage", "database_slowdown", CandidateStrength.STRONG
    )
    evaluator = HypothesisEvaluator(
        config=DiagnosisConfig(
            temporal_tolerance=timedelta(seconds=5),
            onset_consecutive_samples=1,
        )
    )

    evaluation = evaluator.evaluate_all(
        [hypothesis],
        [_candidate("front"), _candidate("storage")],
        metrics,
        (),
        spans,
        {("front", "storage")},
        baseline,
    )[0]
    temporal = next(
        item
        for item in evaluation.evidence
        if item.category is EvidenceCategory.TEMPORAL_PRECEDENCE
    )

    assert temporal.status is EvidenceStatus.SUPPORT
    assert temporal.observation == "trace path=front -> storage"
    assert "synchronized clocks" in temporal.explanation


def _ranked_stub(
    service: str,
    contradictions: int,
    unexplained: int,
    explained: int,
) -> HypothesisEvaluation:
    return HypothesisEvaluation(
        Hypothesis(service, "network_latency", CandidateStrength.STRONG),
        (),
        tuple(f"explained-{index}" for index in range(explained)),
        tuple(f"unexplained-{index}" for index in range(unexplained)),
        RankingComponents(contradictions, unexplained, explained, 1, 1),
    )


def test_ranking_prefers_fewer_contradictions() -> None:
    preferred = _ranked_stub("preferred", 0, 2, 1)
    other = _ranked_stub("other", 1, 0, 5)

    assert rank_evaluations([other, preferred])[0] is preferred


def test_ranking_prefers_explaining_more_strong_candidates() -> None:
    preferred = _ranked_stub("preferred", 0, 0, 3)
    other = _ranked_stub("other", 0, 1, 2)

    assert rank_evaluations([other, preferred])[0] is preferred


def _planner_api(costs: QueryCosts) -> TelemetryQueryAPI:
    return TelemetryQueryAPI(TelemetryStore((), (), ()), 20, costs)


def _planner_hypotheses() -> tuple[Hypothesis, ...]:
    return tuple(
        Hypothesis("worker", mode, CandidateStrength.STRONG)
        for mode in ("cpu_saturation", "database_slowdown", "process_crash")
    )


def test_planner_chooses_query_with_more_pair_separation() -> None:
    query = ActiveQueryPlanner().choose_next(
        _planner_hypotheses(),
        {"worker"},
        START,
        START + timedelta(minutes=5),
        _planner_api(QueryCosts()),
        set(),
    )

    assert query is not None
    assert query.query_type == "logs"
    assert query.pair_separation == 3


def test_planner_incorporates_configured_query_cost() -> None:
    query = ActiveQueryPlanner().choose_next(
        _planner_hypotheses(),
        {"worker"},
        START,
        START + timedelta(minutes=5),
        _planner_api(QueryCosts(metrics=1, logs=5, traces=5)),
        set(),
    )

    assert query is not None
    assert query.query_type == "metrics"
    assert query.cost == 1


def test_planner_is_deterministic() -> None:
    planner = ActiveQueryPlanner()
    arguments = (
        _planner_hypotheses(),
        {"worker"},
        START,
        START + timedelta(minutes=5),
        _planner_api(QueryCosts()),
        set(),
    )

    first = planner.choose_next(*arguments)
    second = planner.choose_next(*arguments)

    assert first == second


def _healthy_inputs(services: tuple[str, ...] = ("worker",)):
    baseline_events = [
        _metric(index, service, "cpu_utilization", 10.0 + (index % 3) * 0.1)
        for service in services
        for index in range(20)
    ]
    incident_events = [
        _metric(100 + index, service, "cpu_utilization", 10.0 + (index % 3) * 0.1)
        for service in services
        for index in range(12)
    ]
    return BaselineStore.from_metrics(set(services), baseline_events), TelemetryStore(
        incident_events, (), ()
    )


def test_agent_reuses_exact_cached_metric_query_without_extra_cost() -> None:
    baseline, store = _healthy_inputs()
    end = START + timedelta(seconds=200)
    api = TelemetryQueryAPI(store, total_budget=1)
    api.query_metrics("worker", START, end)

    result = DiagnosisAgent().diagnose(api, baseline, START, end)

    assert result.status is DiagnosisStatus.NO_CANDIDATES
    assert result.budget_spent == 1
    assert result.queries_executed[0].served_from_cache
    assert api.query_cost("metrics", "worker", START, end) == 0


def test_insufficient_bootstrap_budget_returns_incomplete() -> None:
    baseline, store = _healthy_inputs(("one", "two"))
    api = TelemetryQueryAPI(store, total_budget=1)

    result = DiagnosisAgent().diagnose(
        api, baseline, START, START + timedelta(seconds=200)
    )

    assert result.status is DiagnosisStatus.INCOMPLETE_BUDGET
    assert result.best_hypothesis is None
    assert result.budget_spent == 1


def test_topology_subset_bootstrap_is_explicit_and_deterministic() -> None:
    raw_baseline, store = _healthy_inputs(("entry", "middle", "store"))
    baseline = BaselineStore.from_metrics(
        {"entry", "middle", "store"},
        raw_baseline.metric_history,
        {("entry", "middle"), ("middle", "store")},
    )
    api = TelemetryQueryAPI(store, total_budget=2)
    agent = DiagnosisAgent(
        agent_config=DiagnosisAgentConfig(
            bootstrap_strategy=BootstrapStrategy.TOPOLOGY_METRIC_SUBSET
        )
    )

    result = agent.diagnose(
        api, baseline, START, START + timedelta(seconds=200)
    )

    assert result.status is DiagnosisStatus.INCOMPLETE_BUDGET
    assert result.best_hypothesis is None
    assert [entry.service for entry in result.queries_executed] == [
        "store",
        "middle",
    ]


def test_unaffordable_discriminating_query_returns_incomplete() -> None:
    baseline, _ = _healthy_inputs()
    anomalous = [
        _metric(100 + index, "worker", "cpu_utilization", 100.0)
        for index in range(12)
    ]
    api = TelemetryQueryAPI(
        TelemetryStore(anomalous, (), ()),
        total_budget=2,
        costs=QueryCosts(metrics=1, logs=3, traces=3),
    )

    result = DiagnosisAgent().diagnose(
        api, baseline, START, START + timedelta(seconds=200)
    )

    assert result.status is DiagnosisStatus.INCOMPLETE_BUDGET
    assert result.budget_spent == 1
    assert result.best_hypothesis is not None


def test_healthy_window_returns_no_candidates() -> None:
    baseline, store = _healthy_inputs()

    result = DiagnosisAgent().diagnose(
        TelemetryQueryAPI(store, total_budget=5),
        baseline,
        START,
        START + timedelta(seconds=200),
    )

    assert result.status is DiagnosisStatus.NO_CANDIDATES
    assert result.best_hypothesis is None


def test_database_slowdown_with_decoy_resolves_correct_pair_deterministically() -> None:
    first_incident, first = _run_incident("postgres", "database_slowdown")
    second_incident, second = _run_incident("postgres", "database_slowdown")

    expected = (
        first_incident.ground_truth.root_service,
        first_incident.ground_truth.failure_mode,
    )
    assert first.status is DiagnosisStatus.RESOLVED
    assert first.best_hypothesis is not None
    assert (first.best_hypothesis.service, first.best_hypothesis.failure_mode) == expected
    assert first_incident.ground_truth.decoy_services
    assert first.best_hypothesis.service not in first_incident.ground_truth.decoy_services
    assert first.budget_spent <= 17
    assert second_incident.ground_truth == first_incident.ground_truth
    assert second.best_hypothesis == first.best_hypothesis
    assert [query.key for query in second.planned_queries] == [
        query.key for query in first.planned_queries
    ]


@pytest.mark.parametrize(
    ("service", "failure_mode"),
    (("catalog", "cpu_saturation"), ("auth", "process_crash")),
)
def test_end_to_end_common_faults_resolve_correct_pair(
    service: str, failure_mode: str
) -> None:
    incident, result = _run_incident(
        service, failure_mode, _controlled_fragmentation()
    )

    assert result.status is DiagnosisStatus.RESOLVED
    assert result.best_hypothesis is not None
    assert (
        result.best_hypothesis.service,
        result.best_hypothesis.failure_mode,
    ) == (
        incident.ground_truth.root_service,
        incident.ground_truth.failure_mode,
    )
    assert result.budget_spent <= 17


def test_rca_has_no_weighted_final_score_or_forbidden_dependencies() -> None:
    rca_root = Path(__file__).parents[1] / "src" / "rca"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in rca_root.glob("*.py")
    ).lower()

    assert "weighted" not in source
    assert "src.simulation" not in source
    assert "ground_truth" not in source
    assert "except exception" not in source
    assert "bayesian" not in source
