from __future__ import annotations

import csv
import json
from datetime import timedelta

from src.evaluation import Ablation, RobustnessProfile, build_agent, fragmentation_for
from src.evaluation.benchmark import (
    BenchmarkRunner,
    BudgetCurveRow,
    FaultCase,
    budget_curve,
    enumerate_fault_cases,
    write_outputs,
)
from src.evaluation.models import (
    BenchmarkIncidentResult,
    summarize_healthy_results,
    summarize_results,
)
from src.core import BaselineStore
from src.rca import DiagnosisAgent, EvidenceCategory
from src.simulation import (
    FaultRequest,
    IncidentGenerator,
    ServiceConfig,
    SimulationConfig,
    benchmark_config,
)


def test_benchmark_enumerates_only_configured_valid_pairs() -> None:
    config = benchmark_config(fragmentation_for(RobustnessProfile.CLEAN))
    cases = enumerate_fault_cases(
        config,
        services=["postgres", "gateway"],
        failure_modes=["database_slowdown", "deployment_regression"],
    )

    assert cases == (
        FaultCase("gateway", "deployment_regression"),
        FaultCase("postgres", "database_slowdown"),
    )


def test_exact_pair_summary_requires_resolved_correct_pair() -> None:
    correct = BenchmarkIncidentResult(
        "CLEAN", "FULL_SYSTEM", 1, "db", "slow", "RESOLVED", "db", "slow",
        True, True, True, 10, 12, "FAILOVER", "VERIFIED", True, 1, None,
    )
    unresolved = BenchmarkIncidentResult(
        "CLEAN", "FULL_SYSTEM", 2, "db", "slow", "AMBIGUOUS", "db", "slow",
        False, False, False, 10, 10, None, None, False, 0, None,
    )

    summary = summarize_results(
        (correct, unresolved), profile="CLEAN", ablation="FULL_SYSTEM"
    )

    assert summary.exact_pair_accuracy == 0.5
    assert summary.resolved_rate == 0.5
    assert summary.false_resolution_count == 0


def test_benchmark_is_deterministic_and_full_system_matches_default_agent() -> None:
    config = benchmark_config(fragmentation_for(RobustnessProfile.CLEAN))
    case = (FaultCase("postgres", "database_slowdown"),)
    first = BenchmarkRunner(config, verification_steps=8).run((17,), case)
    second = BenchmarkRunner(config, verification_steps=8).run((17,), case)

    assert first == second

    incident = IncidentGenerator(config).generate(
        17, FaultRequest("postgres", "database_slowdown")
    )
    start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    from src.core import TelemetryQueryAPI

    default = DiagnosisAgent().diagnose(
        TelemetryQueryAPI(incident.telemetry, 17),
        incident.baseline,
        start,
        incident.observed_end_time,
    )
    full = build_agent(Ablation.FULL_SYSTEM).diagnose(
        TelemetryQueryAPI(incident.telemetry, 17),
        incident.baseline,
        start,
        incident.observed_end_time,
    )
    assert full == default


def test_healthy_controls_use_normal_pipeline_and_never_execute_actions() -> None:
    for profile in (RobustnessProfile.CLEAN, RobustnessProfile.DEFAULT):
        runner = BenchmarkRunner(
            benchmark_config(fragmentation_for(profile)),
            profile=profile.value,
            verification_steps=8,
        )

        results = runner.run_healthy((1, 3))
        summary = summarize_healthy_results(results, profile=profile.value)

        assert summary.healthy_case_count == 2
        assert summary.healthy_no_candidate_count == 2
        assert summary.healthy_no_candidate_rate == 1.0
        assert summary.healthy_action_count == 0
        assert summary.healthy_action_rate == 0.0
        assert summary.healthy_preserved_rate == 1.0


def test_each_robustness_profile_is_reproducible() -> None:
    for profile in RobustnessProfile:
        config = SimulationConfig(
            services=(ServiceConfig("worker", 100, 5, 30),),
            baseline_steps=3,
            incident_steps=4,
            fragmentation=fragmentation_for(profile),
        )
        generator = IncidentGenerator(config)
        first = generator.generate(5, FaultRequest("worker", "cpu_saturation"))
        second = generator.generate(5, FaultRequest("worker", "cpu_saturation"))

        assert first.ground_truth == second.ground_truth
        assert first.telemetry.metric_count == second.telemetry.metric_count
        assert first.telemetry.log_count == second.telemetry.log_count
        assert first.telemetry.span_count == second.telemetry.span_count


def test_budget_curve_reuses_identical_incident_keys() -> None:
    config = benchmark_config(fragmentation_for(RobustnessProfile.CLEAN))
    rows = budget_curve(
        config,
        (17,),
        (FaultCase("postgres", "database_slowdown"),),
        (8, 17),
        profile="CLEAN",
    )

    assert [row.budget for row in rows] == [8, 17]
    assert all(row.incident_count == 1 for row in rows)


def test_detector_and_evidence_ablations_remove_intended_support() -> None:
    config = benchmark_config(fragmentation_for(RobustnessProfile.CLEAN))
    incident = IncidentGenerator(config).generate(
        17, FaultRequest("postgres", "database_slowdown")
    )
    start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    from src.core import TelemetryQueryAPI

    no_mad = build_agent(Ablation.NO_MAD).diagnose(
        TelemetryQueryAPI(incident.telemetry, 17),
        incident.baseline,
        start,
        incident.observed_end_time,
    )
    no_trace = build_agent(Ablation.NO_TRACE_EVIDENCE).diagnose(
        TelemetryQueryAPI(incident.telemetry, 17),
        incident.baseline,
        start,
        incident.observed_end_time,
    )

    assert all("mad" not in candidate.fired_detectors for candidate in no_mad.candidates)
    assert all(
        not any(
            evidence.category is EvidenceCategory.TRACE_LOCALIZATION
            and evidence.status.value == "SUPPORT"
            for evidence in evaluation.evidence
        )
        for evaluation in no_trace.ranked_hypotheses
    )


def test_all_ablation_injection_points_are_explicitly_disabled() -> None:
    baseline = BaselineStore.from_metrics({"service"}, ())
    assert build_agent(Ablation.NO_MAD)._mad_analyze("service", (), baseline) is None
    assert build_agent(Ablation.NO_CUSUM)._cusum_analyze("service", (), baseline) is None
    assert (
        build_agent(Ablation.NO_ISOLATION_FOREST)._isolation_analyze(
            "service", (), baseline
        )
        is None
    )
    assert build_agent(Ablation.NO_SEMANTIC_LOG_EVIDENCE)._log_extractor.extract(()) == ()
    metrics_only = build_agent(Ablation.METRICS_ONLY)
    assert metrics_only._log_extractor.extract(()) == ()
    assert metrics_only._trace_analyzer.reconstruct((), baseline).edges == ()


def test_benchmark_agent_never_receives_ground_truth() -> None:
    class InspectingAgent(DiagnosisAgent):
        calls = 0

        def diagnose(self, api, baseline, start_time, end_time, intervention_contradictions=None):
            self.calls += 1
            assert not hasattr(api, "ground_truth")
            assert not hasattr(baseline, "ground_truth")
            return super().diagnose(
                api,
                baseline,
                start_time,
                end_time,
                intervention_contradictions,
            )

    agent = InspectingAgent()
    config = benchmark_config(fragmentation_for(RobustnessProfile.CLEAN))
    BenchmarkRunner(
        config,
        verification_steps=8,
        agent_factory=lambda: agent,
    ).run((17,), (FaultCase("postgres", "database_slowdown"),))

    assert agent.calls == 1


def test_output_csv_and_json_schemas(tmp_path) -> None:
    result = BenchmarkIncidentResult(
        "CLEAN", "FULL_SYSTEM", 1, "db", "slow", "RESOLVED", "db", "slow",
        True, True, True, 10, 12, "FAILOVER", "VERIFIED", True, 1, None,
    )
    summary = summarize_results((result,), profile="CLEAN", ablation="FULL_SYSTEM")
    curve = BudgetCurveRow(17, 1, 1.0, 1.0, 0.0, 12.0)

    write_outputs(tmp_path, (result,), (summary,), (summary,), (curve,))

    with (tmp_path / "benchmark_results.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    payload = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert set(row) == set(result.as_row())
    assert payload["summaries"][0]["incident_count"] == 1
    assert (tmp_path / "ablation_results.csv").is_file()
    assert (tmp_path / "budget_curve.csv").is_file()
