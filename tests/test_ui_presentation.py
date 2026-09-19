from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.core import MetricEvent, TelemetryQueryAPI, TelemetryStore
from src.rca import (
    CandidateStrength,
    CausalEvidence,
    DiagnosisResult,
    DiagnosisStatus,
    EvidenceCategory,
    EvidenceStatus,
    Hypothesis,
    HypothesisEvaluation,
    RankingComponents,
    ServiceCandidate,
)
from src.simulation import ServiceConfig, SimulationConfig
from src.ui.presentation import (
    EVALUATION_TRUTH_LABEL,
    candidate_rows,
    evidence_rows,
    hypothesis_rows,
    load_benchmark_artifacts,
    prominent_evidence,
    query_history_rows,
    service_failure_mode_options,
    telemetry_signal_rows,
)


def _evaluation() -> HypothesisEvaluation:
    hypothesis = Hypothesis("db", "database_slowdown", CandidateStrength.STRONG)
    evidence = CausalEvidence(
        EvidenceCategory.METRIC_PATTERN,
        "db",
        "database_slowdown",
        "request_latency_ms is HIGH",
        EvidenceStatus.SUPPORT,
        "required direction matched",
    )
    return HypothesisEvaluation(
        hypothesis,
        (evidence,),
        ("api", "db"),
        (),
        RankingComponents(0, 0, 2, 3, 1),
    )


def _diagnosis() -> DiagnosisResult:
    evaluation = _evaluation()
    return DiagnosisResult(
        DiagnosisStatus.RESOLVED,
        evaluation.hypothesis,
        (evaluation,),
        (),
        (),
        (),
        4,
        13,
    )


def test_service_failure_modes_are_read_from_config() -> None:
    config = SimulationConfig(
        services=(
            ServiceConfig(
                "custom-db",
                100,
                5,
                valid_failure_modes=frozenset(
                    {"database_slowdown", "process_crash"}
                ),
            ),
        )
    )

    assert service_failure_mode_options(config) == {
        "custom-db": ("database_slowdown", "process_crash")
    }


def test_candidate_table_uses_domain_results() -> None:
    diagnosis = _diagnosis()
    assert candidate_rows(diagnosis) == ()


def test_candidate_table_exposes_each_detector_without_long_reasons() -> None:
    base = _diagnosis()
    candidate = ServiceCandidate(
        "db",
        CandidateStrength.STRONG,
        ("mad", "isolation_forest"),
        2,
        False,
        ("high-severity log",),
    )
    diagnosis = DiagnosisResult(
        base.status,
        base.best_hypothesis,
        base.ranked_hypotheses,
        (candidate,),
        base.planned_queries,
        base.queries_executed,
        base.budget_spent,
        base.budget_remaining,
    )

    assert candidate_rows(diagnosis) == (
        {
            "Service": "db",
            "Strength": "STRONG",
            "MAD": "●",
            "CUSUM": "—",
            "IF": "●",
            "Corroborated": True,
        },
    )


def test_hypothesis_table_preserves_exact_five_ranking_components() -> None:
    row = hypothesis_rows(_diagnosis())[0]

    assert row == {
        "#": 1,
        "Service": "db",
        "Failure Mode": "database_slowdown",
        "Contradictions": 0,
        "Unexplained": 0,
        "Explained": 2,
        "Support Types": 3,
        "Direct Support": 1,
    }
    assert "score" not in " ".join(row)


def test_query_budget_table_uses_real_api_history() -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = MetricEvent(timestamp, timestamp, "db", "cpu_utilization", 0.5, "ratio")
    api = TelemetryQueryAPI(TelemetryStore((event,), (), ()), total_budget=2)

    api.query_metrics("db", timestamp, timestamp)
    api.query_metrics("db", timestamp, timestamp)
    rows = query_history_rows(api.query_history)

    assert [row["Cost"] for row in rows] == [1, 0]
    assert [row["Cache"] for row in rows] == [False, True]
    assert [row["Cumulative"] for row in rows] == [1, 1]


def test_benchmark_loader_handles_missing_files(tmp_path) -> None:
    artifacts = load_benchmark_artifacts(tmp_path)

    assert artifacts.summaries == ()
    assert artifacts.incidents == ()
    assert set(artifacts.missing_files) == {
        "benchmark_summary.json",
        "benchmark_results.csv",
        "ablation_results.csv",
        "budget_curve.csv",
    }


def test_benchmark_loader_reads_json_and_csv(tmp_path) -> None:
    (tmp_path / "benchmark_summary.json").write_text(
        json.dumps({"summaries": [{"profile": "CLEAN", "resolved_rate": 1.0}]}),
        encoding="utf-8",
    )
    (tmp_path / "benchmark_results.csv").write_text(
        "profile,seed,exact_pair_correct\nCLEAN,17,True\n",
        encoding="utf-8",
    )
    (tmp_path / "ablation_results.csv").write_text(
        "profile,ablation,exact_pair_accuracy\nCLEAN,FULL_SYSTEM,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "budget_curve.csv").write_text(
        "budget,resolved_rate\n17,1.0\n",
        encoding="utf-8",
    )

    artifacts = load_benchmark_artifacts(tmp_path)

    assert artifacts.summaries[0]["profile"] == "CLEAN"
    assert artifacts.incidents[0]["seed"] == 17
    assert artifacts.incidents[0]["exact_pair_correct"] is True
    assert artifacts.ablations[0]["exact_pair_accuracy"] == 1.0
    assert artifacts.budget_curve[0]["budget"] == 17
    assert artifacts.missing_files == ()


def test_evidence_table_cannot_expose_ground_truth() -> None:
    rows = evidence_rows(_evaluation())

    assert rows[0]["Verdict"] == "SUPPORT"
    assert rows[0]["Evidence"] == "Metric"
    assert tuple(rows[0]) == ("Evidence", "Service", "Observation", "Verdict")
    assert not any("truth" in key.lower() for key in rows[0])
    assert EVALUATION_TRUTH_LABEL == "EVALUATION ONLY · HIDDEN GROUND TRUTH"


def test_prominent_evidence_never_hides_contradictions() -> None:
    evaluation = _evaluation()
    contradictions = tuple(
        CausalEvidence(
            EvidenceCategory.METRIC_PATTERN,
            "db",
            "database_slowdown",
            f"required metric {index} stayed normal",
            EvidenceStatus.CONTRADICTION,
            "required signal absent",
        )
        for index in range(6)
    )
    evaluation = HypothesisEvaluation(
        evaluation.hypothesis,
        (*evaluation.evidence, *contradictions),
        evaluation.explained_strong_candidates,
        evaluation.unexplained_strong_candidates,
        evaluation.ranking,
    )

    selected = prominent_evidence(evaluation)

    assert all(item in selected for item in contradictions)


def test_telemetry_signals_keep_real_periods_and_operational_metrics() -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = (
        MetricEvent(timestamp, timestamp, "db", "cpu_utilization", 0.7, "ratio"),
        MetricEvent(timestamp, timestamp, "db", "request_latency_ms", 125, "ms"),
        MetricEvent(timestamp, timestamp, "db", "queue_depth", 9, "count"),
    )

    rows = telemetry_signal_rows((("Incident", events),))

    assert rows == (
        {
            "Event time": timestamp,
            "Metric": "cpu_utilization",
            "Signal": "CPU utilization",
            "Value": 0.7,
            "Unit": "ratio",
            "Period": "Incident",
        },
        {
            "Event time": timestamp,
            "Metric": "request_latency_ms",
            "Signal": "Request latency",
            "Value": 125,
            "Unit": "ms",
            "Period": "Incident",
        },
    )


def test_streamlit_config_uses_light_theme_and_disables_file_watcher() -> None:
    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    assert config["server"]["fileWatcherType"] == "none"
    assert config["theme"]["base"] == "light"


def test_landing_state_has_one_action_and_no_result_tabs() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Run incident"]
    assert not app.tabs
    assert app.subheader[0].value == "Cluster 9"
    assert app.header[0].value == "Ready to investigate"


def test_non_ui_packages_import_without_streamlit() -> None:
    script = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name == 'streamlit' or name.startswith('streamlit.'):\n"
        "        raise AssertionError('core package imported streamlit')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "import src.core, src.rca, src.remediation, src.simulation, src.evaluation, src.adapters\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
