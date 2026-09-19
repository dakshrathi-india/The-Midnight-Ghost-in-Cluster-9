from __future__ import annotations

import csv
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.adapters.rcaeval import (
    EXPECTED_LAYOUT,
    RCAEvalLayoutError,
    discover_case_directories,
    load_rcaeval_case,
)
from src.evaluation.rcaeval_smoke import (
    SUPPORTED_EXTERNAL_FAMILIES,
    evaluate_rcaeval_case,
    infer_rcaeval_case,
    summarize_rcaeval_results,
)
from src.rca import CandidateGenerator, DiagnosisAgent, HypothesisEvaluator
from src.remediation import RecoveryVerifier, SafeRemediationPlanner


START = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_case(case_dir: Path, *, family: str, unknown_pattern: bool) -> None:
    case_dir.mkdir()
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case_id": f"fixture-{family}",
                "services": ["Worker API"],
                "dependency_edges": [],
                "label": {
                    "root_service": "Worker API",
                    "fault_family": family,
                },
            }
        ),
        encoding="utf-8",
    )
    metric_fields = (
        "service",
        "event_timestamp",
        "arrival_timestamp",
        "metric_name",
        "value",
        "unit",
    )
    baseline_rows: list[dict[str, object]] = []
    incident_rows: list[dict[str, object]] = []
    metric_names = (
        "cpu_utilization",
        "memory_utilization",
        "request_latency_ms",
        "error_rate",
    )
    for index in range(20):
        timestamp = START + timedelta(seconds=index)
        for name in metric_names:
            baseline_rows.append(
                {
                    "service": "Worker API",
                    "event_timestamp": timestamp.isoformat(),
                    "arrival_timestamp": (timestamp + timedelta(seconds=1)).isoformat(),
                    "metric_name": name,
                    "value": 10.0 + (index % 3) * 0.1,
                    "unit": "ratio" if "latency" not in name else "ms",
                }
            )
    for index in range(12):
        timestamp = START + timedelta(minutes=2, seconds=index)
        for name in metric_names:
            if unknown_pattern:
                value = 80.0 if name == "memory_utilization" else 10.1
            else:
                value = 80.0 if name == "cpu_utilization" else 10.1
            incident_rows.append(
                {
                    "service": "Worker API",
                    "event_timestamp": timestamp.isoformat(),
                    "arrival_timestamp": (timestamp + timedelta(seconds=2)).isoformat(),
                    "metric_name": name,
                    "value": value,
                    "unit": "ratio" if "latency" not in name else "ms",
                }
            )
    _write_csv(case_dir / "baseline_metrics.csv", metric_fields, baseline_rows)
    _write_csv(case_dir / "metrics.csv", metric_fields, incident_rows)
    if not unknown_pattern:
        log_time = START + timedelta(minutes=2, seconds=2)
        _write_csv(
            case_dir / "logs.csv",
            (
                "service",
                "event_timestamp",
                "arrival_timestamp",
                "severity",
                "message",
                "trace_id",
            ),
            [
                {
                    "service": "Worker API",
                    "event_timestamp": log_time.isoformat(),
                    "arrival_timestamp": (log_time + timedelta(seconds=4)).isoformat(),
                    "severity": "ERROR",
                    "message": "processor utilization is saturated and work is backing up",
                    "trace_id": "trace-1",
                }
            ],
        )
        _write_csv(
            case_dir / "traces.csv",
            (
                "service",
                "start_timestamp",
                "arrival_timestamp",
                "trace_id",
                "span_id",
                "parent_span_id",
                "operation",
                "duration_ms",
                "status",
            ),
            [
                {
                    "service": "Worker API",
                    "start_timestamp": log_time.isoformat(),
                    "arrival_timestamp": (log_time + timedelta(seconds=5)).isoformat(),
                    "trace_id": "trace-1",
                    "span_id": "span-1",
                    "parent_span_id": "",
                    "operation": "worker.request",
                    "duration_ms": 10,
                    "status": "OK",
                }
            ],
        )


def test_rcaeval_adapter_preserves_canonical_fields_and_timestamp_roles(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir, family="cpu", unknown_pattern=False)

    bundle = load_rcaeval_case(case_dir)
    metrics = bundle.case_input.telemetry.metrics_between(
        "worker-api", bundle.case_input.analysis_start, bundle.case_input.analysis_end
    )
    logs = bundle.case_input.telemetry.logs_between(
        "worker-api", bundle.case_input.analysis_start, bundle.case_input.analysis_end
    )
    spans = bundle.case_input.telemetry.spans_between(
        "worker-api", bundle.case_input.analysis_start, bundle.case_input.analysis_end
    )

    assert metrics and logs and spans
    assert metrics[0].service == "worker-api"
    assert metrics[0].arrival_timestamp > metrics[0].event_timestamp
    assert logs[0].arrival_timestamp > logs[0].event_timestamp
    assert spans[0].trace_id == "trace-1"
    assert spans[0].span_id == "span-1"
    assert spans[0].parent_span_id is None
    assert spans[0].arrival_timestamp > spans[0].start_timestamp
    assert bundle.truth.root_service == "worker-api"


def test_rcaeval_mapping_is_conservative_and_reports_open_set_safety(
    tmp_path: Path,
) -> None:
    cpu_dir = tmp_path / "cpu"
    memory_dir = tmp_path / "memory"
    _write_case(cpu_dir, family="cpu", unknown_pattern=False)
    _write_case(memory_dir, family="mem", unknown_pattern=True)
    cpu = load_rcaeval_case(cpu_dir)
    memory = load_rcaeval_case(memory_dir)

    results = (
        evaluate_rcaeval_case(cpu.case_input, cpu.truth),
        evaluate_rcaeval_case(memory.case_input, memory.truth),
    )
    summary = summarize_rcaeval_results(results)

    assert SUPPORTED_EXTERNAL_FAMILIES == {
        "cpu": "cpu_saturation",
        "delay": "network_latency",
    }
    assert results[0].mapped_failure_mode == "cpu_saturation"
    assert results[1].mapped_failure_mode is None
    assert results[1].automatic_action is False
    assert summary.unsupported_case_count == 1
    assert summary.unsupported_automatic_action_rate == 0.0


def test_rcaeval_truth_is_not_accepted_by_inference_or_generic_components() -> None:
    assert "truth" not in inspect.signature(infer_rcaeval_case).parameters
    for component in (
        DiagnosisAgent,
        CandidateGenerator,
        HypothesisEvaluator,
        SafeRemediationPlanner,
        RecoveryVerifier,
    ):
        source = inspect.getsource(component).lower()
        assert "rcaeval" not in source
        assert "ground_truth" not in source


def test_rcaeval_missing_layout_fails_with_expected_files(tmp_path: Path) -> None:
    with pytest.raises(RCAEvalLayoutError) as error:
        discover_case_directories(tmp_path)

    assert "case.json" in str(error.value)
    assert "baseline_metrics.csv" in EXPECTED_LAYOUT
