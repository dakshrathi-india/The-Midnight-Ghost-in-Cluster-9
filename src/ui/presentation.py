"""Pure transformations from domain results to dashboard-ready records."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from src.core import MetricEvent, QueryHistoryEntry
from src.rca import (
    CausalEvidence,
    DiagnosisResult,
    EvidenceStatus,
    HypothesisEvaluation,
)
from src.simulation import SimulationConfig


EVALUATION_TRUTH_LABEL = "EVALUATION ONLY · HIDDEN GROUND TRUTH"

_CANDIDATE_ORDER = {"STRONG": 0, "WEAK": 1, "NOT_CANDIDATE": 2}
_SNAPSHOT_METRICS = {
    "cpu_utilization": "CPU utilization",
    "memory_utilization": "Memory utilization",
    "request_latency_ms": "Request latency",
    "error_rate": "Error rate",
    "request_rate": "Request rate",
}
_EVIDENCE_LABELS = {
    "METRIC_PATTERN": "Metric",
    "LOG_SEMANTIC": "Log",
    "TRACE_LOCALIZATION": "Trace",
    "TEMPORAL_PRECEDENCE": "Temporal",
    "GRAPH_PROPAGATION": "Graph",
    "CANDIDATE_STRENGTH": "Candidate",
    "INTERVENTION_OUTCOME": "Intervention",
}


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    directory: Path
    summaries: tuple[dict[str, object], ...]
    incidents: tuple[dict[str, object], ...]
    ablations: tuple[dict[str, object], ...]
    budget_curve: tuple[dict[str, object], ...]
    missing_files: tuple[str, ...]


def service_failure_mode_options(
    config: SimulationConfig,
) -> dict[str, tuple[str, ...]]:
    """Return valid fault choices directly from the simulator configuration."""
    return {
        service.name: tuple(sorted(service.valid_failure_modes))
        for service in sorted(config.services, key=lambda item: item.name)
    }


def candidate_rows(diagnosis: DiagnosisResult) -> tuple[dict[str, object], ...]:
    rows = (
        {
            "Service": candidate.service,
            "Strength": candidate.strength.value,
            "MAD": "●" if "mad" in candidate.fired_detectors else "—",
            "CUSUM": "●" if "change_point" in candidate.fired_detectors else "—",
            "IF": "●" if "isolation_forest" in candidate.fired_detectors else "—",
            "Corroborated": bool(candidate.corroborating_reasons),
        }
        for candidate in diagnosis.candidates
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                _CANDIDATE_ORDER[str(row["Strength"])],
                str(row["Service"]),
            ),
        )
    )


def hypothesis_rows(
    diagnosis: DiagnosisResult,
) -> tuple[dict[str, object], ...]:
    """Preserve only the frozen five causal ranking components."""
    return tuple(
        {
            "#": index,
            "Service": evaluation.hypothesis.service,
            "Failure Mode": evaluation.hypothesis.failure_mode,
            "Contradictions": evaluation.ranking.contradiction_count,
            "Unexplained": evaluation.ranking.unexplained_strong_count,
            "Explained": evaluation.ranking.explained_strong_count,
            "Support Types": evaluation.ranking.supporting_category_count,
            "Direct Support": evaluation.ranking.direct_support_count,
        }
        for index, evaluation in enumerate(diagnosis.ranked_hypotheses, start=1)
    )


def evidence_rows(
    evaluation: HypothesisEvaluation,
) -> tuple[dict[str, object], ...]:
    """Expose causal evidence only; evaluation-only truth is never an input."""
    return tuple(
        {
            "Evidence": _EVIDENCE_LABELS[evidence.category.value],
            "Service": evidence.service,
            "Observation": evidence.observation,
            "Verdict": evidence.status.value,
        }
        for evidence in sorted(
            evaluation.evidence,
            key=lambda item: (
                item.event_timestamp is None,
                item.event_timestamp or datetime.max,
                item.category.value,
                item.service,
            ),
        )
    )


def prominent_evidence(
    evaluation: HypothesisEvaluation,
) -> tuple[CausalEvidence, ...]:
    """Select diverse support while retaining every contradiction."""
    supports: list[CausalEvidence] = []
    contradictions: list[CausalEvidence] = []
    seen_categories: set[str] = set()
    for evidence in evaluation.evidence:
        if evidence.status is EvidenceStatus.CONTRADICTION:
            contradictions.append(evidence)
        elif (
            evidence.status is EvidenceStatus.SUPPORT
            and evidence.category.value not in seen_categories
        ):
            seen_categories.add(evidence.category.value)
            supports.append(evidence)
    support_limit = max(3, 5 - len(contradictions))
    return tuple((*supports[:support_limit], *contradictions))


def query_history_rows(
    history: Sequence[QueryHistoryEntry],
) -> tuple[dict[str, object], ...]:
    cumulative = 0
    rows: list[dict[str, object]] = []
    for number, entry in enumerate(history, start=1):
        cumulative += entry.cost
        rows.append(
            {
                "#": number,
                "Query": entry.query_type,
                "Service": entry.service,
                "Cost": entry.cost,
                "Cache": entry.served_from_cache,
                "Cumulative": cumulative,
            }
        )
    return tuple(rows)


def topology_dot(
    config: SimulationConfig,
    diagnosed_root: str | None = None,
    strong_candidates: Sequence[str] = (),
) -> str:
    strong = frozenset(strong_candidates)
    propagation_edges = _propagation_edges(
        config.dependency_edges, diagnosed_root, strong
    )
    lines = [
        "digraph topology {",
        'rankdir="LR";',
        'graph [bgcolor="transparent", pad="0.22", nodesep="0.42", '
        'ranksep="0.68", splines="polyline"];',
        'node [shape="box", style="rounded,filled", fillcolor="#FCFBF8", '
        'fontcolor="#3A3936", color="#DEDCD6", fontname="Arial", '
        'fontsize="12", margin="0.19,0.12"];',
        'edge [color="#C7C4BD", arrowsize="0.62", penwidth="1.05"];',
    ]
    for service in sorted(config.service_map):
        if service == diagnosed_root:
            attributes = (
                'fillcolor="#E9E8FA", fontcolor="#302E68", '
                'color="#5754C8", penwidth="2.5"'
            )
        elif service in strong:
            attributes = (
                'fillcolor="#F4EBDD", fontcolor="#674923", '
                'color="#B17A32", penwidth="1.5"'
            )
        else:
            attributes = ""
        lines.append(f'"{_dot_escape(service)}" [{attributes}];')
    for caller, callee in sorted(config.dependency_edges):
        attributes = (
            ' [color="#6966CB", penwidth="1.9"]'
            if (caller, callee) in propagation_edges
            else ""
        )
        lines.append(
            f'"{_dot_escape(caller)}" -> "{_dot_escape(callee)}"{attributes};'
        )
    lines.append("}")
    return "\n".join(lines)


def telemetry_signal_rows(
    periods: Sequence[tuple[str, Sequence[MetricEvent]]],
) -> tuple[dict[str, object], ...]:
    """Return long-form operational signals with their real telemetry period."""
    rows: list[dict[str, object]] = []
    for period, events in periods:
        for event in sorted(
            events, key=lambda item: (item.event_timestamp, item.metric_name)
        ):
            signal = _SNAPSHOT_METRICS.get(event.metric_name)
            if signal is None:
                continue
            rows.append(
                {
                    "Event time": event.event_timestamp,
                    "Metric": event.metric_name,
                    "Signal": signal,
                    "Value": event.value,
                    "Unit": event.unit,
                    "Period": period,
                }
            )
    return tuple(rows)


def _propagation_edges(
    edges: Sequence[tuple[str, str]],
    diagnosed_root: str | None,
    strong_candidates: frozenset[str],
) -> frozenset[tuple[str, str]]:
    if diagnosed_root is None:
        return frozenset()

    outgoing: dict[str, list[str]] = {}
    for caller, callee in edges:
        outgoing.setdefault(caller, []).append(callee)

    highlighted: set[tuple[str, str]] = set()
    for start in strong_candidates - {diagnosed_root}:
        stack: list[tuple[str, tuple[tuple[str, str], ...], frozenset[str]]] = [
            (start, (), frozenset({start}))
        ]
        while stack:
            service, path, visited = stack.pop()
            if service == diagnosed_root:
                highlighted.update(path)
                continue
            for dependency in outgoing.get(service, ()):
                if dependency in visited:
                    continue
                stack.append(
                    (
                        dependency,
                        (*path, (service, dependency)),
                        visited | {dependency},
                    )
                )
    return frozenset(highlighted)


def discover_benchmark_artifacts(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    filenames = {
        "benchmark_summary.json",
        "benchmark_results.csv",
        "ablation_results.csv",
        "budget_curve.csv",
    }
    return tuple(
        sorted(
            {
                path.parent
                for path in root.rglob("*")
                if path.is_file() and path.name in filenames
            },
            key=lambda path: str(path).lower(),
        )
    )


def load_benchmark_artifacts(directory: Path) -> BenchmarkArtifacts:
    """Load one generated benchmark directory without running a benchmark."""
    summary_path = directory / "benchmark_summary.json"
    incidents_path = directory / "benchmark_results.csv"
    ablations_path = directory / "ablation_results.csv"
    curve_path = directory / "budget_curve.csv"
    expected = (summary_path, incidents_path, ablations_path, curve_path)

    summaries: tuple[dict[str, object], ...] = ()
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("summaries"), list
        ):
            raise ValueError(
                f"{summary_path} must contain a top-level summaries list"
            )
        summaries = tuple(
            _object_row(row, summary_path) for row in payload["summaries"]
        )

    return BenchmarkArtifacts(
        directory=directory,
        summaries=summaries,
        incidents=_read_csv(incidents_path),
        ablations=_read_csv(ablations_path),
        budget_curve=_read_csv(curve_path),
        missing_files=tuple(path.name for path in expected if not path.is_file()),
    )


def _read_csv(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")
        return tuple(
            {key: _parse_scalar(value) for key, value in row.items()}
            for row in reader
        )


def _object_row(raw: object, path: Path) -> dict[str, object]:
    if not isinstance(raw, Mapping) or not all(
        isinstance(key, str) for key in raw
    ):
        raise ValueError(f"{path} contains a non-object summary row")
    return dict(raw)


def _parse_scalar(value: str | None) -> object:
    if value is None or value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
