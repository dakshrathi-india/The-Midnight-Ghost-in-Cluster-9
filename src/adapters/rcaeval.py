"""Local RCAEval RE2-style case ingestion into canonical telemetry."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.core import BaselineStore, LogEvent, MetricEvent, SpanEvent, TelemetryStore
from src.core.normalization import TelemetryNormalizer


class RCAEvalLayoutError(ValueError):
    """Raised when a local case does not match the supported fixture layout."""


@dataclass(frozen=True, slots=True)
class RCAEvalTruth:
    root_service: str
    fault_family: str


@dataclass(frozen=True, slots=True)
class RCAEvalCaseInput:
    case_id: str
    baseline: BaselineStore
    telemetry: TelemetryStore
    analysis_start: datetime
    analysis_end: datetime


@dataclass(frozen=True, slots=True)
class RCAEvalCaseBundle:
    case_input: RCAEvalCaseInput
    truth: RCAEvalTruth


EXPECTED_LAYOUT = """Expected each case directory to contain:
  case.json: {"case_id": str, "services": [str],
              "dependency_edges": [[caller, callee], ...],
              "label": {"root_service": str, "fault_family": str}}
  baseline_metrics.csv: service,event_timestamp[,arrival_timestamp],metric_name,value,unit
  metrics.csv:          service,event_timestamp[,arrival_timestamp],metric_name,value,unit
Optional:
  logs.csv:   service,event_timestamp[,arrival_timestamp],severity,message[,trace_id]
  traces.csv: service,start_timestamp[,arrival_timestamp],trace_id,span_id,
              parent_span_id,operation,duration_ms,status
"""


def discover_case_directories(path: Path) -> tuple[Path, ...]:
    if (path / "case.json").is_file():
        return (path,)
    if not path.is_dir():
        raise RCAEvalLayoutError(f"case directory does not exist: {path}\n{EXPECTED_LAYOUT}")
    cases = tuple(sorted(child for child in path.iterdir() if (child / "case.json").is_file()))
    if not cases:
        raise RCAEvalLayoutError(f"no RCAEval cases found under {path}\n{EXPECTED_LAYOUT}")
    return cases


def load_rcaeval_case(case_dir: Path) -> RCAEvalCaseBundle:
    metadata_path = case_dir / "case.json"
    baseline_path = case_dir / "baseline_metrics.csv"
    metrics_path = case_dir / "metrics.csv"
    missing = [path.name for path in (metadata_path, baseline_path, metrics_path) if not path.is_file()]
    if missing:
        raise RCAEvalLayoutError(
            f"case {case_dir} is missing required files: {', '.join(missing)}\n{EXPECTED_LAYOUT}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RCAEvalLayoutError(f"invalid {metadata_path}: {error}") from error
    if not isinstance(metadata, Mapping):
        raise RCAEvalLayoutError(f"{metadata_path} must contain a JSON object")

    normalizer = TelemetryNormalizer()
    services = _text_sequence(metadata, "services")
    edges = _dependency_edges(metadata.get("dependency_edges", ()), normalizer)
    baseline_metrics = normalizer.normalize_metrics(_read_metrics(baseline_path))
    incident_metrics = normalizer.normalize_metrics(_read_metrics(metrics_path))
    logs = normalizer.normalize_logs(_read_logs(case_dir / "logs.csv"))
    spans = normalizer.normalize_spans(_read_spans(case_dir / "traces.csv"))
    canonical_services = {
        normalizer.normalize_service_name(service) for service in services
    }
    canonical_services.update(event.service for event in baseline_metrics)
    canonical_services.update(event.service for event in incident_metrics)
    if not canonical_services:
        raise RCAEvalLayoutError("case must define at least one service")
    baseline = BaselineStore.from_metrics(
        canonical_services,
        baseline_metrics,
        set(edges),
    )
    all_times = (
        [event.event_timestamp for event in incident_metrics]
        + [event.event_timestamp for event in logs]
        + [event.start_timestamp for event in spans]
    )
    if not all_times:
        raise RCAEvalLayoutError("case contains no incident telemetry")
    label = metadata.get("label")
    if not isinstance(label, Mapping):
        raise RCAEvalLayoutError("case.json requires a label object")
    root_service = _required_text(label, "root_service")
    fault_family = _required_text(label, "fault_family").lower()
    case_id = str(metadata.get("case_id", case_dir.name)).strip() or case_dir.name
    return RCAEvalCaseBundle(
        RCAEvalCaseInput(
            case_id,
            baseline,
            TelemetryStore(incident_metrics, logs, spans),
            min(all_times),
            max(all_times),
        ),
        RCAEvalTruth(normalizer.normalize_service_name(root_service), fault_family),
    )


def _read_metrics(path: Path) -> tuple[MetricEvent, ...]:
    return tuple(
        MetricEvent(
            _timestamp(row, "event_timestamp"),
            _timestamp(row, "arrival_timestamp", fallback="event_timestamp"),
            _required_text(row, "service"),
            _required_text(row, "metric_name"),
            _float(row, "value"),
            str(row.get("unit", "")),
        )
        for row in _read_rows(path)
    )


def _read_logs(path: Path) -> tuple[LogEvent, ...]:
    if not path.is_file():
        return ()
    return tuple(
        LogEvent(
            _timestamp(row, "event_timestamp"),
            _timestamp(row, "arrival_timestamp", fallback="event_timestamp"),
            _required_text(row, "service"),
            _required_text(row, "severity"),
            _required_text(row, "message"),
            str(row["trace_id"]).strip() if row.get("trace_id") else None,
        )
        for row in _read_rows(path)
    )


def _read_spans(path: Path) -> tuple[SpanEvent, ...]:
    if not path.is_file():
        return ()
    return tuple(
        SpanEvent(
            _required_text(row, "trace_id"),
            _required_text(row, "span_id"),
            str(row["parent_span_id"]).strip() if row.get("parent_span_id") else None,
            _required_text(row, "service"),
            _required_text(row, "operation"),
            _timestamp(row, "start_timestamp"),
            _timestamp(row, "arrival_timestamp", fallback="start_timestamp"),
            _float(row, "duration_ms"),
            _required_text(row, "status"),
        )
        for row in _read_rows(path)
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except OSError as error:
        raise RCAEvalLayoutError(f"could not read {path}: {error}") from error


def _timestamp(
    row: Mapping[str, object], name: str, *, fallback: str | None = None
) -> datetime:
    raw = row.get(name) or (row.get(fallback) if fallback else None)
    if raw is None or not str(raw).strip():
        raise RCAEvalLayoutError(f"missing required timestamp column {name!r}")
    text = str(raw).strip()
    try:
        if text.replace(".", "", 1).isdigit():
            value = float(text)
            absolute = abs(value)
            if absolute >= 1e17:
                divisor = 1_000_000_000
            elif absolute >= 1e14:
                divisor = 1_000_000
            elif absolute >= 1e11:
                divisor = 1_000
            else:
                divisor = 1
            return datetime.fromtimestamp(value / divisor, tz=timezone.utc)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (OverflowError, ValueError) as error:
        raise RCAEvalLayoutError(f"invalid timestamp {text!r}") from error
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _float(row: Mapping[str, object], name: str) -> float:
    try:
        return float(_required_text(row, name))
    except ValueError as error:
        raise RCAEvalLayoutError(f"field {name!r} must be numeric") from error


def _required_text(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise RCAEvalLayoutError(f"missing required field {name!r}")
    return str(value).strip()


def _text_sequence(metadata: Mapping[str, object], name: str) -> tuple[str, ...]:
    raw = metadata.get(name, ())
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise RCAEvalLayoutError(f"{name!r} must be a list of strings")
    return tuple(raw)


def _dependency_edges(
    raw: object, normalizer: TelemetryNormalizer
) -> frozenset[tuple[str, str]]:
    if not isinstance(raw, list):
        raise RCAEvalLayoutError("dependency_edges must be a list")
    result: set[tuple[str, str]] = set()
    for edge in raw:
        if not isinstance(edge, list) or len(edge) != 2:
            raise RCAEvalLayoutError("dependency edges must be [caller, callee] pairs")
        result.add(
            (
                normalizer.normalize_service_name(str(edge[0])),
                normalizer.normalize_service_name(str(edge[1])),
            )
        )
    return frozenset(result)
