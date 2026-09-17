"""Independent robust, change-point, and multivariate anomaly signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from src.core.baseline import BaselineStore, MetricSummary
from src.core.models import MetricEvent


@dataclass(frozen=True, slots=True)
class MADConfig:
    modified_z_threshold: float = 3.5
    extreme_modified_z_threshold: float = 8.0
    minimum_anomalous_samples: int = 2
    minimum_anomalous_fraction: float = 0.2
    dispersion_epsilon: float = 1e-9


@dataclass(frozen=True, slots=True)
class MADMetricEvidence:
    metric_name: str
    flagged: bool
    baseline_median: float
    baseline_mad: float
    dispersion_used: float
    modified_z_scores: tuple[float, ...]
    maximum_absolute_modified_z: float
    anomalous_count: int
    anomalous_fraction: float


@dataclass(frozen=True, slots=True)
class MADServiceEvidence:
    service: str
    flagged: bool
    metric_evidence: Mapping[str, MADMetricEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_evidence", MappingProxyType(dict(self.metric_evidence)))


class MADDetector:
    def __init__(self, config: MADConfig | None = None) -> None:
        self.config = config or MADConfig()

    def analyze(
        self,
        service: str,
        incident_metrics: Sequence[MetricEvent],
        baseline: BaselineStore,
    ) -> MADServiceEvidence:
        grouped = _group_metric_values(service, incident_metrics)
        evidence: dict[str, MADMetricEvidence] = {}
        for metric_name, values in sorted(grouped.items()):
            summary = baseline.metric_summaries.get((service, metric_name))
            if summary is None or not values:
                continue
            dispersion = _robust_dispersion(summary, self.config.dispersion_epsilon)
            scores = [0.67448975 * (value - summary.median) / dispersion for value in values]
            anomalous_count = sum(
                abs(score) >= self.config.modified_z_threshold for score in scores
            )
            anomalous_fraction = anomalous_count / len(scores)
            maximum_score = max(map(abs, scores), default=0.0)
            sustained_anomaly = (
                anomalous_count >= self.config.minimum_anomalous_samples
                and anomalous_fraction >= self.config.minimum_anomalous_fraction
            )
            evidence[metric_name] = MADMetricEvidence(
                metric_name=metric_name,
                flagged=(
                    sustained_anomaly
                    or maximum_score >= self.config.extreme_modified_z_threshold
                ),
                baseline_median=summary.median,
                baseline_mad=summary.median_absolute_deviation,
                dispersion_used=dispersion,
                modified_z_scores=tuple(scores),
                maximum_absolute_modified_z=maximum_score,
                anomalous_count=anomalous_count,
                anomalous_fraction=anomalous_fraction,
            )
        return MADServiceEvidence(service, any(item.flagged for item in evidence.values()), evidence)


@dataclass(frozen=True, slots=True)
class CUSUMConfig:
    threshold: float = 5.0
    drift: float = 1.0
    minimum_shifted_samples: int = 3
    dispersion_epsilon: float = 1e-9


@dataclass(frozen=True, slots=True)
class CUSUMMetricEvidence:
    metric_name: str
    flagged: bool
    positive_cusum_series: tuple[float, ...]
    negative_cusum_series: tuple[float, ...]
    maximum_positive_cusum: float
    maximum_negative_cusum: float
    longest_positive_run: int
    longest_negative_run: int
    direction: str | None
    threshold: float
    drift: float


@dataclass(frozen=True, slots=True)
class CUSUMServiceEvidence:
    service: str
    flagged: bool
    metric_evidence: Mapping[str, CUSUMMetricEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_evidence", MappingProxyType(dict(self.metric_evidence)))


class CUSUMDetector:
    def __init__(self, config: CUSUMConfig | None = None) -> None:
        self.config = config or CUSUMConfig()

    def analyze(
        self,
        service: str,
        incident_metrics: Sequence[MetricEvent],
        baseline: BaselineStore,
    ) -> CUSUMServiceEvidence:
        grouped = _group_metric_events(service, incident_metrics)
        evidence: dict[str, CUSUMMetricEvidence] = {}
        for metric_name, events in sorted(grouped.items()):
            summary = baseline.metric_summaries.get((service, metric_name))
            if summary is None:
                continue
            dispersion = _standard_dispersion(summary, self.config.dispersion_epsilon)
            positive = negative = maximum_positive = maximum_negative = 0.0
            positive_run = negative_run = 0
            longest_positive_run = longest_negative_run = 0
            positive_detected = negative_detected = False
            positive_series: list[float] = []
            negative_series: list[float] = []
            for event in events:
                standardized = (event.value - summary.mean) / dispersion
                positive = max(0.0, positive + standardized - self.config.drift)
                negative = max(0.0, negative - standardized - self.config.drift)
                positive_run = positive_run + 1 if standardized > self.config.drift else 0
                negative_run = negative_run + 1 if standardized < -self.config.drift else 0
                longest_positive_run = max(longest_positive_run, positive_run)
                longest_negative_run = max(longest_negative_run, negative_run)
                positive_detected = positive_detected or (
                    positive >= self.config.threshold
                    and positive_run >= self.config.minimum_shifted_samples
                )
                negative_detected = negative_detected or (
                    negative >= self.config.threshold
                    and negative_run >= self.config.minimum_shifted_samples
                )
                positive_series.append(positive)
                negative_series.append(negative)
                maximum_positive = max(maximum_positive, positive)
                maximum_negative = max(maximum_negative, negative)
            positive_flagged = positive_detected
            negative_flagged = negative_detected
            flagged = positive_flagged or negative_flagged
            direction = None
            if flagged:
                direction = (
                    "upward"
                    if positive_flagged
                    and (not negative_flagged or maximum_positive >= maximum_negative)
                    else "downward"
                )
            evidence[metric_name] = CUSUMMetricEvidence(
                metric_name=metric_name,
                flagged=flagged,
                positive_cusum_series=tuple(positive_series),
                negative_cusum_series=tuple(negative_series),
                maximum_positive_cusum=maximum_positive,
                maximum_negative_cusum=maximum_negative,
                longest_positive_run=longest_positive_run,
                longest_negative_run=longest_negative_run,
                direction=direction,
                threshold=self.config.threshold,
                drift=self.config.drift,
            )
        return CUSUMServiceEvidence(
            service, any(item.flagged for item in evidence.values()), evidence
        )


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    random_state: int = 17
    contamination: str | float = 0.12
    n_estimators: int = 100
    minimum_baseline_samples: int = 4
    minimum_anomalous_samples: int = 3
    minimum_anomalous_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class IsolationForestEvidence:
    service: str
    flagged: bool
    feature_names: tuple[str, ...]
    training_sample_count: int
    incident_sample_count: int
    anomaly_scores: tuple[float, ...]
    anomalous_count: int
    anomalous_fraction: float
    maximum_anomaly_score: float


class IsolationForestDetector:
    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()

    def analyze(
        self,
        service: str,
        incident_metrics: Sequence[MetricEvent],
        baseline: BaselineStore,
    ) -> IsolationForestEvidence:
        baseline_events = [event for event in baseline.metric_history if event.service == service]
        incident_events = [event for event in incident_metrics if event.service == service]
        baseline_names = {event.metric_name for event in baseline_events}
        incident_names = {event.metric_name for event in incident_events}
        feature_names = tuple(sorted(baseline_names & incident_names))
        if not feature_names:
            return self._empty(service)

        fill_values = {
            name: baseline.metric_summaries[(service, name)].median
            for name in feature_names
            if (service, name) in baseline.metric_summaries
        }
        feature_names = tuple(name for name in feature_names if name in fill_values)
        baseline_matrix = _feature_matrix(baseline_events, feature_names, fill_values)
        incident_matrix = _feature_matrix(incident_events, feature_names, fill_values)
        if (
            len(baseline_matrix) < self.config.minimum_baseline_samples
            or len(incident_matrix) == 0
            or not feature_names
        ):
            return self._empty(
                service, feature_names, len(baseline_matrix), len(incident_matrix)
            )

        model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        model.fit(baseline_matrix)
        predictions = model.predict(incident_matrix)
        scores = -model.decision_function(incident_matrix)
        anomalous_count = int(np.count_nonzero(predictions == -1))
        anomalous_fraction = anomalous_count / len(predictions)
        return IsolationForestEvidence(
            service=service,
            flagged=(
                anomalous_count >= self.config.minimum_anomalous_samples
                and anomalous_fraction >= self.config.minimum_anomalous_fraction
            ),
            feature_names=feature_names,
            training_sample_count=len(baseline_matrix),
            incident_sample_count=len(incident_matrix),
            anomaly_scores=tuple(float(score) for score in scores),
            anomalous_count=anomalous_count,
            anomalous_fraction=anomalous_fraction,
            maximum_anomaly_score=max((float(score) for score in scores), default=0.0),
        )

    @staticmethod
    def _empty(
        service: str,
        feature_names: tuple[str, ...] = (),
        training_samples: int = 0,
        incident_samples: int = 0,
    ) -> IsolationForestEvidence:
        return IsolationForestEvidence(
            service, False, feature_names, training_samples, incident_samples, (), 0, 0.0, 0.0
        )


def _group_metric_values(
    service: str, events: Sequence[MetricEvent]
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        if event.service == service:
            grouped.setdefault(event.metric_name, []).append(event.value)
    return grouped


def _group_metric_events(
    service: str, events: Sequence[MetricEvent]
) -> dict[str, list[MetricEvent]]:
    grouped: dict[str, list[MetricEvent]] = {}
    for event in sorted(events, key=lambda item: (item.event_timestamp, item.metric_name)):
        if event.service == service:
            grouped.setdefault(event.metric_name, []).append(event)
    return grouped


def _robust_dispersion(summary: MetricSummary, epsilon: float) -> float:
    if summary.median_absolute_deviation > epsilon:
        return summary.median_absolute_deviation
    if summary.standard_deviation > epsilon:
        return summary.standard_deviation * 0.67448975
    return max(abs(summary.median) * 0.01, epsilon)


def _standard_dispersion(summary: MetricSummary, epsilon: float) -> float:
    if summary.standard_deviation > epsilon:
        return summary.standard_deviation
    robust_sigma = summary.median_absolute_deviation / 0.67448975
    if robust_sigma > epsilon:
        return robust_sigma
    return max(abs(summary.mean) * 0.01, epsilon)


def _feature_matrix(
    events: Sequence[MetricEvent],
    feature_names: tuple[str, ...],
    fill_values: Mapping[str, float],
) -> np.ndarray:
    by_timestamp: dict[datetime, dict[str, list[float]]] = {}
    for event in events:
        if event.metric_name in feature_names:
            by_timestamp.setdefault(event.event_timestamp, {}).setdefault(
                event.metric_name, []
            ).append(event.value)
    rows = []
    for timestamp in sorted(by_timestamp):
        values = by_timestamp[timestamp]
        rows.append(
            [
                float(np.mean(values[name])) if name in values else fill_values[name]
                for name in feature_names
            ]
        )
    return np.asarray(rows, dtype=float)
