"""High-level deterministic incident generation and truth isolation."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Mapping

from src.core.baseline import BaselineStore
from src.core.budget import QueryCosts
from src.core.models import LogEvent, MetricEvent, SpanEvent
from src.core.normalization import TelemetryNormalizer
from src.core.telemetry import TelemetryQueryAPI, TelemetryStore
from src.remediation.domain import FreshObservationWindow

from .config import FaultRequest, SimulationConfig
from .faults import resolve_parameters
from .simulator import MicroserviceSimulator, SimulationOutput


@dataclass(frozen=True, slots=True)
class GroundTruth:
    incident_id: str
    root_service: str
    failure_mode: str
    true_fault_start_time: datetime
    affected_services: frozenset[str]
    injected_fault_parameters: Mapping[str, float]
    clock_offsets_seconds: Mapping[str, float]
    decoy_services: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "injected_fault_parameters", MappingProxyType(dict(self.injected_fault_parameters))
        )
        object.__setattr__(
            self, "clock_offsets_seconds", MappingProxyType(dict(self.clock_offsets_seconds))
        )


@dataclass(frozen=True, slots=True)
class HealthyGroundTruth:
    incident_id: str
    clock_offsets_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clock_offsets_seconds",
            MappingProxyType(dict(self.clock_offsets_seconds)),
        )


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    telemetry: TelemetryStore
    baseline: BaselineStore
    ground_truth: GroundTruth | HealthyGroundTruth
    observed_start_time: datetime
    observed_end_time: datetime


class IncidentGenerator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def generate(self, seed: int, fault: FaultRequest | None = None) -> Incident:
        return self.start_session(seed, fault).initial_incident

    def start_session(
        self, seed: int, fault: FaultRequest | None = None
    ) -> "SimulationIncidentSession":
        return self._start_session(seed, fault, healthy=False)

    def generate_healthy(self, seed: int) -> Incident:
        return self.start_healthy_session(seed).initial_incident

    def start_healthy_session(self, seed: int) -> "SimulationIncidentSession":
        return self._start_session(seed, None, healthy=True)

    def _start_session(
        self,
        seed: int,
        fault: FaultRequest | None,
        *,
        healthy: bool,
    ) -> "SimulationIncidentSession":
        rng = random.Random(seed)
        if healthy:
            root_service = failure_mode = None
            parameters: dict[str, float] = {}
        else:
            root_service, failure_mode, parameters = self._resolve_fault(
                rng, fault or FaultRequest()
            )
        start_time = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=seed % 365)
        fault_step = self.config.baseline_steps
        simulator = MicroserviceSimulator(self.config, seed)
        if healthy:
            raw = simulator.run_healthy(start_time)
        else:
            assert root_service is not None and failure_mode is not None
            raw = simulator.run(
                start_time,
                fault_step,
                root_service,
                failure_mode,
                parameters,
            )

        normalizer = TelemetryNormalizer()
        canonical_names = {
            name: normalizer.normalize_service_name(name) for name in self.config.service_map
        }
        if len(set(canonical_names.values())) != len(canonical_names):
            raise ValueError("configured service names collide after normalization")
        fault_time = start_time + timedelta(seconds=fault_step * self.config.step_seconds)
        baseline_metrics = normalizer.normalize_metrics(
            tuple(event for event in raw.metrics if event.event_timestamp < fault_time)
        )
        baseline = BaselineStore.from_metrics(
            set(canonical_names.values()),
            baseline_metrics,
            {
                (canonical_names[caller], canonical_names[dependency])
                for caller, dependency in self.config.dependency_edges
            },
        )
        offsets = self._clock_offsets(rng)
        if healthy:
            decoys = ()
        else:
            assert root_service is not None
            decoys = self._select_decoys(rng, root_service)
        fragmented = _apply_imperfections(
            self.config, raw, rng, offsets, decoys, fault_time
        )
        metrics = normalizer.normalize_metrics(fragmented.metrics)
        logs = normalizer.normalize_logs(fragmented.logs)
        spans = normalizer.normalize_spans(fragmented.spans)
        store = TelemetryStore(metrics, logs, spans)

        canonical_decoys = tuple(canonical_names[name] for name in decoys)
        canonical_offsets = {canonical_names[name]: value for name, value in offsets.items()}
        if healthy:
            incident_id = f"healthy-{seed}"
            ground_truth: GroundTruth | HealthyGroundTruth = HealthyGroundTruth(
                incident_id,
                canonical_offsets,
            )
        else:
            assert root_service is not None and failure_mode is not None
            canonical_root = canonical_names[root_service]
            affected = frozenset(
                canonical_names[name]
                for name in self._affected_services(root_service)
            )
            incident_id = f"incident-{seed}-{canonical_root}-{failure_mode}"
            ground_truth = GroundTruth(
                incident_id,
                canonical_root,
                failure_mode,
                fault_time,
                affected,
                parameters,
                canonical_offsets,
                canonical_decoys,
            )
        all_times = (
            [event.event_timestamp for event in metrics]
            + [event.event_timestamp for event in logs]
            + [event.start_timestamp for event in spans]
        )
        expected_end = start_time + timedelta(
            seconds=(self.config.baseline_steps + self.config.incident_steps - 1)
            * self.config.step_seconds
        )
        incident = Incident(
            incident_id,
            store,
            baseline,
            ground_truth,
            min(all_times, default=start_time),
            max(all_times, default=expected_end),
        )
        return SimulationIncidentSession(
            self.config,
            simulator,
            rng,
            offsets,
            decoys,
            normalizer,
            store,
            baseline,
            ground_truth,
            incident,
            fault_time,
        )

    def _resolve_fault(
        self, rng: random.Random, request: FaultRequest
    ) -> tuple[str, str, dict[str, float]]:
        service_map = self.config.service_map
        service_name = request.service or rng.choice(sorted(service_map))
        if service_name not in service_map:
            raise ValueError(f"unknown fault service: {service_name}")
        service = service_map[service_name]
        failure_mode = request.failure_mode or rng.choice(sorted(service.valid_failure_modes))
        if failure_mode not in service.valid_failure_modes:
            raise ValueError(f"{failure_mode} is not valid for {service_name}")
        return service_name, failure_mode, resolve_parameters(failure_mode, request.parameters)

    def _clock_offsets(self, rng: random.Random) -> dict[str, float]:
        configured = self.config.fragmentation.clock_offsets_seconds
        bound = self.config.fragmentation.clock_skew_range_seconds
        return {
            name: float(configured[name] if name in configured else rng.uniform(-bound, bound))
            for name in self.config.service_map
        }

    def _select_decoys(self, rng: random.Random, root_service: str) -> tuple[str, ...]:
        if not self.config.fragmentation.include_decoy_anomaly:
            return ()
        candidates = sorted(
            set(self.config.service_map) - self._affected_services(root_service)
        )
        return (rng.choice(candidates),) if candidates else ()

    def _affected_services(self, root_service: str) -> frozenset[str]:
        reverse: dict[str, set[str]] = {name: set() for name in self.config.service_map}
        for caller, dependency in self.config.dependency_edges:
            reverse[dependency].add(caller)
        affected = {root_service}
        pending = [root_service]
        while pending:
            current = pending.pop()
            for upstream in reverse[current]:
                if upstream not in affected:
                    affected.add(upstream)
                    pending.append(upstream)
        return frozenset(affected)


class SimulationIncidentSession:
    """Stateful incident stream with one simulator and fragmentation pipeline."""

    def __init__(
        self,
        config: SimulationConfig,
        simulator: MicroserviceSimulator,
        fragmentation_rng: random.Random,
        clock_offsets: dict[str, float],
        decoy_services: tuple[str, ...],
        normalizer: TelemetryNormalizer,
        telemetry: TelemetryStore,
        baseline: BaselineStore,
        ground_truth: GroundTruth | HealthyGroundTruth,
        initial_incident: Incident,
        fault_time: datetime,
    ) -> None:
        self.config = config
        self.simulator = simulator
        self._fragmentation_rng = fragmentation_rng
        self._clock_offsets = dict(clock_offsets)
        self._decoy_services = tuple(decoy_services)
        self._normalizer = normalizer
        self.telemetry = telemetry
        self.baseline = baseline
        self.ground_truth = ground_truth
        self.initial_incident = initial_incident
        self._fault_time = fault_time
        self._current_observation_time = initial_incident.observed_end_time

    @property
    def current_observation_time(self) -> datetime:
        return self._current_observation_time

    def create_query_api(
        self, total_budget: int, costs: QueryCosts | None = None
    ) -> TelemetryQueryAPI:
        return TelemetryQueryAPI(self.telemetry, total_budget, costs)

    def observe(
        self, api: TelemetryQueryAPI, step_count: int
    ) -> FreshObservationWindow:
        return self.advance(api, step_count)

    def advance(
        self, api: TelemetryQueryAPI, step_count: int
    ) -> FreshObservationWindow:
        if step_count < 1:
            raise ValueError("step_count must be positive")
        raw = self.simulator.continue_run(step_count)
        fragmented = _apply_imperfections(
            self.config,
            raw,
            self._fragmentation_rng,
            self._clock_offsets,
            self._decoy_services,
            self._fault_time,
        )
        metrics = self._normalizer.normalize_metrics(fragmented.metrics)
        logs = self._normalizer.normalize_logs(fragmented.logs)
        spans = self._normalizer.normalize_spans(fragmented.spans)
        api.ingest(metrics, logs, spans)
        event_times = (
            [event.event_timestamp for event in metrics]
            + [event.event_timestamp for event in logs]
            + [event.start_timestamp for event in spans]
        )
        if not event_times:
            assert self.simulator.last_step_time is not None
            event_times = [self.simulator.last_step_time]
        window = FreshObservationWindow(min(event_times), max(event_times), step_count)
        self._current_observation_time = window.end_time
        return window


def _apply_imperfections(
    config: SimulationConfig,
    raw: SimulationOutput,
    rng: random.Random,
    offsets: Mapping[str, float],
    decoys: tuple[str, ...],
    fault_time: datetime,
) -> SimulationOutput:
    cfg = config.fragmentation
    metrics: list[MetricEvent] = []
    for event in raw.metrics:
        if rng.random() < cfg.missing_observation_probability:
            continue
        value = event.value
        if value and cfg.metric_noise_fraction:
            value *= 1.0 + rng.uniform(-cfg.metric_noise_fraction, cfg.metric_noise_fraction)
        if event.service in decoys and event.event_timestamp >= fault_time:
            if event.metric_name == "cpu_utilization":
                value = min(1.0, value + 0.38)
            elif event.metric_name == "request_latency_ms":
                value *= 1.75
        metrics.append(
            replace(
                event,
                event_timestamp=event.event_timestamp
                + timedelta(seconds=offsets[event.service]),
                arrival_timestamp=_arrival_time(config, event.arrival_timestamp, rng),
                value=value,
            )
        )

    logs = [
        replace(
            event,
            event_timestamp=event.event_timestamp
            + timedelta(seconds=offsets[event.service]),
            arrival_timestamp=_arrival_time(config, event.arrival_timestamp, rng),
        )
        for event in raw.logs
        if rng.random() >= cfg.missing_observation_probability
    ]

    spans: list[SpanEvent] = []
    by_trace: dict[str, list[SpanEvent]] = {}
    for span in raw.spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    for trace in by_trace.values():
        if rng.random() < cfg.missing_observation_probability:
            continue
        spans.extend(
            replace(
                span,
                start_timestamp=span.start_timestamp
                + timedelta(seconds=offsets[span.service]),
                arrival_timestamp=_arrival_time(config, span.arrival_timestamp, rng),
            )
            for span in trace
        )
    return SimulationOutput(tuple(metrics), tuple(logs), tuple(spans), raw.state_history)


def _arrival_time(
    config: SimulationConfig,
    event_time: datetime,
    rng: random.Random,
) -> datetime:
    delay = 0.0
    cfg = config.fragmentation
    if rng.random() < cfg.delayed_observation_probability:
        delay = rng.uniform(0.001, cfg.max_delay_seconds)
    return event_time + timedelta(seconds=delay)
