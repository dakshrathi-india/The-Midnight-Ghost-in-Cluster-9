"""Readable discrete-time microservice load and cascade simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.core.models import LogEvent, MetricEvent, SpanEvent

from .config import SimulationConfig
from .faults import FaultEffects, describe_fault, effects_for


@dataclass(frozen=True, slots=True)
class ServiceState:
    request_rate: float
    processed_rate: float
    capacity: float
    queue_length: float
    latency_ms: float
    cpu_utilization: float
    memory_utilization: float
    error_rate: float
    available: bool


@dataclass(frozen=True, slots=True)
class SimulationOutput:
    metrics: tuple[MetricEvent, ...]
    logs: tuple[LogEvent, ...]
    spans: tuple[SpanEvent, ...]
    state_history: tuple[dict[str, ServiceState], ...]


class MicroserviceSimulator:
    def __init__(self, config: SimulationConfig, seed: int) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._service_map = config.service_map
        self._order = config.topological_order()
        self._queues = {name: 0.0 for name in self._order}
        self._previous_states = {
            name: ServiceState(0, 0, service.capacity_rps, 0, service.base_latency_ms, 0, service.base_memory_utilization, 0, True)
            for name, service in self._service_map.items()
        }

    def run(
        self,
        start_time: datetime,
        fault_start_step: int,
        root_service: str,
        failure_mode: str,
        fault_parameters: dict[str, float],
    ) -> SimulationOutput:
        metrics: list[MetricEvent] = []
        logs: list[LogEvent] = []
        spans: list[SpanEvent] = []
        history: list[dict[str, ServiceState]] = []
        total_steps = self.config.baseline_steps + self.config.incident_steps

        for step in range(total_steps):
            true_time = start_time + timedelta(seconds=step * self.config.step_seconds)
            fault_active = step >= fault_start_step
            states = self._simulate_step(root_service, failure_mode, fault_parameters, fault_active)
            history.append(states)
            metrics.extend(self._metric_events(true_time, states))
            logs.extend(self._log_events(true_time, states, root_service, failure_mode, step, fault_start_step))
            spans.extend(self._trace_events(true_time, states, step))
            self._previous_states = states

        return SimulationOutput(tuple(metrics), tuple(logs), tuple(spans), tuple(history))

    def _simulate_step(
        self,
        root_service: str,
        failure_mode: str,
        parameters: dict[str, float],
        fault_active: bool,
    ) -> dict[str, ServiceState]:
        arrivals = {name: 0.0 for name in self._order}
        for service in self._service_map.values():
            if service.external_request_rate:
                arrivals[service.name] += service.external_request_rate * self._rng.uniform(0.94, 1.06)

        for name in self._order:
            service = self._service_map[name]
            for dependency in service.dependencies:
                previous_error = self._previous_states[dependency].error_rate
                retry_multiplier = 1.0 + self.config.retry_factor * previous_error
                arrivals[dependency] += arrivals[name] * retry_multiplier

        local: dict[str, ServiceState] = {}
        for name in self._order:
            service = self._service_map[name]
            effects = (
                effects_for(failure_mode, parameters)
                if fault_active and name == root_service
                else FaultEffects()
            )
            capacity = service.capacity_rps * effects.capacity_factor
            demand = arrivals[name] + self._queues[name]
            processed = min(demand, capacity) if effects.available else 0.0
            queue = min(max(0.0, demand - capacity), service.capacity_rps * 20)
            self._queues[name] = queue
            utilization = min(1.0, arrivals[name] / max(capacity, 0.001))
            cpu = max(effects.cpu_floor, min(1.0, 0.12 + 0.83 * utilization))
            queue_pressure = queue / service.capacity_rps
            latency = (
                service.base_latency_ms * effects.latency_factor * (1.0 + 1.8 * queue_pressure)
                + effects.latency_add_ms
            )
            overload_error = min(0.7, queue_pressure * 0.08)
            error_rate = min(1.0, effects.error_addition + overload_error)
            memory = min(1.0, service.base_memory_utilization + queue_pressure * 0.025)
            local[name] = ServiceState(
                request_rate=arrivals[name],
                processed_rate=processed,
                capacity=capacity,
                queue_length=queue,
                latency_ms=latency,
                cpu_utilization=cpu,
                memory_utilization=memory,
                error_rate=error_rate,
                available=effects.available,
            )

        combined: dict[str, ServiceState] = dict(local)
        for name in reversed(self._order):
            service = self._service_map[name]
            if not service.dependencies:
                continue
            dependency_states = [combined[dependency] for dependency in service.dependencies]
            dependency_latency = sum(state.latency_ms for state in dependency_states)
            dependency_error = 1.0
            for state in dependency_states:
                timeout_probability = min(0.55, max(0.0, state.latency_ms - 120.0) / 700.0)
                dependency_error *= 1.0 - min(1.0, state.error_rate + timeout_probability)
            propagated_error = 1.0 - dependency_error
            state = combined[name]
            combined[name] = ServiceState(
                request_rate=state.request_rate,
                processed_rate=state.processed_rate,
                capacity=state.capacity,
                queue_length=state.queue_length,
                latency_ms=state.latency_ms + dependency_latency,
                cpu_utilization=state.cpu_utilization,
                memory_utilization=state.memory_utilization,
                error_rate=min(1.0, state.error_rate + (1.0 - state.error_rate) * propagated_error),
                available=state.available,
            )
        return combined

    def _metric_events(self, true_time: datetime, states: dict[str, ServiceState]) -> list[MetricEvent]:
        result: list[MetricEvent] = []
        fields = (
            ("cpu_utilization", "ratio"),
            ("memory_utilization", "ratio"),
            ("latency_ms", "ms"),
            ("error_rate", "ratio"),
            ("request_rate", "requests/s"),
            ("queue_length", "requests"),
        )
        metric_names = {"latency_ms": "request_latency_ms"}
        for service, state in states.items():
            for field_name, unit in fields:
                result.append(
                    MetricEvent(
                        true_time,
                        true_time,
                        service,
                        metric_names.get(field_name, field_name),
                        float(getattr(state, field_name)),
                        unit,
                    )
                )
        return result

    def _log_events(
        self,
        true_time: datetime,
        states: dict[str, ServiceState],
        root_service: str,
        failure_mode: str,
        step: int,
        fault_start_step: int,
    ) -> list[LogEvent]:
        messages: list[LogEvent] = []
        normal_variants = ("request batch completed", "health check passed", "worker cycle completed")
        for index, (service, state) in enumerate(states.items()):
            if step == fault_start_step and service == root_service:
                messages.append(
                    LogEvent(true_time, true_time, service, "ERROR", describe_fault(failure_mode))
                )
            elif state.error_rate >= 0.12:
                symptom = self._rng.choice(
                    ("upstream request failed after dependency error", "request timed out while awaiting dependency", "elevated request failure rate observed")
                )
                messages.append(LogEvent(true_time, true_time, service, "ERROR", symptom))
            elif state.queue_length > self._service_map[service].capacity_rps * 0.5:
                messages.append(
                    LogEvent(
                        true_time,
                        true_time,
                        service,
                        "WARN",
                        "request queue is above normal range",
                    )
                )
            elif step % 3 == index % 3:
                messages.append(
                    LogEvent(
                        true_time,
                        true_time,
                        service,
                        "INFO",
                        self._rng.choice(normal_variants),
                    )
                )
        return messages

    def _trace_events(
        self, true_time: datetime, states: dict[str, ServiceState], step: int
    ) -> list[SpanEvent]:
        spans: list[SpanEvent] = []
        counter = 0

        def visit(service_name: str, trace_id: str, parent_id: str | None, depth: int) -> None:
            nonlocal counter
            counter += 1
            span_id = f"s{step:03d}-{counter:04d}"
            state = states[service_name]
            failed = self._rng.random() < state.error_rate
            start_timestamp = true_time + timedelta(milliseconds=depth)
            duration_ms = max(0.1, state.latency_ms * self._rng.uniform(0.9, 1.1))
            spans.append(
                SpanEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_id,
                    service=service_name,
                    operation=f"{service_name}.handle_request",
                    start_timestamp=start_timestamp,
                    arrival_timestamp=start_timestamp
                    + timedelta(milliseconds=duration_ms),
                    duration_ms=duration_ms,
                    status="ERROR" if failed else "OK",
                )
            )
            for dependency in self._service_map[service_name].dependencies:
                visit(dependency, trace_id, span_id, depth + 1)

        for root in self.config.root_services:
            for sample in range(self.config.traces_per_root_per_step):
                visit(root, f"trace-{step:03d}-{root}-{sample}", None, 0)
        return spans
