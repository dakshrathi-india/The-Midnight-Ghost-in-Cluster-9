"""Configuration for synthetic services, faults, and telemetry imperfections."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

SUPPORTED_FAILURE_MODES = frozenset(
    {
        "cpu_saturation",
        "deployment_regression",
        "database_slowdown",
        "connection_exhaustion",
        "network_latency",
        "process_crash",
    }
)


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    name: str
    capacity_rps: float
    base_latency_ms: float
    external_request_rate: float = 0.0
    dependencies: tuple[str, ...] = ()
    valid_failure_modes: frozenset[str] = SUPPORTED_FAILURE_MODES
    base_memory_utilization: float = 0.35

    def __post_init__(self) -> None:
        if not self.name or self.capacity_rps <= 0 or self.base_latency_ms <= 0:
            raise ValueError("services require a name, positive capacity, and positive latency")
        unknown_modes = self.valid_failure_modes - SUPPORTED_FAILURE_MODES
        if unknown_modes:
            raise ValueError(f"unsupported failure modes: {sorted(unknown_modes)}")


@dataclass(frozen=True, slots=True)
class FragmentationConfig:
    clock_skew_range_seconds: float = 4.0
    clock_offsets_seconds: Mapping[str, float] = field(default_factory=dict)
    missing_observation_probability: float = 0.02
    delayed_observation_probability: float = 0.05
    max_delay_seconds: float = 20.0
    metric_noise_fraction: float = 0.02
    include_decoy_anomaly: bool = True

    def __post_init__(self) -> None:
        probabilities = (
            self.missing_observation_probability,
            self.delayed_observation_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("telemetry probabilities must be in [0, 1]")
        if self.clock_skew_range_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("clock skew and delay bounds cannot be negative")
        if self.metric_noise_fraction < 0:
            raise ValueError("metric noise cannot be negative")
        object.__setattr__(
            self, "clock_offsets_seconds", MappingProxyType(dict(self.clock_offsets_seconds))
        )


@dataclass(frozen=True, slots=True)
class FaultRequest:
    service: str | None = None
    failure_mode: str | None = None
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    services: tuple[ServiceConfig, ...]
    baseline_steps: int = 10
    incident_steps: int = 15
    step_seconds: int = 30
    traces_per_root_per_step: int = 1
    retry_factor: float = 0.7
    fragmentation: FragmentationConfig = field(default_factory=FragmentationConfig)
    historical_context_counts_toward_budget: bool = False

    def __post_init__(self) -> None:
        if not self.services:
            raise ValueError("at least one service is required")
        if self.baseline_steps < 1 or self.incident_steps < 1 or self.step_seconds < 1:
            raise ValueError("simulation periods and step size must be positive")
        names = [service.name for service in self.services]
        if len(names) != len(set(names)):
            raise ValueError("service names must be unique")
        known = set(names)
        for service in self.services:
            unknown = set(service.dependencies) - known
            if unknown:
                raise ValueError(f"{service.name} has unknown dependencies: {sorted(unknown)}")
        self.topological_order()

    @property
    def service_map(self) -> dict[str, ServiceConfig]:
        return {service.name: service for service in self.services}

    @property
    def dependency_edges(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (service.name, dependency)
            for service in self.services
            for dependency in service.dependencies
        )

    @property
    def root_services(self) -> tuple[str, ...]:
        depended_on = {dependency for service in self.services for dependency in service.dependencies}
        configured_roots = tuple(
            service.name for service in self.services if service.external_request_rate > 0
        )
        return configured_roots or tuple(
            service.name for service in self.services if service.name not in depended_on
        )

    def topological_order(self) -> tuple[str, ...]:
        indegree = {service.name: 0 for service in self.services}
        outgoing: dict[str, list[str]] = {service.name: [] for service in self.services}
        for service in self.services:
            for dependency in service.dependencies:
                indegree[dependency] += 1
                outgoing[service.name].append(dependency)
        ready = [name for name, degree in indegree.items() if degree == 0]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for dependency in outgoing[current]:
                indegree[dependency] -= 1
                if indegree[dependency] == 0:
                    ready.append(dependency)
        if len(order) != len(self.services):
            raise ValueError("service dependency graph must be acyclic")
        return tuple(order)


def benchmark_config(fragmentation: FragmentationConfig | None = None) -> SimulationConfig:
    """Return the default nine-component benchmark topology."""
    general_modes = frozenset(
        {"cpu_saturation", "deployment_regression", "connection_exhaustion", "network_latency", "process_crash"}
    )
    datastore_modes = frozenset(
        {"cpu_saturation", "database_slowdown", "connection_exhaustion", "network_latency", "process_crash"}
    )
    services = (
        ServiceConfig("gateway", 180, 8, 105, ("auth", "checkout"), general_modes),
        ServiceConfig("auth", 145, 7, dependencies=("redis",), valid_failure_modes=general_modes),
        ServiceConfig("checkout", 140, 12, dependencies=("payment", "orders"), valid_failure_modes=general_modes),
        ServiceConfig("payment", 115, 18, dependencies=("postgres",), valid_failure_modes=general_modes),
        ServiceConfig("orders", 120, 15, dependencies=("postgres", "inventory"), valid_failure_modes=general_modes),
        ServiceConfig("inventory", 125, 10, dependencies=("catalog",), valid_failure_modes=general_modes),
        ServiceConfig("catalog", 135, 9, valid_failure_modes=general_modes),
        ServiceConfig("redis", 170, 3, valid_failure_modes=datastore_modes),
        ServiceConfig("postgres", 260, 6, valid_failure_modes=datastore_modes),
    )
    return SimulationConfig(services=services, fragmentation=fragmentation or FragmentationConfig())
