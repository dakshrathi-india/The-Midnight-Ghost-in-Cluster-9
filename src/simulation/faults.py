"""Behavioural fault effects used by the discrete-time simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_FAULT_PARAMETERS: dict[str, dict[str, float]] = {
    "cpu_saturation": {
        "capacity_factor": 0.35,
        "latency_factor": 1.8,
        "cpu_floor": 0.96,
    },
    "deployment_regression": {"latency_factor": 2.3, "error_addition": 0.22},
    "database_slowdown": {"latency_factor": 7.0},
    "connection_exhaustion": {"capacity_factor": 0.65, "error_addition": 0.42},
    "network_latency": {"latency_add_ms": 180.0},
    "process_crash": {"capacity_factor": 0.0, "error_addition": 1.0},
}


@dataclass(frozen=True, slots=True)
class FaultEffects:
    capacity_factor: float = 1.0
    latency_factor: float = 1.0
    latency_add_ms: float = 0.0
    error_addition: float = 0.0
    available: bool = True
    cpu_floor: float = 0.0


def effects_for(failure_mode: str, parameters: Mapping[str, float]) -> FaultEffects:
    """Translate a configured fault into understandable simulator effects."""
    if failure_mode == "cpu_saturation":
        return FaultEffects(
            capacity_factor=parameters.get("capacity_factor", 0.35),
            latency_factor=parameters.get("latency_factor", 1.8),
            cpu_floor=parameters.get("cpu_floor", 0.96),
        )
    if failure_mode == "deployment_regression":
        return FaultEffects(
            latency_factor=parameters.get("latency_factor", 2.3),
            error_addition=parameters.get("error_addition", 0.22),
        )
    if failure_mode == "database_slowdown":
        return FaultEffects(latency_factor=parameters.get("latency_factor", 7.0))
    if failure_mode == "connection_exhaustion":
        return FaultEffects(
            capacity_factor=parameters.get("capacity_factor", 0.65),
            error_addition=parameters.get("error_addition", 0.42),
        )
    if failure_mode == "network_latency":
        return FaultEffects(latency_add_ms=parameters.get("latency_add_ms", 180.0))
    if failure_mode == "process_crash":
        return FaultEffects(capacity_factor=0.0, error_addition=1.0, available=False)
    raise ValueError(f"unsupported failure mode: {failure_mode}")


def resolve_parameters(
    failure_mode: str, overrides: Mapping[str, float]
) -> dict[str, float]:
    try:
        resolved = dict(DEFAULT_FAULT_PARAMETERS[failure_mode])
    except KeyError as error:
        raise ValueError(f"unsupported failure mode: {failure_mode}") from error
    resolved.update(overrides)
    return resolved


def describe_fault(failure_mode: str) -> str:
    messages = {
        "cpu_saturation": "CPU saturation reduced processing capacity",
        "deployment_regression": "deployment regression increased request failures",
        "database_slowdown": "database operations exceeded normal latency",
        "connection_exhaustion": "connection pool exhausted while serving requests",
        "network_latency": "network round-trip latency increased",
        "process_crash": "service process became unavailable",
    }
    return messages[failure_mode]
