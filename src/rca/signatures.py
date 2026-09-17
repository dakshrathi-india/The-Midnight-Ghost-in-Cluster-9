"""Declarative, service-agnostic failure signatures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class MetricDirection(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class TraceExpectation(str, Enum):
    LOCAL_LATENCY = "LOCAL_LATENCY"
    DEPENDENCY_LATENCY = "DEPENDENCY_LATENCY"
    FAILURES = "FAILURES"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class MetricExpectation:
    metric_name: str
    direction: MetricDirection
    required: bool = True


@dataclass(frozen=True, slots=True)
class FailureSignature:
    failure_mode: str
    metric_expectations: tuple[MetricExpectation, ...]
    log_categories: frozenset[str]
    trace_expectations: frozenset[TraceExpectation]


class FailureSignatureLibrary:
    def __init__(self) -> None:
        signatures = (
            FailureSignature(
                "cpu_saturation",
                (
                    MetricExpectation("cpu_utilization", MetricDirection.HIGH),
                    MetricExpectation("queue_length", MetricDirection.HIGH, required=False),
                    MetricExpectation(
                        "request_latency_ms", MetricDirection.HIGH, required=False
                    ),
                ),
                frozenset({"resource_pressure"}),
                frozenset({TraceExpectation.LOCAL_LATENCY}),
            ),
            FailureSignature(
                "deployment_regression",
                (
                    MetricExpectation("error_rate", MetricDirection.HIGH),
                    MetricExpectation(
                        "request_latency_ms", MetricDirection.HIGH, required=False
                    ),
                ),
                frozenset({"deployment_regression"}),
                frozenset({TraceExpectation.FAILURES}),
            ),
            FailureSignature(
                "database_slowdown",
                (MetricExpectation("request_latency_ms", MetricDirection.HIGH),),
                frozenset({"database_slowdown"}),
                frozenset({TraceExpectation.LOCAL_LATENCY}),
            ),
            FailureSignature(
                "connection_exhaustion",
                (
                    MetricExpectation("error_rate", MetricDirection.HIGH),
                    MetricExpectation("queue_length", MetricDirection.HIGH, required=False),
                ),
                frozenset({"connection_exhaustion"}),
                frozenset({TraceExpectation.FAILURES}),
            ),
            FailureSignature(
                "network_latency",
                (MetricExpectation("request_latency_ms", MetricDirection.HIGH),),
                frozenset({"timeout_network_delay"}),
                frozenset(
                    {TraceExpectation.LOCAL_LATENCY, TraceExpectation.DEPENDENCY_LATENCY}
                ),
            ),
            FailureSignature(
                "process_crash",
                (MetricExpectation("error_rate", MetricDirection.HIGH),),
                frozenset({"process_unavailable"}),
                frozenset(
                    {TraceExpectation.FAILURES, TraceExpectation.UNAVAILABLE}
                ),
            ),
        )
        self._signatures: Mapping[str, FailureSignature] = MappingProxyType(
            {signature.failure_mode: signature for signature in signatures}
        )

    @property
    def failure_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self._signatures))

    def get(self, failure_mode: str) -> FailureSignature:
        try:
            return self._signatures[failure_mode]
        except KeyError as error:
            raise ValueError(f"unknown failure mode: {failure_mode}") from error
