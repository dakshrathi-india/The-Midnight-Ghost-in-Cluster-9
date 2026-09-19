"""Production-facing telemetry ingestion adapters."""

from .opentelemetry import (
    OpenTelemetryAdapter,
    OpenTelemetryBatch,
    OpenTelemetryParseError,
)

__all__ = [
    "OpenTelemetryAdapter",
    "OpenTelemetryBatch",
    "OpenTelemetryParseError",
]
