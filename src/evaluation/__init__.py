"""Controlled evaluation harness; ground truth is joined only after inference."""

from .ablations import Ablation, build_agent
from .models import (
    BenchmarkIncidentResult,
    BenchmarkSummary,
    HealthyControlResult,
    HealthyControlSummary,
    summarize_healthy_results,
    summarize_results,
)
from .profiles import RobustnessProfile, fragmentation_for

__all__ = [
    "Ablation",
    "BenchmarkIncidentResult",
    "BenchmarkSummary",
    "HealthyControlResult",
    "HealthyControlSummary",
    "RobustnessProfile",
    "build_agent",
    "fragmentation_for",
    "summarize_healthy_results",
    "summarize_results",
]
