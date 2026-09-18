"""Configuration-driven synthetic benchmark and evaluation backend."""

from .config import (
    FaultRequest,
    FragmentationConfig,
    ServiceConfig,
    SimulationConfig,
    benchmark_config,
)
from .generator import GroundTruth, Incident, IncidentGenerator, SimulationIncidentSession
from .remediation import SimulatorRemediationExecutor

__all__ = [
    "FaultRequest",
    "FragmentationConfig",
    "GroundTruth",
    "Incident",
    "IncidentGenerator",
    "ServiceConfig",
    "SimulationConfig",
    "SimulationIncidentSession",
    "SimulatorRemediationExecutor",
    "benchmark_config",
]
