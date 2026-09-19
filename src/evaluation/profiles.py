"""Named deterministic telemetry-fragmentation stress profiles."""

from __future__ import annotations

from enum import Enum

from src.simulation import FragmentationConfig


class RobustnessProfile(str, Enum):
    CLEAN = "CLEAN"
    DEFAULT = "DEFAULT"
    CLOCK_SKEW = "CLOCK_SKEW"
    MISSING = "MISSING"
    DELAYED = "DELAYED"
    NOISY = "NOISY"
    DECOY = "DECOY"
    COMBINED_STRESS = "COMBINED_STRESS"


def fragmentation_for(profile: RobustnessProfile) -> FragmentationConfig:
    profiles = {
        RobustnessProfile.CLEAN: FragmentationConfig(
            clock_skew_range_seconds=0,
            missing_observation_probability=0,
            delayed_observation_probability=0,
            metric_noise_fraction=0,
            include_decoy_anomaly=False,
        ),
        RobustnessProfile.DEFAULT: FragmentationConfig(),
        RobustnessProfile.CLOCK_SKEW: FragmentationConfig(
            clock_skew_range_seconds=15,
            missing_observation_probability=0,
            delayed_observation_probability=0,
            metric_noise_fraction=0,
            include_decoy_anomaly=False,
        ),
        RobustnessProfile.MISSING: FragmentationConfig(
            clock_skew_range_seconds=0,
            missing_observation_probability=0.15,
            delayed_observation_probability=0,
            metric_noise_fraction=0,
            include_decoy_anomaly=False,
        ),
        RobustnessProfile.DELAYED: FragmentationConfig(
            clock_skew_range_seconds=0,
            missing_observation_probability=0,
            delayed_observation_probability=0.35,
            max_delay_seconds=45,
            metric_noise_fraction=0,
            include_decoy_anomaly=False,
        ),
        RobustnessProfile.NOISY: FragmentationConfig(
            clock_skew_range_seconds=0,
            missing_observation_probability=0,
            delayed_observation_probability=0,
            metric_noise_fraction=0.08,
            include_decoy_anomaly=False,
        ),
        RobustnessProfile.DECOY: FragmentationConfig(
            clock_skew_range_seconds=0,
            missing_observation_probability=0,
            delayed_observation_probability=0,
            metric_noise_fraction=0,
            include_decoy_anomaly=True,
        ),
        RobustnessProfile.COMBINED_STRESS: FragmentationConfig(
            clock_skew_range_seconds=15,
            missing_observation_probability=0.15,
            delayed_observation_probability=0.35,
            max_delay_seconds=45,
            metric_noise_fraction=0.08,
            include_decoy_anomaly=True,
        ),
    }
    return profiles[profile]
