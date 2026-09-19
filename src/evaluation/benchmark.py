"""Deterministic benchmark runner and command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

from src.rca import DiagnosisAgent, DiagnosisStatus
from src.remediation import RecoveryController, RecoveryControllerConfig
from src.simulation import (
    FaultRequest,
    IncidentGenerator,
    SimulationConfig,
    SimulatorRemediationExecutor,
    benchmark_config,
)

from .ablations import Ablation, build_agent
from .models import BenchmarkIncidentResult, BenchmarkSummary, summarize_results
from .profiles import RobustnessProfile, fragmentation_for


@dataclass(frozen=True, slots=True, order=True)
class FaultCase:
    service: str
    failure_mode: str


@dataclass(frozen=True, slots=True)
class BudgetCurveRow:
    budget: int
    incident_count: int
    exact_pair_accuracy: float
    resolved_rate: float
    incomplete_budget_rate: float
    mean_query_usage: float


def enumerate_fault_cases(
    config: SimulationConfig,
    services: Sequence[str] | None = None,
    failure_modes: Sequence[str] | None = None,
) -> tuple[FaultCase, ...]:
    requested_services = set(services or ())
    requested_modes = set(failure_modes or ())
    known_services = {service.name for service in config.services}
    unknown_services = requested_services - known_services
    if unknown_services:
        raise ValueError(f"unknown benchmark services: {sorted(unknown_services)}")
    known_modes = {
        mode for service in config.services for mode in service.valid_failure_modes
    }
    unknown_modes = requested_modes - known_modes
    if unknown_modes:
        raise ValueError(f"unknown benchmark failure modes: {sorted(unknown_modes)}")
    return tuple(
        sorted(
            FaultCase(service.name, failure_mode)
            for service in config.services
            if not requested_services or service.name in requested_services
            for failure_mode in service.valid_failure_modes
            if not requested_modes or failure_mode in requested_modes
        )
    )


class BenchmarkRunner:
    def __init__(
        self,
        config: SimulationConfig,
        *,
        total_budget: int = 17,
        profile: str = "CUSTOM",
        ablation: Ablation = Ablation.FULL_SYSTEM,
        verification_steps: int = 150,
        agent_factory: Callable[[], DiagnosisAgent] | None = None,
    ) -> None:
        if total_budget < 0:
            raise ValueError("total budget cannot be negative")
        if verification_steps < 1:
            raise ValueError("verification steps must be positive")
        self.config = config
        self.total_budget = total_budget
        self.profile = profile
        self.ablation = ablation
        self.verification_steps = verification_steps
        self._agent_factory = agent_factory or (lambda: build_agent(ablation))

    def run(
        self,
        seeds: Sequence[int],
        cases: Sequence[FaultCase] | None = None,
    ) -> tuple[BenchmarkIncidentResult, ...]:
        selected_cases = tuple(
            enumerate_fault_cases(self.config) if cases is None else cases
        )
        if not seeds:
            raise ValueError("at least one benchmark seed is required")
        if not selected_cases:
            raise ValueError("at least one valid fault case is required")
        agent = self._agent_factory()
        controller = RecoveryController(
            diagnosis_agent=agent,
            config=RecoveryControllerConfig(self.verification_steps, 1),
        )
        generator = IncidentGenerator(self.config)
        results: list[BenchmarkIncidentResult] = []
        for seed in seeds:
            for case in selected_cases:
                session = generator.start_session(
                    seed, FaultRequest(case.service, case.failure_mode)
                )
                incident = session.initial_incident
                analysis_start = max(
                    event.event_timestamp
                    for event in incident.baseline.metric_history
                ) + timedelta(microseconds=1)
                api = session.create_query_api(self.total_budget)
                run = controller.run(
                    api,
                    incident.baseline,
                    analysis_start,
                    incident.observed_end_time,
                    SimulatorRemediationExecutor(session.simulator),
                    session,
                )

                # Evaluation joins hidden truth only after the agent/controller finish.
                truth = incident.ground_truth
                predicted = run.diagnosis.best_hypothesis
                resolved = run.diagnosis.status is DiagnosisStatus.RESOLVED
                predicted_service = predicted.service if predicted else None
                predicted_mode = predicted.failure_mode if predicted else None
                exact = resolved and (
                    predicted_service,
                    predicted_mode,
                ) == (truth.root_service, truth.failure_mode)
                action = run.planning.action
                recovery = run.verification
                remediation_correct = action is not None and (
                    action.target_service,
                    action.diagnosed_failure_mode,
                ) == (truth.root_service, truth.failure_mode)
                results.append(
                    BenchmarkIncidentResult(
                        profile=self.profile,
                        ablation=self.ablation.value,
                        seed=seed,
                        true_root_service=truth.root_service,
                        true_failure_mode=truth.failure_mode,
                        diagnosis_status=run.diagnosis.status.value,
                        predicted_service=predicted_service,
                        predicted_failure_mode=predicted_mode,
                        exact_pair_correct=exact,
                        root_service_correct=resolved
                        and predicted_service == truth.root_service,
                        failure_mode_correct=resolved
                        and predicted_mode == truth.failure_mode,
                        diagnosis_query_budget_spent=run.diagnosis.budget_spent,
                        total_query_budget_spent=api.spent_budget,
                        planned_remediation_action=(
                            action.action_type.value if action else None
                        ),
                        recovery_status=recovery.status.value if recovery else None,
                        remediation_target_correct=remediation_correct,
                        verification_window_count=len(run.observed_windows),
                        follow_up_diagnosis_status=(
                            run.follow_up_diagnosis.status.value
                            if run.follow_up_diagnosis
                            else None
                        ),
                    )
                )
        return tuple(results)


def budget_curve(
    config: SimulationConfig,
    seeds: Sequence[int],
    cases: Sequence[FaultCase],
    budgets: Sequence[int],
    *,
    profile: str,
) -> tuple[BudgetCurveRow, ...]:
    rows: list[BudgetCurveRow] = []
    expected_keys = {(seed, case.service, case.failure_mode) for seed in seeds for case in cases}
    for budget in budgets:
        results = BenchmarkRunner(
            config,
            total_budget=budget,
            profile=profile,
        ).run(seeds, cases)
        observed_keys = {
            (item.seed, item.true_root_service, item.true_failure_mode)
            for item in results
        }
        if observed_keys != expected_keys:
            raise RuntimeError("budget curve did not evaluate the same incident matrix")
        summary = summarize_results(
            results, profile=profile, ablation=Ablation.FULL_SYSTEM.value
        )
        rows.append(
            BudgetCurveRow(
                budget,
                summary.incident_count,
                summary.exact_pair_accuracy,
                summary.resolved_rate,
                summary.incomplete_budget_rate,
                summary.mean_total_query_cost,
            )
        )
    return tuple(rows)


def write_outputs(
    output_dir: Path,
    benchmark_results: Sequence[BenchmarkIncidentResult],
    summaries: Sequence[BenchmarkSummary],
    ablation_summaries: Sequence[BenchmarkSummary] = (),
    curve_rows: Sequence[BudgetCurveRow] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(
        output_dir / "benchmark_results.csv",
        BenchmarkIncidentResult,
        benchmark_results,
    )
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {"summaries": [summary.as_dict() for summary in summaries]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_dataclass_csv(
        output_dir / "ablation_results.csv",
        BenchmarkSummary,
        ablation_summaries,
    )
    _write_dataclass_csv(
        output_dir / "budget_curve.csv",
        BudgetCurveRow,
        curve_rows,
    )


def _write_dataclass_csv(
    path: Path,
    record_type: type,
    rows: Iterable[object],
) -> None:
    names = [field.name for field in fields(record_type)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in names})


def _print_summaries(title: str, summaries: Sequence[BenchmarkSummary]) -> None:
    print(f"\n{title}")
    print("profile/ablation                 n  exact  resolved  verified  mean cost  false resolved")
    for summary in summaries:
        label = (
            summary.profile
            if summary.ablation == Ablation.FULL_SYSTEM.value
            else summary.ablation
        )
        print(
            f"{label:<32} {summary.incident_count:>3} "
            f"{summary.exact_pair_accuracy:>6.1%} "
            f"{summary.resolved_rate:>8.1%} "
            f"{summary.verified_rate:>8.1%} "
            f"{summary.mean_total_query_cost:>9.2f} "
            f"{summary.false_resolution_count:>14}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--budget", type=int, default=17)
    parser.add_argument(
        "--profiles",
        nargs="+",
        type=RobustnessProfile,
        default=[RobustnessProfile.CLEAN],
        choices=list(RobustnessProfile),
    )
    parser.add_argument("--services", nargs="+")
    parser.add_argument("--failure-modes", nargs="+")
    parser.add_argument(
        "--ablations",
        nargs="+",
        type=Ablation,
        choices=list(Ablation),
    )
    parser.add_argument("--budget-curve", nargs="+", type=int)
    parser.add_argument("--verification-steps", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = perf_counter()
    all_results: list[BenchmarkIncidentResult] = []
    summaries: list[BenchmarkSummary] = []
    cases_by_profile: dict[RobustnessProfile, tuple[FaultCase, ...]] = {}
    for profile in args.profiles:
        config = benchmark_config(fragmentation_for(profile))
        cases = enumerate_fault_cases(config, args.services, args.failure_modes)
        cases_by_profile[profile] = cases
        results = BenchmarkRunner(
            config,
            total_budget=args.budget,
            profile=profile.value,
            verification_steps=args.verification_steps,
        ).run(args.seeds, cases)
        all_results.extend(results)
        summaries.append(
            summarize_results(
                results,
                profile=profile.value,
                ablation=Ablation.FULL_SYSTEM.value,
            )
        )

    first_profile = args.profiles[0]
    first_config = benchmark_config(fragmentation_for(first_profile))
    first_cases = cases_by_profile[first_profile]
    ablation_summaries: list[BenchmarkSummary] = []
    for ablation in args.ablations or ():
        results = BenchmarkRunner(
            first_config,
            total_budget=args.budget,
            profile=first_profile.value,
            ablation=ablation,
            verification_steps=args.verification_steps,
        ).run(args.seeds, first_cases)
        ablation_summaries.append(
            summarize_results(
                results,
                profile=first_profile.value,
                ablation=ablation.value,
            )
        )

    curve_rows = (
        budget_curve(
            first_config,
            args.seeds,
            first_cases,
            args.budget_curve,
            profile=first_profile.value,
        )
        if args.budget_curve
        else ()
    )
    write_outputs(
        args.output_dir,
        all_results,
        summaries,
        ablation_summaries,
        curve_rows,
    )
    _print_summaries("Overall diagnosis and recovery", summaries)
    if ablation_summaries:
        _print_summaries("Ablations", ablation_summaries)
    if curve_rows:
        print("\nBudget curve")
        print("budget  exact  resolved  incomplete  mean usage")
        for row in curve_rows:
            print(
                f"{row.budget:>6} {row.exact_pair_accuracy:>6.1%} "
                f"{row.resolved_rate:>8.1%} {row.incomplete_budget_rate:>10.1%} "
                f"{row.mean_query_usage:>10.2f}"
            )
    print(f"\nArtifacts: {args.output_dir}")
    print(f"Runtime: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
