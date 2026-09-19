"""Run bounded inference on local RCAEval-compatible case directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.adapters.rcaeval import (
    EXPECTED_LAYOUT,
    RCAEvalCaseInput,
    RCAEvalLayoutError,
    RCAEvalTruth,
    discover_case_directories,
    load_rcaeval_case,
)
from src.core import TelemetryQueryAPI
from src.rca import DiagnosisAgent, DiagnosisResult, DiagnosisStatus
from src.remediation import PlanningOutcome, PlanningStatus, SafeRemediationPlanner


SUPPORTED_EXTERNAL_FAMILIES = {
    "cpu": "cpu_saturation",
    "delay": "network_latency",
}


@dataclass(frozen=True, slots=True)
class RCAEvalInference:
    diagnosis: DiagnosisResult
    planning: PlanningOutcome
    total_budget_spent: int


@dataclass(frozen=True, slots=True)
class RCAEvalCaseResult:
    case_id: str
    external_fault_family: str
    mapped_failure_mode: str | None
    true_root_service: str
    predicted_root_service: str | None
    predicted_failure_mode: str | None
    diagnosis_status: str
    root_localization_correct: bool
    mode_correct: bool | None
    exact_pair_correct: bool | None
    unsupported_abstention: bool | None
    incorrect_known_mode: bool | None
    automatic_action: bool
    total_budget_spent: int


@dataclass(frozen=True, slots=True)
class RCAEvalSummary:
    case_count: int
    root_localization_accuracy: float
    supported_mapped_case_count: int
    mapped_mode_accuracy: float
    mapped_exact_pair_accuracy: float
    unsupported_case_count: int
    unsupported_abstention_rate: float
    unsupported_incorrect_known_mode_rate: float
    unsupported_automatic_action_rate: float


def infer_rcaeval_case(
    case_input: RCAEvalCaseInput, *, budget: int = 17
) -> RCAEvalInference:
    """Run inference without accepting or accessing external ground truth."""
    api = TelemetryQueryAPI(case_input.telemetry, budget)
    diagnosis = DiagnosisAgent().diagnose(
        api,
        case_input.baseline,
        case_input.analysis_start,
        case_input.analysis_end,
    )
    planning = SafeRemediationPlanner().plan(diagnosis, case_input.baseline)
    return RCAEvalInference(diagnosis, planning, api.spent_budget)


def evaluate_rcaeval_case(
    case_input: RCAEvalCaseInput,
    truth: RCAEvalTruth,
    *,
    budget: int = 17,
) -> RCAEvalCaseResult:
    inference = infer_rcaeval_case(case_input, budget=budget)
    diagnosis = inference.diagnosis
    predicted = diagnosis.best_hypothesis
    predicted_root = (
        predicted.service if predicted is not None else diagnosis.localized_service
    )
    predicted_mode = predicted.failure_mode if predicted is not None else None
    mapped_mode = SUPPORTED_EXTERNAL_FAMILIES.get(truth.fault_family.lower())
    supported = mapped_mode is not None
    root_correct = predicted_root == truth.root_service
    planned = inference.planning.status is PlanningStatus.PLANNED
    return RCAEvalCaseResult(
        case_id=case_input.case_id,
        external_fault_family=truth.fault_family,
        mapped_failure_mode=mapped_mode,
        true_root_service=truth.root_service,
        predicted_root_service=predicted_root,
        predicted_failure_mode=predicted_mode,
        diagnosis_status=diagnosis.status.value,
        root_localization_correct=root_correct,
        mode_correct=(predicted_mode == mapped_mode) if supported else None,
        exact_pair_correct=(root_correct and predicted_mode == mapped_mode)
        if supported
        else None,
        unsupported_abstention=(
            diagnosis.status is DiagnosisStatus.UNSUPPORTED
        )
        if not supported
        else None,
        incorrect_known_mode=(
            diagnosis.status is DiagnosisStatus.RESOLVED
            and predicted_mode is not None
        )
        if not supported
        else None,
        automatic_action=planned,
        total_budget_spent=inference.total_budget_spent,
    )


def summarize_rcaeval_results(
    results: Sequence[RCAEvalCaseResult],
) -> RCAEvalSummary:
    if not results:
        raise ValueError("cannot summarize an empty RCAEval result set")
    mapped = [item for item in results if item.mapped_failure_mode is not None]
    unsupported = [item for item in results if item.mapped_failure_mode is None]
    return RCAEvalSummary(
        case_count=len(results),
        root_localization_accuracy=_rate(
            sum(item.root_localization_correct for item in results), len(results)
        ),
        supported_mapped_case_count=len(mapped),
        mapped_mode_accuracy=_rate(sum(bool(item.mode_correct) for item in mapped), len(mapped)),
        mapped_exact_pair_accuracy=_rate(
            sum(bool(item.exact_pair_correct) for item in mapped), len(mapped)
        ),
        unsupported_case_count=len(unsupported),
        unsupported_abstention_rate=_rate(
            sum(bool(item.unsupported_abstention) for item in unsupported),
            len(unsupported),
        ),
        unsupported_incorrect_known_mode_rate=_rate(
            sum(bool(item.incorrect_known_mode) for item in unsupported),
            len(unsupported),
        ),
        unsupported_automatic_action_rate=_rate(
            sum(item.automatic_action for item in unsupported), len(unsupported)
        ),
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        directories = discover_case_directories(args.case_dir)
        bundles = tuple(load_rcaeval_case(path) for path in directories)
    except RCAEvalLayoutError as error:
        raise SystemExit(f"{error}\n{EXPECTED_LAYOUT}") from error
    results = tuple(
        evaluate_rcaeval_case(bundle.case_input, bundle.truth, budget=args.budget)
        for bundle in bundles
    )
    payload = {
        "summary": asdict(summarize_rcaeval_results(results)),
        "cases": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
