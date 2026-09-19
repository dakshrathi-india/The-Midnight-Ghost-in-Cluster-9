"""Professional demonstration console for the budget-aware causal RCA system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, cast

import altair as alt
import pandas as pd
import streamlit as st

from src.core import TelemetryQueryAPI
from src.evaluation import RobustnessProfile, fragmentation_for
from src.rca import (
    CandidateStrength,
    CausalEvidence,
    DiagnosisStatus,
    EvidenceCategory,
    EvidenceStatus,
    FailureSignatureLibrary,
    HypothesisEvaluation,
)
from src.remediation import (
    RecoveryController,
    RecoveryRunResult,
)
from src.simulation import (
    FaultRequest,
    Incident,
    IncidentGenerator,
    SimulationConfig,
    SimulatorRemediationExecutor,
    benchmark_config,
)
from src.ui import (
    BenchmarkArtifacts,
    EVALUATION_TRUTH_LABEL,
    candidate_rows,
    discover_benchmark_artifacts,
    evidence_rows,
    hypothesis_rows,
    load_benchmark_artifacts,
    prominent_evidence,
    query_history_rows,
    service_failure_mode_options,
    telemetry_signal_rows,
    topology_dot,
)


@dataclass(frozen=True, slots=True)
class DemoExecution:
    config: SimulationConfig
    incident: Incident
    run: RecoveryRunResult
    api: TelemetryQueryAPI
    analysis_start: datetime
    seed: int
    requested_service: str
    requested_failure_mode: str
    profile: RobustnessProfile


def execute_incident(
    seed: int,
    service: str,
    failure_mode: str,
    budget: int,
    profile: RobustnessProfile,
) -> DemoExecution:
    config = benchmark_config(fragmentation_for(profile))
    session = IncidentGenerator(config).start_session(
        seed=seed,
        fault=FaultRequest(service=service, failure_mode=failure_mode),
    )
    incident = session.initial_incident
    analysis_start = max(
        event.event_timestamp for event in incident.baseline.metric_history
    ) + timedelta(microseconds=1)
    api = session.create_query_api(total_budget=budget)
    run = RecoveryController().run(
        api,
        incident.baseline,
        analysis_start,
        incident.observed_end_time,
        SimulatorRemediationExecutor(session.simulator),
        session,
    )
    return DemoExecution(
        config,
        incident,
        run,
        api,
        analysis_start,
        seed,
        service,
        failure_mode,
        profile,
    )


def main() -> None:
    st.set_page_config(
        page_title="Cluster 9 | Incident RCA",
        page_icon=":material/troubleshoot:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _compact_shell()
    st.session_state.setdefault("demo_execution", None)
    st.session_state.setdefault("editing_setup", False)

    base_config = benchmark_config()
    options = service_failure_mode_options(base_config)
    existing = cast(DemoExecution | None, st.session_state.get("demo_execution"))
    section = _page_header(existing)
    seed, service, failure_mode, profile, budget, run_pressed = _setup_toolbar(
        options, existing
    )

    if run_pressed:
        if failure_mode not in options[service]:
            st.warning(
                f"{failure_mode} is not supported by {service}. "
                "Choose a compatible service and failure mode."
            )
        else:
            with st.spinner("Investigating telemetry and verifying recovery..."):
                st.session_state["demo_execution"] = execute_incident(
                    seed, service, failure_mode, budget, profile
                )
            st.session_state["editing_setup"] = False
            st.rerun()

    execution = cast(DemoExecution | None, st.session_state.get("demo_execution"))
    if execution is None:
        _landing_state(base_config)
        return

    if section == "Overview":
        _overview(execution)
    elif section == "Investigation":
        _investigation(execution)
    else:
        _evaluation(execution)


def _setup_toolbar(
    options: dict[str, tuple[str, ...]],
    execution: DemoExecution | None,
) -> tuple[int, str, str, RobustnessProfile, int, bool]:
    if execution is not None and not st.session_state["editing_setup"]:
        summary_column, edit_column = st.columns(
            (10, 1), gap="small", vertical_alignment="center"
        )
        with summary_column:
            st.caption(
                "TEST CASE · "
                f"{execution.requested_service} / {execution.requested_failure_mode} · "
                f"{execution.profile.value} · seed {execution.seed}"
            )
        with edit_column:
            if st.button("Edit", type="tertiary", width="stretch"):
                st.session_state["editing_setup"] = True
                st.rerun()
        return (
            execution.seed,
            execution.requested_service,
            execution.requested_failure_mode,
            execution.profile,
            execution.api.total_budget,
            False,
        )

    services = tuple(options)
    all_modes = tuple(sorted({mode for modes in options.values() for mode in modes}))
    profiles = tuple(RobustnessProfile)
    default_service = execution.requested_service if execution else "postgres"
    default_mode = (
        execution.requested_failure_mode if execution else "database_slowdown"
    )
    default_profile = execution.profile if execution else RobustnessProfile.DEFAULT
    default_seed = execution.seed if execution else 17
    default_budget = execution.api.total_budget if execution else 17
    with st.form("simulation-setup", border=False):
        st.caption(
            "TEST CASE · SIMULATION — Fault labels configure the simulator only. "
            "RCA receives telemetry."
        )
        controls = st.columns(
            (2.0, 2.3, 1.6, 0.8, 0.9, 1.25),
            gap="small",
            vertical_alignment="bottom",
        )
        with controls[0]:
            service = st.selectbox(
                "Service", services, index=services.index(default_service)
            )
        with controls[1]:
            failure_mode = st.selectbox(
                "Failure mode", all_modes, index=all_modes.index(default_mode)
            )
        with controls[2]:
            profile = st.selectbox(
                "Profile",
                profiles,
                index=profiles.index(default_profile),
                format_func=lambda item: item.value,
            )
        with controls[3]:
            seed = int(st.number_input("Seed", min_value=0, value=default_seed, step=1))
        with controls[4]:
            budget = int(
                st.number_input(
                    "Budget",
                    min_value=1,
                    max_value=100,
                    value=default_budget,
                    step=1,
                )
            )
        with controls[5]:
            run_pressed = st.form_submit_button(
                "Run incident", type="primary", width="stretch"
            )
    return seed, service, failure_mode, profile, budget, run_pressed


def _compact_shell() -> None:
    st.html("""
        <style>
        div[data-testid="stMainBlockContainer"] {
            width: calc(100% - 48px) !important;
            max-width: 1320px !important;
            padding-top: 7rem !important;
            padding-bottom: 2.5rem !important;
        }
        </style>
        """)


def _page_header(execution: DemoExecution | None) -> str:
    identity, navigation, status = st.columns(
        (3.1, 4.7, 2.2), gap="medium", vertical_alignment="center"
    )
    with identity:
        st.subheader("Cluster 9")
        st.caption("Incident Investigator")
    with navigation:
        selected = st.segmented_control(
            "Workspace section",
            ("Overview", "Investigation", "Evaluation"),
            default="Overview",
            key="workspace-section",
            label_visibility="collapsed",
            width="stretch",
            wrap=False,
        )
    with status:
        if execution is not None:
            recovery = execution.run.verification
            recovery_status = recovery.status.value if recovery else "NOT VERIFIED"
            st.markdown(
                f"**{execution.run.diagnosis.status.value}** · "
                f"**{recovery_status}**",
                text_alignment="right",
            )
    st.divider()
    return str(selected or "Overview")


def _landing_state(config: SimulationConfig) -> None:
    st.space("small")
    topology, prompt = st.columns((7, 4), gap="large", vertical_alignment="center")
    with topology:
        st.caption("SERVICE TOPOLOGY")
        st.graphviz_chart(topology_dot(config), width="stretch", height=390)
    with prompt:
        st.header("Ready to investigate")
        st.write("Run a synthetic incident to generate telemetry for the RCA engine.")


def _overview(execution: DemoExecution) -> None:
    diagnosis = execution.run.diagnosis
    diagnosed = diagnosis.best_hypothesis
    localized_service = diagnosis.localized_service
    affected_services = tuple(
        candidate.service
        for candidate in diagnosis.candidates
        if candidate.strength is not CandidateStrength.NOT_CANDIDATE
    )
    root = diagnosed.service if diagnosed else localized_service or "Unresolved"
    failure_mode = (
        diagnosed.failure_mode
        if diagnosed
        else (
            "Unknown / unsupported"
            if diagnosis.status is DiagnosisStatus.UNSUPPORTED
            else "No failure mode"
        )
    )

    st.space("small")
    map_column, diagnosis_column = st.columns((8, 4), gap="large")
    with map_column:
        st.caption("FAILURE PROPAGATION")
        st.graphviz_chart(
            topology_dot(
                execution.config,
                diagnosed.service if diagnosed else localized_service,
                affected_services,
            ),
            width="stretch",
            height=390,
        )
        st.caption(
            ":violet[Root cause] · :orange[Affected] · :gray[Healthy]     "
            "Direction: caller → callee"
        )

    with diagnosis_column:
        with st.container(border=True, height=430):
            st.caption("INCIDENT LOCALIZED")
            st.title(root)
            st.caption(failure_mode)
            st.subheader("Why this diagnosis")
            if diagnosis.ranked_hypotheses:
                best = diagnosis.ranked_hypotheses[0]
                selected_evidence = _diagnosis_evidence(best)
                for index, evidence in enumerate(selected_evidence, start=1):
                    number, detail = st.columns((0.6, 4.4), gap="small")
                    with number:
                        st.caption(f"{index:02d}")
                    with detail:
                        st.caption(_evidence_label(evidence.category))
                        st.write(evidence.observation)
                        if evidence.status is EvidenceStatus.CONTRADICTION:
                            st.markdown(":red[CONTRADICTION]")
                st.caption(
                    f"{best.ranking.contradiction_count} contradictions · "
                    f"{best.ranking.direct_support_count} direct modalities"
                )
            else:
                st.caption("No causal evidence was produced.")

    root_service = diagnosed.service if diagnosed else localized_service
    if root_service is not None:
        _telemetry_panel(execution, root_service)
    _response_and_budget(execution)


def _diagnosis_evidence(
    evaluation: HypothesisEvaluation,
) -> tuple[CausalEvidence, ...]:
    evidence = prominent_evidence(evaluation)
    contradictions = tuple(
        item for item in evidence if item.status is EvidenceStatus.CONTRADICTION
    )
    supports = tuple(item for item in evidence if item.status is EvidenceStatus.SUPPORT)
    support_limit = max(0, 3 - len(contradictions))
    return (*supports[:support_limit], *contradictions[:3])


def _evidence_label(category: EvidenceCategory) -> str:
    return {
        EvidenceCategory.METRIC_PATTERN: "Metric shift",
        EvidenceCategory.LOG_SEMANTIC: "Semantic log evidence",
        EvidenceCategory.TRACE_LOCALIZATION: "Trace localization",
        EvidenceCategory.TEMPORAL_PRECEDENCE: "Temporal order",
        EvidenceCategory.GRAPH_PROPAGATION: "Causal propagation",
        EvidenceCategory.CANDIDATE_STRENGTH: "Candidate evidence",
        EvidenceCategory.INTERVENTION_OUTCOME: "Intervention outcome",
    }[category]


def _telemetry_panel(execution: DemoExecution, root_service: str) -> None:
    baseline = tuple(
        event
        for event in execution.incident.baseline.metric_history
        if event.service == root_service
    )
    incident = execution.incident.telemetry.metrics_between(
        root_service,
        execution.analysis_start,
        execution.incident.observed_end_time,
    )
    post = ()
    if execution.run.observed_windows:
        post = execution.incident.telemetry.metrics_between(
            root_service,
            execution.run.observed_windows[0].start_time,
            execution.run.observed_windows[-1].end_time,
        )
    rows = telemetry_signal_rows(
        (("Baseline", baseline), ("Incident", incident), ("Post-remediation", post))
    )

    st.space("medium")
    st.subheader("Telemetry")
    st.caption(f"Root-service signals · {root_service}")
    if not rows:
        st.caption("No root-service metric samples are available.")
        return
    visible_metrics = _visible_signal_metrics(execution, rows)
    columns = st.columns(len(visible_metrics), gap="medium")
    for index, (column, metric_name) in enumerate(
        zip(columns, visible_metrics, strict=True)
    ):
        signal_rows = tuple(row for row in rows if row["Metric"] == metric_name)
        with column:
            st.caption(f"{signal_rows[0]['Signal']} · {signal_rows[0]['Unit']}")
            _telemetry_chart(signal_rows, show_legend=index == 0)


def _telemetry_chart(rows: Sequence[dict[str, object]], *, show_legend: bool) -> None:
    frame = pd.DataFrame(rows)
    legend = alt.Legend(orient="bottom", title=None, columns=3) if show_legend else None
    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X(
                "Event time:T",
                title=None,
                axis=alt.Axis(grid=False, labelAngle=0, tickCount=4),
            ),
            y=alt.Y(
                "Value:Q",
                title=None,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True, tickCount=4),
            ),
            color=alt.Color(
                "Period:N",
                scale=alt.Scale(
                    domain=("Baseline", "Incident", "Post-remediation"),
                    range=("#8E8B84", "#5754C8", "#4F7A5A"),
                ),
                legend=legend,
            ),
            tooltip=(
                alt.Tooltip("Event time:T", title="Event time"),
                alt.Tooltip("Period:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ),
        )
        .properties(height=205)
    )
    st.altair_chart(chart, width="stretch")


def _visible_signal_metrics(
    execution: DemoExecution,
    rows: Sequence[dict[str, object]],
) -> tuple[str, ...]:
    available = {str(row["Metric"]) for row in rows}
    preferred = [
        "request_latency_ms",
        "error_rate",
        "cpu_utilization",
        "request_rate",
    ]
    hypothesis = execution.run.diagnosis.best_hypothesis
    if hypothesis is not None:
        signature = FailureSignatureLibrary().get(hypothesis.failure_mode)
        defining = [
            expectation.metric_name
            for expectation in signature.metric_expectations
            if expectation.required and expectation.metric_name in available
        ]
        preferred = defining + [name for name in preferred if name not in defining]
    selected = [name for name in preferred if name in available][:4]
    if len(selected) < 4:
        selected.extend(sorted(available - set(selected))[: 4 - len(selected)])
    return tuple(selected)


def _investigation(execution: DemoExecution) -> None:
    ranked = execution.run.diagnosis.ranked_hypotheses
    if not ranked:
        st.caption("No causal hypotheses were produced.")
        return

    st.space("small")
    ranked_column, explanation_column = st.columns((7, 5), gap="large")
    with ranked_column:
        st.subheader("Ranked hypotheses")
        ranked_rows = tuple(
            {
                "#": row["#"],
                "Hypothesis": f"{row['Service']} / {row['Failure Mode']}",
                "Contradictions": row["Contradictions"],
                "Explained": row["Explained"],
                "Direct": row["Direct Support"],
            }
            for row in hypothesis_rows(execution.run.diagnosis)
        )
        st.dataframe(
            ranked_rows,
            hide_index=True,
            width="stretch",
            height=300,
            key="ranked-hypotheses",
        )
    with explanation_column:
        _winner_explanation(ranked[0], ranked[1] if len(ranked) > 1 else None)

    st.space("medium")
    detail = st.segmented_control(
        "Investigation detail",
        ("Evidence", "Candidates", "Queries"),
        default="Evidence",
        key=f"investigation-detail-{execution.incident.incident_id}",
        label_visibility="collapsed",
    )
    if detail == "Candidates":
        candidates = tuple(
            row
            for row in candidate_rows(execution.run.diagnosis)
            if row["Strength"] != "NOT_CANDIDATE"
        )
        st.subheader("Candidate services")
        if candidates:
            st.dataframe(
                candidates,
                hide_index=True,
                width="stretch",
                height=min(330, 36 + 35 * len(candidates)),
                column_config={
                    "Corroborated": st.column_config.CheckboxColumn("Corroborated")
                },
                key="candidate-table",
            )
        else:
            st.caption("No candidate services were identified.")
    elif detail == "Queries":
        query_rows = query_history_rows(execution.api.query_history)
        st.subheader("Telemetry queries")
        st.caption(
            f"{execution.api.spent_budget} / {execution.api.total_budget} budget used"
        )
        if query_rows:
            st.dataframe(
                query_rows,
                hide_index=True,
                width="stretch",
                height=330,
                key="query-history",
            )
        else:
            st.caption("No telemetry queries were executed.")
    else:
        st.subheader("Causal evidence")
        evidence_frame = pd.DataFrame(evidence_rows(ranked[0]))
        st.dataframe(
            evidence_frame.style.apply(_evidence_row_style, axis=1),
            hide_index=True,
            width="stretch",
            height=330,
            key="causal-evidence",
        )


def _winner_explanation(
    selected: HypothesisEvaluation,
    runner: HypothesisEvaluation | None,
) -> None:
    st.caption("WHY THIS WON")
    st.caption("SELECTED")
    st.markdown(
        f"**{selected.hypothesis.service} / {selected.hypothesis.failure_mode}**"
    )
    if runner is None:
        st.caption("No runner-up hypothesis was produced.")
        return

    st.caption("RUNNER-UP")
    st.write(f"{runner.hypothesis.service} / {runner.hypothesis.failure_mode}")
    separation = _first_separating_criterion(selected, runner)
    st.caption("FIRST SEPARATING CRITERION")
    if separation is None:
        st.write("Causal evidence tied; deterministic service/mode order followed.")
    else:
        label, selected_value, runner_value = separation
        st.markdown(f"**{label}**")
        values = st.columns(2, gap="small")
        with values[0]:
            st.caption("Selected")
            st.write(str(selected_value))
        with values[1]:
            st.caption("Runner-up")
            st.write(str(runner_value))

    unique_support = _unique_support(selected, runner)
    if unique_support is not None:
        st.caption(f"+ {unique_support.observation}")
    if runner.contradictions:
        st.caption(f"− {runner.contradictions[0].observation}")


def _first_separating_criterion(
    selected: HypothesisEvaluation,
    runner: HypothesisEvaluation,
) -> tuple[str, int, int] | None:
    criteria = (
        (
            "Contradictions",
            selected.ranking.contradiction_count,
            runner.ranking.contradiction_count,
        ),
        (
            "Unexplained strong candidates",
            selected.ranking.unexplained_strong_count,
            runner.ranking.unexplained_strong_count,
        ),
        (
            "Explained strong candidates",
            selected.ranking.explained_strong_count,
            runner.ranking.explained_strong_count,
        ),
        (
            "Supporting evidence types",
            selected.ranking.supporting_category_count,
            runner.ranking.supporting_category_count,
        ),
        (
            "Direct support modalities",
            selected.ranking.direct_support_count,
            runner.ranking.direct_support_count,
        ),
    )
    return next(
        (criterion for criterion in criteria if criterion[1] != criterion[2]),
        None,
    )


def _unique_support(
    selected: HypothesisEvaluation,
    runner: HypothesisEvaluation,
) -> CausalEvidence | None:
    runner_evidence = {
        (item.category, item.service, item.observation)
        for item in runner.supporting_evidence
    }
    return next(
        (
            item
            for item in selected.supporting_evidence
            if (item.category, item.service, item.observation) not in runner_evidence
        ),
        None,
    )


def _evidence_row_style(row: pd.Series) -> list[str]:
    color = {
        "SUPPORT": "background-color: #E8F1EA",
        "CONTRADICTION": "background-color: #F7E9E7",
        "NEUTRAL": "color: #77736C",
    }.get(str(row["Verdict"]), "")
    return [color] * len(row)


def _response_and_budget(execution: DemoExecution) -> None:
    run = execution.run
    hypothesis = run.diagnosis.best_hypothesis
    localized_service = run.diagnosis.localized_service
    action = run.planning.action
    verification = run.verification
    action_value = action.action_type.value if action else "ACTION BLOCKED"
    if (
        action is not None
        and run.execution is not None
        and run.execution.status.value != "APPLIED"
    ):
        action_value = f"{action.action_type.value} / {run.execution.status.value}"
    st.space("medium")
    response_column, budget_column = st.columns((9, 3), gap="large")
    stages = (
        (
            "Diagnosis →",
            (
                f"{hypothesis.service} / {hypothesis.failure_mode}"
                if hypothesis
                else (
                    f"{localized_service} / Unknown / unsupported"
                    if run.diagnosis.status is DiagnosisStatus.UNSUPPORTED
                    and localized_service is not None
                    else "Not resolved"
                )
            ),
        ),
        ("Safety gate →", "Passed" if action is not None else "Blocked"),
        ("Action →", action_value),
        (
            "Fresh telemetry →",
            (
                "Observed"
                if verification is not None and verification.fresh_window is not None
                else "Not requested"
            ),
        ),
        (
            "Recovery",
            verification.status.value if verification else "NOT VERIFIED",
        ),
    )
    with response_column:
        st.caption("AUTONOMOUS RESPONSE")
        stage_columns = st.columns(len(stages), gap="small")
        for column, (label, value) in zip(stage_columns, stages, strict=True):
            with column:
                st.caption(label)
                st.markdown(f"**{value}**")
        st.caption("Execution alone is not recovery.")

    with budget_column:
        spent = execution.api.spent_budget
        total = execution.api.total_budget
        st.caption("QUERY BUDGET")
        st.markdown(f"**{spent} / {total}**")
        st.progress(min(spent / total, 1.0))
        diagnosis_spent = run.diagnosis.budget_spent
        verification_spent = verification.budget_spent if verification else 0
        st.caption(f"Diagnosis {diagnosis_spent} · Verification {verification_spent}")


def _evaluation(execution: DemoExecution) -> None:
    st.space("small")
    st.caption("Synthetic evaluation — not a production accuracy claim.")
    directories = discover_benchmark_artifacts(Path("outputs"))
    artifacts = tuple(_load_cached_artifact(directory) for directory in directories)
    if not artifacts:
        st.caption("No generated benchmark artifacts are available.")
        _ground_truth(execution)
        return

    robustness = _best_robustness_artifact(artifacts)
    ablations = max(artifacts, key=lambda item: len(item.ablations))
    curve = max(artifacts, key=lambda item: len(item.budget_curve))

    robustness_rows = tuple(
        {
            "profile": row.get("profile"),
            "exact accuracy": row.get("exact_pair_accuracy"),
            "resolved": row.get("resolved_rate"),
            "verified": row.get("verified_rate"),
            "false resolutions": row.get("false_resolution_count"),
        }
        for row in robustness.summaries
        if row.get("ablation") == "FULL_SYSTEM"
    )
    clean_row = next(
        (row for row in robustness_rows if row.get("profile") == "CLEAN"),
        None,
    )
    clean_column, benchmark_column = st.columns((3, 9), gap="large")
    with clean_column:
        st.caption("CLEAN BENCHMARK")
        if clean_row is not None and clean_row["exact accuracy"] is not None:
            st.header(f"{float(clean_row['exact accuracy']):.1%}")
            st.caption("Exact RCA")
        else:
            st.header("—")
            st.caption("Exact RCA unavailable")
    with benchmark_column:
        st.subheader("Benchmark profile")
        if robustness_rows:
            st.bar_chart(
                robustness_rows,
                x="profile",
                y="exact accuracy",
                x_label="Profile",
                y_label="Exact pair accuracy",
                height=220,
            )
        else:
            st.caption("No robustness summary is available.")

    st.space("medium")
    robustness_column, ablation_column = st.columns((7, 5), gap="large")
    with robustness_column:
        st.subheader("Robustness")
    if robustness_rows:
        with robustness_column:
            st.dataframe(
                robustness_rows,
                hide_index=True,
                width="stretch",
                height=300,
                column_config={
                    "exact accuracy": st.column_config.NumberColumn(format="percent"),
                    "resolved": st.column_config.NumberColumn(format="percent"),
                    "verified": st.column_config.NumberColumn(format="percent"),
                },
                key="robustness-results",
            )
    else:
        with robustness_column:
            st.caption("No robustness summary is available.")

    ablation_rows = tuple(
        {
            "ablation": row.get("ablation"),
            "exact accuracy": row.get("exact_pair_accuracy"),
            "resolved": row.get("resolved_rate"),
            "verified": row.get("verified_rate"),
        }
        for row in ablations.ablations
    )
    with ablation_column:
        st.subheader("Ablation")
        if ablation_rows:
            ordered_ablations = tuple(
                sorted(
                    ablation_rows,
                    key=lambda row: float(row["exact accuracy"] or 0.0),
                )
            )
            st.bar_chart(
                ordered_ablations,
                x="ablation",
                y="exact accuracy",
                x_label="Exact pair accuracy",
                y_label="Ablation",
                horizontal=True,
                sort=False,
                height=300,
            )
        else:
            st.caption("No ablation results are available.")

    st.space("medium")
    st.subheader("Budget sensitivity")
    budget_rows = tuple(
        {
            "budget": row.get("budget"),
            "exact accuracy": row.get("exact_pair_accuracy"),
            "resolved": row.get("resolved_rate"),
            "incomplete": row.get("incomplete_budget_rate"),
            "mean usage": row.get("mean_query_usage"),
        }
        for row in curve.budget_curve
    )
    if budget_rows:
        chart_column, table_column = st.columns((5, 7), gap="large")
        with chart_column:
            st.line_chart(
                budget_rows,
                x="budget",
                y=("exact accuracy", "resolved", "incomplete"),
                x_label="Query budget",
                y_label="Rate",
                height=280,
            )
        with table_column:
            st.dataframe(
                budget_rows,
                hide_index=True,
                width="stretch",
                height=280,
                column_config={
                    "exact accuracy": st.column_config.NumberColumn(format="percent"),
                    "resolved": st.column_config.NumberColumn(format="percent"),
                    "incomplete": st.column_config.NumberColumn(format="percent"),
                },
                key="budget-curve",
            )
    else:
        st.caption("No budget curve is available.")

    _ground_truth(execution)


def _ground_truth(execution: DemoExecution) -> None:
    st.space("large")
    with st.expander(EVALUATION_TRUTH_LABEL):
        truth = execution.incident.ground_truth
        st.caption("Ground truth is joined only after inference.")
        st.table(
            {
                "Injected service": truth.root_service,
                "Injected failure mode": truth.failure_mode,
                "Decoys": ", ".join(truth.decoy_services) or "None",
            },
            border="horizontal",
        )


def _load_cached_artifact(directory: Path) -> BenchmarkArtifacts:
    signature = tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.glob("*"))
        if path.is_file()
    )
    return _cached_artifacts(str(directory), signature)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_artifacts(
    directory: str,
    file_signature: tuple[tuple[str, int, int], ...],
) -> BenchmarkArtifacts:
    del file_signature
    return load_benchmark_artifacts(Path(directory))


def _best_robustness_artifact(
    artifacts: Sequence[BenchmarkArtifacts],
) -> BenchmarkArtifacts:
    return max(
        artifacts,
        key=lambda item: len(
            {
                row.get("profile")
                for row in item.summaries
                if row.get("ablation") == "FULL_SYSTEM"
            }
        ),
    )


if __name__ == "__main__":
    main()
