# Budget-Aware Root-Cause Analysis Foundation

This project implements a deterministic, budget-aware root-cause analysis (RCA) system for cascading failures in distributed microservices. It turns fragmented telemetry into observable service candidates, evaluates explainable `(service, failure_mode)` hypotheses, and actively fetches only useful additional evidence.

## What is implemented

- Typed canonical `MetricEvent`, `LogEvent`, and `SpanEvent` models.
- Backend-independent service, severity, unit, timestamp, and duplicate normalization.
- A configuration-driven, discrete-time microservice simulator with queueing, capacity, retries, dependency propagation, and bounded noise.
- Six behavioural fault families: CPU saturation, deployment regression, database slowdown, connection exhaustion, network latency, and process crash.
- Reproducible metrics, logs, and recursively consistent parent/child trace intervals with configurable clock skew, missing or delayed observations, metric noise, and structurally unrelated decoy anomalies.
- Historical robust metric summaries, immutable healthy metric history, and optional cached dependency edges in a separate `BaselineStore`.
- Evaluation-only ground truth kept outside diagnosis-facing telemetry.
- A controlled `TelemetryQueryAPI` with one global budget, configurable per-query costs, exact-query caching, query history, and service-seeded trace queries that return complete available trace trees.
- Generic service dependency reconstruction and per-service failure/latency evidence from parent-child spans.
- Independent MAD, CUSUM, and unsupervised multivariate Isolation Forest anomaly signals with raw evidence. Isolation Forest uses an explicit portable core feature policy with opt-in optional features such as queue length.
- Lazy sentence-transformer log semantics using `sentence-transformers/all-MiniLM-L6-v2`, multiple prototypes per category, explicit UNKNOWN abstention, and an explicit TF-IDF offline/failure fallback.
- Deterministic extraction of high-confidence log facts such as durations, percentages, codes, queue sizes, connection counts, and retries.
- Exact detector-count candidate generation with trace/log corroboration promotion.
- Declarative signatures for six generic failure modes.
- Structured metric, log, trace, temporal, and dependency-graph evidence with explicit contradictions and conservative local-versus-dependency trace localization.
- Generic `(service, failure_mode)` hypothesis generation and lexicographic ranking.
- Deterministic pair-separation query planning using configured query costs.
- An autonomous diagnosis loop with explicit exhaustive and topology-prioritized bootstrap policies plus resolved, ambiguous, incomplete-budget, and no-candidate outcomes.
- A simulator-independent safety-gated remediation planner with six declarative failure-mode policies and immutable execution contracts.
- Stateful simulator continuation and a simulator remediation executor that changes only future fault effects without resetting queues, prior states, or RNG state.

Recovery verification, diagnose-act-verify retries, final benchmark reporting, UI, and OpenTelemetry adapters are not implemented yet.

## Architecture

`src/core` contains generic telemetry and access primitives. `src/simulation` owns topology, fault injection, synthetic state, telemetry imperfections, hidden truth, and the simulator-specific remediation executor. `src/rca` consumes only queried canonical events and historical baseline context to detect candidates, evaluate causal hypotheses, rank them without a weighted score, and select additional queries under one global budget. `src/remediation` contains simulator-independent action, planning, and execution contracts. Ground truth remains an evaluation-only sibling object.

## Setup and run

Python 3.11 or newer is required.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python main.py
pytest
```

## Repository structure

```text
src/core/          canonical models, normalization, baselines, budget, query API
src/rca/           observability, signatures, causal diagnosis, planner, agent
src/remediation/   safe action planning and generic execution contracts
src/simulation/    benchmark configuration, faults, simulator, incident generator
tests/             focused foundation and observability tests
main.py            deterministic budget-aware diagnosis demonstration
approach.md        implemented boundaries and planned RCA pipeline
```
