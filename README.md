# Budget-Aware Root-Cause Analysis Foundation

This project is the foundation of an autonomous root-cause analysis (RCA) system for cascading failures in distributed microservices. It currently turns fragmented, budgeted telemetry into independently inspectable observability evidence and service-level anomaly candidates. It does not yet select a causal root cause.

## What is implemented

- Typed canonical `MetricEvent`, `LogEvent`, and `SpanEvent` models.
- Backend-independent service, severity, unit, timestamp, and duplicate normalization.
- A configuration-driven, discrete-time microservice simulator with queueing, capacity, retries, dependency propagation, and bounded noise.
- Six behavioural fault families: CPU saturation, deployment regression, database slowdown, connection exhaustion, network latency, and process crash.
- Reproducible metrics, logs, and parent/child traces with configurable clock skew, missing or delayed observations, metric noise, and decoy anomalies.
- Historical robust metric summaries, immutable healthy metric history, and optional cached dependency edges in a separate `BaselineStore`.
- Evaluation-only ground truth kept outside diagnosis-facing telemetry.
- A controlled `TelemetryQueryAPI` with one global budget, configurable per-query costs, exact-query caching, and query history.
- Generic service dependency reconstruction and per-service failure/latency evidence from parent-child spans.
- Independent MAD, CUSUM, and unsupervised multivariate Isolation Forest anomaly signals with raw evidence.
- Deterministic structured and TF-IDF semantic log evidence.
- Exact detector-count candidate generation with trace/log corroboration promotion.

Failure signatures, `(service, failure_mode)` hypotheses, causal ranking, active query planning, remediation, recovery verification, UI, and OpenTelemetry adapters are not implemented yet.

## Architecture

`src/core` contains generic telemetry and access primitives. `src/simulation` owns topology, fault injection, synthetic state, telemetry imperfections, and hidden truth. `src/rca` consumes only queried canonical events and historical baseline context to reconstruct trace topology, calculate three separate anomaly signals, interpret log evidence, and apply the frozen candidate rule. Ground truth remains an evaluation-only sibling object.

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
src/rca/           trace graph, anomaly signals, log evidence, candidates
src/simulation/    benchmark configuration, faults, simulator, incident generator
tests/             focused foundation and observability tests
main.py            deterministic foundation and observability demonstration
approach.md        implemented boundaries and planned RCA pipeline
```
