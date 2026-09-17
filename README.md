# Budget-Aware Root-Cause Analysis Foundation

This project is the foundation of an autonomous root-cause analysis (RCA) system for cascading failures in distributed microservices. The intended system will reason over fragmented metrics, logs, and traces while spending a limited telemetry-query budget. This repository currently implements the data and simulation foundation only; it does not yet diagnose incidents.

## What is implemented

- Typed canonical `MetricEvent`, `LogEvent`, and `SpanEvent` models.
- Backend-independent service, severity, unit, timestamp, and duplicate normalization.
- A configuration-driven, discrete-time microservice simulator with queueing, capacity, retries, dependency propagation, and bounded noise.
- Six behavioural fault families: CPU saturation, deployment regression, database slowdown, connection exhaustion, network latency, and process crash.
- Reproducible metrics, logs, and parent/child traces with configurable clock skew, missing or delayed observations, metric noise, and decoy anomalies.
- Historical metric summaries and optional cached dependency edges in a separate `BaselineStore`.
- Evaluation-only ground truth kept outside diagnosis-facing telemetry.
- A controlled `TelemetryQueryAPI` with one global budget, configurable per-query costs, exact-query caching, and query history.

Anomaly detection, graph reconstruction, hypothesis generation, diagnosis, query planning, remediation, UI, and OpenTelemetry adapters are deliberately not implemented at this stage.

## Architecture

`src/core` contains generic telemetry and access primitives and knows nothing about benchmark service names or simulator fault internals. `src/simulation` owns topology, fault injection, synthetic state, telemetry imperfections, and hidden truth. An incident returns canonical telemetry and historical baseline context for future RCA code, while ground truth remains an evaluation-only sibling object.

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
src/simulation/    benchmark configuration, faults, simulator, incident generator
tests/             focused foundation tests
main.py            deterministic demonstration
approach.md        implemented boundaries and planned RCA pipeline
```
