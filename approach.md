# Technical Approach

## Frozen end-to-end pipeline

The planned pipeline is:

```text
metrics / logs / traces
  -> telemetry normalization
  -> service and dependency understanding
  -> anomaly detection
  -> abnormal service candidates
  -> (service, failure_mode) hypotheses
  -> causal evidence-based diagnosis
  -> budget-aware active telemetry querying
  -> safe remediation
  -> recovery verification
```

The diagnosis target is always a `(service, failure_mode)` pair. Core RCA will remain topology- and service-name-agnostic and will not depend on Bayesian inference or a supervised model trained on synthetic incidents.

## Implemented

Canonical immutable events provide one contract for metrics, logs, and traces. A generic normalizer handles basic source variation without interpreting anomalies or log meaning. `BaselineStore` keeps pre-incident context separate from incident evidence and records whether historical access should count toward a future query budget.

The simulation backend models discrete-time demand, finite capacity, queues, latency, CPU and memory utilization, failures, dependency propagation, and retry amplification. Faults change these behaviours rather than directly painting arbitrary bad metrics. The generator adds bounded noise and configurable fragmentation after simulation, then emits only canonical observed telemetry.

`TelemetryQueryAPI` is the future RCA access boundary. It applies a single shared incident budget, uses configuration-driven costs, caches exact requests, and records every successful request including cache hits. A service summary costs once and is computed internally, rather than charging separate metric, log, and trace queries.

The observability layer reconstructs deterministic service edges solely from trace parent-child relationships and retains edge support plus per-service failure and latency evidence. Metric analysis produces three independent outputs: robust modified-z evidence from median/MAD, bidirectional CUSUM evidence for sustained shifts, and deterministic multivariate Isolation Forest evidence trained only on immutable healthy history. The signals are not collapsed into a weighted score.

Log analysis preserves structured fields and uses explainable TF-IDF similarity against generic failure-symptom prototypes. Candidate generation applies the frozen detector-count rule exactly: two signals are strong, one is weak unless corroborated by traces or high-severity logs, and zero metric signals can never become a candidate.

## Isolation decisions

Synthetic simulation is a benchmark backend, not the RCA algorithm. Simulator topology, internal state, true timing, fault injection, and service-specific validity rules live under `src/simulation`; the generic `src/core` layer can accept future OpenTelemetry-derived events without importing simulation code.

Ground truth is returned only as an evaluation object beside the telemetry store. It is never stored in or exposed by `TelemetryStore` or `TelemetryQueryAPI`. This prevents diagnosis code from accidentally learning the answer while still enabling deterministic evaluation.

Budgeting represents the operational cost and latency of fetching evidence in real systems. Exact-query caching prevents repeated reasoning steps from paying twice, while immutable history makes query decisions auditable.

## Planned, not implemented

Later stages will add failure signatures, `(service, failure_mode)` hypothesis generation, causal contradiction testing and ranking, active query selection, remediation safeguards, recovery verification, OpenTelemetry ingestion, and a user interface. None of those components is claimed as part of the current implementation.
