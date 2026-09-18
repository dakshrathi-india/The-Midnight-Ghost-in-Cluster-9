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

Canonical immutable events provide one contract for metrics, logs, and traces. A generic normalizer handles basic source variation without interpreting anomalies or log meaning. `BaselineStore` keeps immutable pre-incident context separate from incident evidence.

The simulation backend models discrete-time demand, finite capacity, queues, latency, CPU and memory utilization, failures, dependency propagation, and retry amplification. Faults change these behaviours rather than directly painting arbitrary bad metrics. Synthetic traces use recursively nested intervals with deterministic sequential sibling calls: each parent duration is local work plus its child dependency durations. The generator adds bounded noise and configurable fragmentation after simulation, then emits only canonical observed telemetry. Injected decoys are selected outside the root failure's structural caller closure, or omitted when no unrelated service exists.

`TelemetryQueryAPI` is the future RCA access boundary. It applies a single shared incident budget, uses configuration-driven costs, caches exact requests, and records every successful request including cache hits. A trace query is seeded by a service and time window, then returns every available span from each matching trace so parent/child context can cross service boundaries without additional query charges. Missing spans remain missing. A service summary costs once and is computed internally, rather than charging separate metric, log, and trace queries.

The observability layer reconstructs deterministic service edges solely from trace parent-child relationships and retains edge support plus per-service failure and latency evidence. Metric analysis produces three independent outputs: robust modified-z evidence from median/MAD, bidirectional CUSUM evidence for sustained shifts, and deterministic multivariate Isolation Forest evidence trained only on immutable healthy history. Isolation Forest considers only configured features present in both windows; its portable core is CPU, memory, request latency, error rate, and request rate, while queue length remains opt-in. The signals are not collapsed into a weighted score.

Log analysis preserves structured fields and original messages. Its primary semantic representation is a lazily loaded configurable sentence-transformer, with cached embeddings for multiple generic prototypes per category and cosine similarity over batched messages. It abstains with UNKNOWN below threshold; TF-IDF remains an explicitly labeled offline/failure fallback. Deterministic patterns separately extract only high-confidence facts such as durations, percentages, status codes, connection counts, queues, and retries. Candidate generation applies the frozen detector-count rule exactly: two signals are strong, one is weak unless corroborated by traces or high-severity categorized logs, and zero metric signals can never become a candidate.

The diagnosis layer declares generic metric, log, and trace expectations for all six failure modes and generates every applicable `(service, failure_mode)` pair from non-empty candidates. It evaluates each pair using interpretable metric direction, semantic logs, trace localization, tolerant onset ordering, and caller-to-dependency paths. When a reliable service metric onset exists, signature-specific semantic logs provide direct support only when their event timestamps align with that onset within the configured tolerance; otherwise matching logs remain neutral secondary evidence. If onset is unavailable, the earliest signature-specific match remains usable. Trace localization retains parent duration, dependency-duration bounds, and residual local time; incomplete or overlap-ambiguous evidence remains UNKNOWN or MIXED. Explicit parent/child identities establish causal structure even if cross-service clocks reverse apparent timestamps. Strong candidates that a root cannot explain remain explicit. Hypotheses are ordered lexicographically by contradictions, unexplained and explained strong candidates, independent evidence categories, and direct root evidence categories; individual observation counts do not affect ranking. Service and failure-mode names provide deterministic ordering only after the causal ranking key. No weighted final score is used.

The active planner compares the expected observations of surviving hypotheses before querying. It selects the affordable uncached metric, log, or trace query with the greatest pair separation per configured cost. The autonomous agent exposes its initial observation policy explicitly: the default exhaustive metric bootstrap is retained when affordable, while a topology-prioritized metric subset gathers deterministic partial evidence under smaller budgets and returns `INCOMPLETE_BUDGET`. After bootstrap, evaluation and targeted querying continue until the agent resolves a unique supported hypothesis, exhausts useful queries or budget, or finds no candidates.

The remediation layer is separate from RCA and simulation. A declarative policy maps each supported failure mode to one operational action. The safety planner emits an immutable single-service action only for a resolved, strong, contradiction-free hypothesis with direct root log or trace confirmation and a known target; every other outcome is explicitly blocked. A generic executor protocol reports whether a command was applied, rejected, or failed without claiming recovery.

The simulator implements that executor behind the generic boundary. Its original `run()` method and incremental continuation share one state-update path, preserving queues, prior service states, deterministic RNG state, and step numbering. A compatible action on the actual target clears the corresponding effect only for future steps. Wrong targets and incompatible actions can still be applied as commands but leave the hidden fault active, allowing later recovery verification to observe failure without exposing ground truth.

The append-only telemetry boundary keeps the same store, query budget, history, and valid cache entries as a stateful incident session advances the same simulator. Fresh raw events pass through the original fragmentation RNG, clock offsets, decoy policy, normalizer, and selective cache invalidation. Recovery verification queries only fresh metric windows for the diagnosed root, causally explained strong services, and one eligible lexical healthy sentinel. It requires all required root signature metrics to normalize, no strong residual affected-region candidate, continuing traffic, and a healthy sentinel. Missing, transient, or unaffordable evidence stays inconclusive. A decisive failure after an applied action becomes an `INTERVENTION_OUTCOME` contradiction on only the attempted pair before generic investigation runs again.

## Isolation decisions

Synthetic simulation is a benchmark backend, not the RCA algorithm. Simulator topology, internal state, true timing, fault injection, and service-specific validity rules live under `src/simulation`; the generic `src/core` layer can accept future OpenTelemetry-derived events without importing simulation code.

Ground truth is returned only as an evaluation object beside the telemetry store. It is never stored in or exposed by `TelemetryStore` or `TelemetryQueryAPI`. This prevents diagnosis code from accidentally learning the answer while still enabling deterministic evaluation.

Budgeting represents the operational cost and latency of fetching evidence in real systems. Exact-query caching prevents repeated reasoning steps from paying twice, while immutable history makes query decisions auditable.

## Planned, not implemented

Later stages will add OpenTelemetry ingestion, a user interface, and final benchmark reporting.
