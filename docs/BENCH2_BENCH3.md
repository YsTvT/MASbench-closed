# Bench2 / Bench3 executable protocol

## Bench2: LongHorizon Workflow

Bench2 adapts the compositional-workflow idea behind WorkArena++ and the simulated-company setup of TheAgentCompany into a small, deterministic environment. A case contains independent branches (`report`, `dashboard`, and at higher difficulty `quality`/`security`), a joint approval gate, and a final release. Every branch has an owner, preconditions, an output artifact, a duration, and a verifier.

Actions are `inspect`, `execute`, `approve`, `handoff`, `message`, and `submit`. The environment reserves output locks at dispatch, applies version checks at completion, delays messages by one tick, and rejects stale or conflicting work. This makes parallelism necessary for the harder cases without making a task depend on a particular plan.

The reference baseline is a single serial controller. MAS runs use the same worker pool and action/message budgets. The primary score is success; completion ticks are aggregated only over successful runs. The checker verifies every branch artifact, approval, release, deadline, and absence of stale writes.

## Bench3: Incident Diagnosis

Bench3 adapts AIOpsLab's fault-injection and telemetry/verification loop. Each case hides one root cause, exposes target symptoms plus plausible distractor telemetry, and requires evidence collection, diagnosis, mitigation, verification, communication, and submission. Higher difficulty adds distractor signal families, so serial diagnosis performs more evidence queries while MAS can batch them. Query actions can run in parallel; mitigation and verification require evidence and the correct causal chain.

Actions are `query`, `diagnose`, `mitigate`, `verify`, `communicate`, `message`, and `submit`. A case is successful only when the root cause, mitigation, post-mitigation checks, and communication are all correct. Partial diagnostic progress is retained as a diagnostic metric but never replaces the binary success metric.

## Data quality

`scripts/generate_longhorizon_data.py` produces 223 deterministic cases aligned with the original benchmark scales: 175 workflow cases (TheAgentCompany scale) and 48 incident cases (AIOpsLab paper scale). The partitions are genuinely disjoint and stratified by family and difficulty: 133 train, 33 dev, and 57 held-out test cases. Each case includes a reference causal/dependency graph, deterministic state checker, and an executable baseline. A data audit checks: no duplicate IDs or split leakage, all preconditions are satisfiable, every root branch is independently useful, at least two branches can overlap, distractors are plausible but non-solutional, signals match the hidden fault, and every success has a finite completion tick.

The same model, worker count, token/action budget, tool permissions, and deadline are used for serial and MAS comparisons. Report success rate with a confidence interval first, then successful-run mean/median completion ticks. Do not average failed-run time.

The no-GPU closed-loop controller is implemented in
`masbench.longhorizon.closed_loop`; run `scripts/run_all_sanity.py` to audit
all three benchmark families and regenerate the CPU sanity manifest. Model
calls and real Minecraft server rollouts are intentionally outside this sanity
step and remain the final experiment stage.
