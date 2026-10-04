# Bench2/Bench3 readiness

Bench2 and Bench3 are materialized as independent, executable benchmark pools.
The environments are deterministic state machines and the data are generated
by the same constructors used by the checker.

## Bench2: Workflow

`masbench.longhorizon.workflow.WorkflowEnv` models a long workflow with two to
four independent branches, a joint approval barrier, and a release stage.
`execute`, `approve`, `inspect`, `handoff`, `message`, and `submit` are the only
actions. Branch artifacts are locked on dispatch and precondition versions are
checked at completion. The serial baseline has one worker; MAS uses four
workers and the same action/message budgets.

Data: `data/bench2_workflow/{train,dev,test}.jsonl` (105/26/44 cases, 175 total).
Each case includes a dependency graph, deadline, critical path, parallel width,
scenario metadata, and a success contract. The released test pool contains all
three difficulty levels and has no task-ID overlap with train or dev.

## Bench3: Incident diagnosis

`masbench.longhorizon.incident.IncidentEnv` hides one root cause behind causal
signals and plausible distractor signals. Agents must query evidence, diagnose,
mitigate, verify post-mitigation telemetry, communicate, and submit. Queries can
be dispatched in parallel; diagnosis and mitigation are gated by evidence and
the checker requires the correct causal chain. The gold fault and fix remain
evaluation-only fields and are never included in the controller prompt.

Data: `data/bench3_incident/{train,dev,test}.jsonl` (28/7/13 cases, 48 total).
Each case includes service/severity context, signal budget, hidden causal chain,
required evidence, distractor families, and a success contract.

## Audit and run

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/generate_bench23_data.py
PYTHONPATH=MASbench python3 MASbench/scripts/audit_bench23_data.py
PYTHONPATH=MASbench python3 MASbench/scripts/run_bench23.py \
  --bench all --split test --out MASbench/results/bench23/test_normal
```

The audit verifies disjoint IDs, split consistency, stratification, dependency
integrity, deterministic MAS solvability, and the success-first metric contract.
The runner writes one JSONL trace and one summary per benchmark. A failed case
has `completion_ticks: null`; successful-run time alone contributes to the
conditional time statistic and the composite score is
`SR * (1 - mean(success_time / deadline | success))`.

The CPU deterministic test pool currently gives MAS SR 1.0 on both benchmarks;
this is an environment/controller calibration result, not an LLM result. Model
experiments must use the same cases and checker after prompt/model settings are
frozen.
