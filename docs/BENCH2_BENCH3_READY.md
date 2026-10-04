# Bench2 / Bench3 ready checklist

## Bench2: Workflow

The executable environment is `masbench.longhorizon.workflow.WorkflowEnv`. It models independent report/dashboard branches, optional quality/security branches, a joint approval gate, and a release task. Dispatch-time locks and version checks reject conflicting or stale writes.

- Data: `data/longhorizon_train.jsonl` (133), `data/longhorizon_dev.jsonl` (33), `data/longhorizon_test.jsonl` (57).
- Actions: `inspect`, `execute`, `approve`, `handoff`, `message`, `submit`.
- Success: every branch artifact, approval, release, deadline, and checker condition must pass.
- Time: only successful episodes contribute completion ticks; composite score is `SR * (1 - successful_mean_ticks / deadline)`.
- Baseline: `workflow.serial_baseline`; MAS uses the same case, worker budget, deadline, and checker.

## Bench3: Incident

The executable environment is `masbench.longhorizon.incident.IncidentEnv`. It hides one root cause, exposes causal and distractor telemetry, and requires diagnosis, mitigation, verification, communication, and submission.

- Data: the same mutually exclusive train/dev/test files; 175 workflow and 48 incident cases in total.
- Actions: `query`, `diagnose`, `mitigate`, `verify`, `communicate`, `message`, `submit`.
- Success: correct root cause, correct mitigation, post-mitigation verification, communication, deadline, and submission.
- Baseline: `incident.serial_baseline`; parallel query batches are available to MAS under the same budgets and checker.

## Quality gates

Run from the repository root:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/audit_longhorizon_data.py
PYTHONPATH=MASbench python3 MASbench/scripts/run_all_sanity.py
PYTHONPATH=MASbench python3 -m unittest discover -s MASbench/tests -v
```

The current local audit reports 223 cases, no split leakage, and all three difficulties in every split. The complete test suite passes 12 tests. The deterministic closed-loop suite reports both serial and multi-agent traces without requiring a model or GPU.

Model experiments must read these frozen JSONL files and write new result directories; they must not overwrite the source data or checker.

