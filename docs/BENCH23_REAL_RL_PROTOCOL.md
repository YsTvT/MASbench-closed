# Bench2/Bench3 real-environment RL protocol

Bench2 (workflow) and Bench3 (incident) use the same train/test boundary as
Minecraft, but their adapters connect to an external workflow or incident
service.  The deterministic Python environments remain baselines and unit-test
fixtures only.

## Train batch

`train_real_bench23_rl.py` reads only `data/bench2_workflow/train.jsonl` or
`data/bench3_incident/train.jsonl`.  For each case it asks an injected adapter
factory to reset an isolated real environment, collects observations and typed
actions, waits for the checker, writes a trajectory, and updates the policy
after a batch.  The adapter must expose:

```python
backend_name = "company-sandbox-v1"
family = "workflow"  # or "incident"
real_rollout = True
```

It must implement `reset`, `observe`, `step_batch`, and `score`.  A symbolic
adapter is rejected before the first episode and is never persisted as a train
record.  The learner also rejects non-train splits and `unassessed` checker
results.

## Frozen test/eval batch

`eval_real_bench23_rl.py` reads only the disjoint `test.jsonl` files, loads a
checkpoint without calling `update`, and executes the test cases in fresh real
environments.  Evaluation accepts only checker rows with
`evaluation_status="evaluated"` and `real_rollout=true`; otherwise it fails
loudly instead of treating missing telemetry as a failure.  The report uses
the benchmark metric: SR first, completion time only among successful episodes,
and `composite = SR * (1 - mean(completion_ticks/deadline | success))`.

The authoritative terminal reward is 1 for checker success and 0 for checker
failure.  Intermediate checker progress can provide a small shaping signal to
the policy update, but cannot change the final reward or SR.

## Adapter boundary

The repository intentionally does not ship a fake "real" service.  A service
adapter should reset a fresh case, expose each worker/operator observation,
execute a batch atomically, and return the final checker score including
`success`, `checks`, and `completion_ticks` for successful cases.  This makes
the trainset auditable: every trajectory contains backend identity, case split,
all observations/actions/results, and the raw terminal checker payload.
