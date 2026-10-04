# Paper experiment plan

Run three deadline regimes for Bench2 and Bench3:

```bash
PYTHONPATH=. python3 -m masbench.longhorizon.experiment --out results/longhorizon/normal --seeds 30 --deadline-scale 1.0
PYTHONPATH=. python3 -m masbench.longhorizon.experiment --out results/longhorizon/tight --seeds 30 --deadline-scale 0.75
PYTHONPATH=. python3 -m masbench.longhorizon.experiment --out results/longhorizon/extreme --seeds 30 --deadline-scale 0.60
# Held-out test pool (175 workflow + 48 incident total; 57 test cases)
PYTHONPATH=. python3 -m masbench.longhorizon.experiment --out results/longhorizon/full_test_tight --data-dir data --split test --deadline-scale 0.75
# Development pool, used only for prompt/deadline calibration
PYTHONPATH=. python3 -m masbench.longhorizon.experiment --out results/longhorizon/dev_tight --data-dir data --split dev --deadline-scale 0.75
# Closed-loop scripted sanity run with full trace and the same controller API
PYTHONPATH=. python3 - <<'PY'
from masbench.longhorizon.closed_loop import cases_from_jsonl, run_closed_loop_suite
run_closed_loop_suite(cases_from_jsonl('data', 'test'), 'results/longhorizon/closed_loop_test_1_0', 1.0)
PY
```

The only independent variable is the controller: `serial` is a single worker/controller baseline; `mas` dispatches independent ready branches concurrently and uses the same action budget, workers, state, deadline and deterministic checker. Each run is paired by task ID. Report success first. Completion time is defined only for successful runs. The JSONL trace preserves every action, failed precondition, delayed message and checker result for post-hoc responsibility analysis.

The paper should report **SR first**, with a Wilson 95% interval. If a run fails, its SR contribution is zero and its time is discarded completely. For successful runs only, report total completion ticks and deadline-normalized time. The optional composite score is:

```text
time_score = 1 - mean(completion_ticks / deadline | success)
composite = SR * time_score
```

Thus a failed run cannot receive time credit, while faster successful runs receive a larger composite score. Do not replace a failed run's time with the deadline or zero.

## Closed-loop deterministic sanity run

The closed-loop controller consumes observations and emits actions through the
same API that an LLM controller will use. On the held-out test split at the
normal deadline, workflow is `0.659/1.000` (serial/MAS), with successful mean
ticks `14.45/10`; incident is `0.615/1.000`, with successful mean ticks
`15/11.77`. At deadline scale `0.75`, serial SR is zero for both families,
while MAS remains `1.000` on workflow and `0.692` on incident. At `0.60`, the
MAS controller also begins to fail. These are environment calibration results,
not model leaderboard scores; they show that the deadline and dependency graph
now produce a controllable difficulty window.

The expanded data pool contains 175 workflow cases and 48 incident cases, partitioned into 133 train, 33 dev and 57 test rows with no task-ID overlap. Run `python3 scripts/audit_longhorizon_data.py` before every paper evaluation. Never tune prompts, deadlines or routing rules on the held-out test split.
