# Experiment-readiness checklist

The CPU-only implementation is frozen at the following boundary.

## Completed

- Bench2 Workflow environment, closed-loop controller, deterministic checker and serial/MAS suite.
- Bench3 Incident environment, evidence/causal chain checker, closed-loop controller and serial/MAS suite.
- Minecraft task pool with 300 specifications: 100 construction, 100 cooking, 25 escape and 75 MAS-hard.
- Minecraft topology-held-out train/dev/test split: 180/50/70.
- Minecraft world configuration, initial inventory, roles, typed action specs, events, deadlines and concrete checker fields.
- 900 oracle plans and 1,800 labelled synthetic adapter-level replays.
- CPU-only symbolic Minecraft adapter with reset/observe/step/score API.
- VillagerAgent adapter contract with injected client methods and world-state checker.
- Shared success-first metrics and trace format.
- Data audits, topology leakage checks, deadline calibration checks and unit tests.

## Experiment-only dependencies

- A running VillagerAgent/Minecraft server for real block/entity observations and trajectories.
- GPU or API access for model inference.
- Full model matrix: Qwen, OpenAI, Anthropic and Gemini; Single Agent, centralized and MAS controllers.

Run the no-GPU readiness check with:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/run_all_sanity.py
```

The symbolic and deterministic results are calibration evidence. They should
not be used as final model leaderboard numbers until the real Minecraft
adapter and model controllers have produced world trajectories.
