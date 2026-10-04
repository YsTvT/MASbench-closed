# Minecraft task protocol

The task file is independent of the world backend. `minecraft_tasks.jsonl`
contains 300 graph-structured task specifications: 100 constructions, 100
cooking tasks, 25 escape tasks and 75 MAS-hard extensions. Every case has a
held-out split, world seed, deadline, explicit subtask dependencies, parallel
groups, typed action specs, world configuration, agent roles and programmatic
success checks. The baseline is `villageragent_serial`.

The action interface is deliberately small: `observe`, `move`, `gather`,
`craft`, `cook`, `place`, `attack`, `trade`, `handoff`, `deposit` and `submit`. A VillagerAgent or
MAS controller emits these actions; the Minecraft adapter translates them to
the actual server API. The checker evaluates the final world state and only
then exposes completion time.

The expanded pool has 180 train, 50 dev and 70 test cases. Splits are assigned
by topology family, so a blueprint/recipe/room topology is not copied from
train into test. Construction varies blueprint style, material family and
size; cooking uses explicit crafting/furnace prerequisites; escape varies
switch counts and timing windows; the MAS-hard extension adds dynamic routes,
resource competition, handoffs and resets. At least two root subtasks can run
in parallel in every case. Run
`scripts/audit_minecraft_tasks.py` before connecting a world server.

`minecraft_reference_plans.jsonl` contains three oracle plans per task
(serial, parallel and handoff), 900 plans total. These are supervision/replay
plans, not claims of completed Minecraft trajectories. Once the VillagerAgent
adapter is connected, generate actual observation-action-state-delta episodes
from these plans and keep those episodes split by task topology.

The current `minecraft_episodes.jsonl` contains 1,800 **synthetic adapter-level
replays**: one oracle and one stale-handoff negative for each of the 900 plans.
They are useful for testing the planner/controller interface, but are not
reported as real Minecraft rollouts. Real trajectories must be generated after
resetting the world server and must include block/entity state deltas.

The real Mineflayer dependency is installed in the server's vendored
VillagerAgent tree. Run the connection smoke after a Minecraft server is
listening on port 25565:

```bash
cd /path/to/VillagerAgent
node /path/to/MASbench/scripts/mineflayer_connect_smoke.js
```

The smoke currently reaches the real Mineflayer client and reports
`ECONNREFUSED` when no world server is running; no fake success is recorded.

The CPU-only symbolic adapter is implemented in `masbench.minecraft.env` and
can run all 300 task specs before the real server is available:

```bash
PYTHONPATH=MASbench python3 MASbench/scripts/run_minecraft_symbolic_suite.py \
  --data-dir MASbench/data --out MASbench/results/minecraft_symbolic
```

Its normal-deadline sanity result is serial/MAS SR `0.420/1.000` for
construction, `0.710/1.000` for cooking, `0.560/1.000` for escape and
`0.333/1.000` for MAS-hard. This validates task graph execution and deadline
calibration; it is not a Minecraft model leaderboard result.
