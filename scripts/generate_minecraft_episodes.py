"""Create replayable oracle/negative supervision episodes from task plans.

These episodes are symbolic adapter-level data. They are useful for training a
planner/controller before the Minecraft server is available, but are explicitly
marked as synthetic and must not be reported as executed world trajectories.
"""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data"
tasks = {json.loads(line)["task_id"]: json.loads(line) for line in (ROOT / "minecraft_tasks.jsonl").open() if line.strip()}
plans = [json.loads(line) for line in (ROOT / "minecraft_reference_plans.jsonl").open() if line.strip()]
episodes = []
for plan in plans:
    task = tasks[plan["task_id"]]
    actions = copy.deepcopy(plan["actions"])
    for variant in ("oracle", "stale_handoff"):
        trace = []
        completed = []
        for step, action in enumerate(actions):
            observation = {"tick": step, "completed_subtasks": list(completed),
                          "available_subtasks": [s["id"] for s in task["subtasks"] if s["id"] not in completed],
                          "world_config": task["world_config"], "inventory": task["initial_inventory"]}
            emitted = copy.deepcopy(action)
            delta = {"completed_subtask": action.get("subtask_id")} if action.get("subtask_id") else {}
            if variant == "stale_handoff" and step == max(0, len(actions) // 2) and action["type"] != "submit":
                emitted["type"] = "handoff"
                emitted["recipient"] = "agent_missing"
                delta = {"error": "invalid_recipient"}
            trace.append({"observation": observation, "action": emitted, "state_delta": delta})
            if action.get("subtask_id") and variant == "oracle":
                completed.append(action["subtask_id"])
        success = variant == "oracle"
        episodes.append({"episode_id": "%s__%s__%s" % (plan["task_id"], plan["controller"], variant),
                         "task_id": plan["task_id"], "split": plan["split"],
                         "topology_id": task["topology_id"], "family": plan["family"],
                         "controller": plan["controller"], "variant": variant,
                         "synthetic_oracle_replay": True, "success": success,
                         "failure_reason": None if success else "invalid_recipient",
                         "trace": trace})
with (ROOT / "minecraft_episodes.jsonl").open("w") as handle:
    for episode in episodes:
        handle.write(json.dumps(episode, sort_keys=True) + "\n")
manifest_path = ROOT / "minecraft_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest.update({"synthetic_replays": len(episodes), "synthetic_replay_variants": ["oracle", "stale_handoff"]})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps({"episodes": len(episodes), "tasks": len(tasks), "status": "ok"}, sort_keys=True))
