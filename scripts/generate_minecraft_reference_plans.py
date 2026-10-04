"""Generate oracle plan supervision, separate from actual Minecraft trajectories."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data"


def make_plan(task, controller):
    subtasks = {s["id"]: s for s in task["subtasks"]}
    plan = []
    owners = {}
    if controller == "serial":
        for step, node in enumerate(n for group in task["parallel_groups"] for n in group):
            owners[node] = "agent_0"
            plan.append({"step": step, "agent": "agent_0", "type": subtasks[node]["action"],
                         "subtask_id": node, "depends_on": subtasks[node]["depends_on"]})
    else:
        for step, group in enumerate(task["parallel_groups"]):
            for j, node in enumerate(group):
                agent = "agent_%d" % (j % task["required_agents"])
                owners[node] = agent
                plan.append({"step": step, "agent": agent, "type": subtasks[node]["action"],
                             "subtask_id": node, "depends_on": subtasks[node]["depends_on"]})
        if controller == "handoff":
            for node in owners:
                for dep in subtasks[node]["depends_on"]:
                    if owners.get(dep) != owners[node]:
                        plan.append({"step": len(task["parallel_groups"]), "agent": owners[dep],
                                     "type": "handoff", "subtask_id": dep, "recipient": owners[node]})
    plan.append({"step": len(task["parallel_groups"]), "agent": "agent_0", "type": "submit"})
    return plan


rows = [json.loads(line) for line in (ROOT / "minecraft_tasks.jsonl").open() if line.strip()]
plans = []
for task in rows:
    for controller in ("serial", "parallel", "handoff"):
        plans.append({"task_id": task["task_id"], "family": task["subfamily"],
                      "split": task["split"], "controller": controller,
                      "world_seed": task["world_seed"], "oracle": True,
                      "actions": make_plan(task, controller)})
with (ROOT / "minecraft_reference_plans.jsonl").open("w") as handle:
    for row in plans:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
manifest_path = ROOT / "minecraft_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest.update({"reference_plans": len(plans), "reference_plan_controllers": ["serial", "parallel", "handoff"]})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps({"tasks": len(rows), "plans": len(plans), "status": "ok"}, sort_keys=True))
