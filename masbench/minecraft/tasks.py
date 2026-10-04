"""Model-independent Minecraft task schema.

The actual world adapter can be VillagerAgent, MineDojo or a local server. The
benchmark task JSON stays independent of rendering and model implementation.
"""

import json
import re
from pathlib import Path


ACTIONS = ("observe", "move", "gather", "craft", "cook", "place", "attack", "trade", "handoff", "deposit", "submit")


def validate_task(task):
    required = ("task_id", "family", "split", "difficulty", "deadline_ticks", "world_seed",
                "goal", "subtasks", "parallel_groups", "success_checks", "baseline",
                "topology_id", "world_config", "initial_inventory", "agent_roles",
                "action_specs", "concrete_success_checks", "serial_estimated_ticks",
                "critical_path_ticks", "environment_events")
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError("missing task fields: " + ",".join(missing))
    if task["family"] != "minecraft" or task["split"] not in ("train", "dev", "test"):
        raise ValueError("invalid family or split")
    if task["difficulty"] not in (1, 2, 3) or task["deadline_ticks"] <= 0:
        raise ValueError("invalid difficulty/deadline")
    if not task["topology_id"] or task["critical_path_ticks"] > task["deadline_ticks"]:
        raise ValueError("invalid topology or impossible critical path")
    subtasks = task["subtasks"]
    ids = [s["id"] for s in subtasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate subtask id")
    known = set(ids)
    for subtask in subtasks:
        if not subtask.get("action") in ACTIONS:
            raise ValueError("unsupported subtask action")
        if not set(subtask.get("depends_on", [])) <= known:
            raise ValueError("unknown dependency")
        if subtask.get("estimated_ticks", 0) <= 0:
            raise ValueError("invalid subtask duration")
    flattened = [item for group in task["parallel_groups"] for item in group]
    grouped = set(flattened)
    if grouped != known or len(flattened) != len(grouped):
        raise ValueError("parallel groups must cover every subtask exactly")
    if len(task["parallel_groups"]) < 2 or not any(len(group) >= 2 for group in task["parallel_groups"]):
        raise ValueError("task must expose at least one parallel group")
    group_index = {node: i for i, group in enumerate(task["parallel_groups"]) for node in group}
    if any(group_index[d] >= group_index[s["id"]] for s in subtasks for d in s.get("depends_on", [])):
        raise ValueError("parallel groups are not topologically ordered")
    if len(task["agent_roles"]) != task.get("required_agents"):
        raise ValueError("agent role count does not match required_agents")
    specs = {s.get("subtask_id") for s in task["action_specs"]}
    if specs != known or len(task["action_specs"]) != len(known):
        raise ValueError("action specs do not cover subtasks")
    outputs = {key for s in subtasks for key in s.get("outputs", {})}
    for check in task["success_checks"]:
        name = re.split(r"[<>=!]", check, maxsplit=1)[0]
        if name not in outputs:
            raise ValueError("success check has no declared output: " + name)
    if not all(isinstance(event, dict) and "type" in event and "trigger" in event and "effects" in event
               for event in task["environment_events"]):
        raise ValueError("events must be parameterized dictionaries")
    return True


def load_tasks(path, split=None):
    path = Path(path)
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                task = json.loads(line)
                validate_task(task)
                if split is None or task["split"] == split:
                    rows.append(task)
    return rows
