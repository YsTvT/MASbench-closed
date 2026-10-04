"""VillagerAgent integration contract.

The module intentionally imports no VillagerAgent implementation.  Existing
Mineflayer/VillagerAgent clients can be injected through the small client
protocol below, making the benchmark backend testable without credentials or a
running Minecraft server.
"""

from copy import deepcopy

from .env import MinecraftAdapter
from .tasks import validate_task


def check_world_state(task, state):
    """Backend-neutral checker for a VillagerAgent world snapshot."""
    completed = set(state.get("completed_subtasks", []))
    inventory = state.get("inventory", {})
    predicates = state.get("predicates", {})
    def outputs_satisfied(outputs):
        for key, required in outputs.items():
            actual = inventory.get(key, 0)
            if isinstance(required, (int, float)) and isinstance(actual, (int, float)):
                if actual < required:
                    return False
            elif actual != required:
                return False
        return True
    checks = {
        "all_subtasks": completed >= {s["id"] for s in task["subtasks"]},
        "required_outputs": all(outputs_satisfied(s["outputs"])
                                 for s in task["subtasks"]),
        "concrete_predicates": all(predicates.get(c["predicate"], False)
                                    for c in task.get("concrete_success_checks", [])),
    }
    return checks


class VillagerAgentAdapter(MinecraftAdapter):
    """Adapter around a real VillagerAgent/Mineflayer client.

    Required client methods are `reset_world`, `observe`, `execute_action`,
    `world_snapshot`, and optionally `advance`.  This keeps Minecraft process
    management outside MASbench while making the evaluator deterministic.
    """

    def __init__(self, client, agents):
        self.client = client
        self.agents = list(agents)
        self.task = None
        self.tick = 0
        self.trace = []
        self.submitted = False

    def reset(self, task):
        validate_task(task)
        self.task = deepcopy(task)
        self.tick = 0
        self.trace = []
        self.submitted = False
        self.client.reset_world(task["world_config"], task["world_seed"], task["initial_inventory"], task["agent_roles"])
        return self

    def observe(self, agent):
        result = self.client.observe(agent)
        return {"tick": self.tick, "deadline": self.task["deadline_ticks"],
                "agent": agent, "world": result}

    def step_batch(self, actions):
        results = {}
        for agent, action in sorted(actions.items()):
            if action.get("type") == "submit":
                self.submitted = True
                results[agent] = {"ok": True}
            else:
                results[agent] = self.client.execute_action(agent, action)
            self.trace.append({"tick": self.tick, "agent": agent,
                               "action": deepcopy(action), "result": deepcopy(results[agent])})
        self.tick += 1
        if hasattr(self.client, "advance"):
            self.client.advance(1)
        return results

    def score(self):
        state = self.client.world_snapshot()
        checks = check_world_state(self.task, state)
        success = bool(self.submitted and all(checks.values()) and self.tick <= self.task["deadline_ticks"])
        return {"evaluation_status": "evaluated", "success": success, "reward": int(success),
                "completion_ticks": self.tick if success else None,
                "deadline": self.task["deadline_ticks"], "checks": checks,
                "trace": deepcopy(self.trace), "backend": "villageragent"}
