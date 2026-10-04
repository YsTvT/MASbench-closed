"""Backend-neutral Minecraft adapter and CPU-only symbolic executor.

The symbolic executor has the same reset/observe/step/score contract expected
from a VillagerAgent adapter. It validates dependency, role, handoff, event and
deadline behavior without pretending to be a real Minecraft rollout.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from statistics import mean

from .tasks import ACTIONS, validate_task


@dataclass
class _Job:
    agent: str
    subtask_id: str
    finishes: int


class MinecraftAdapter:
    """Interface implemented by the real VillagerAgent/Mineflayer backend."""

    def reset(self, task):
        raise NotImplementedError

    def observe(self, agent):
        raise NotImplementedError

    def step_batch(self, actions):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


class SymbolicMinecraftEnv(MinecraftAdapter):
    def __init__(self, task, workers=None):
        validate_task(task)
        self.task = deepcopy(task)
        self.agents = workers or [r["agent_id"] for r in task["agent_roles"]]
        self.reset(task)

    def reset(self, task=None):
        if task is not None:
            validate_task(task)
            self.task = deepcopy(task)
        self.tick = 0
        self.completed = set()
        self.inventory = deepcopy(self.task["initial_inventory"])
        self.jobs = {}
        self.inbox = {a: [] for a in self.agents}
        self.pending_messages = []
        self.trace = []
        self.actions_used = 0
        self.failure_reason = None
        self.submitted = False
        self.event_state = {}
        return self

    @property
    def done(self):
        return self.submitted or self.failure_reason is not None or self.tick >= self.task["deadline_ticks"]

    def _log(self, agent, action, result):
        self.trace.append({"tick": self.tick, "agent": agent, "action": deepcopy(action),
                           "result": deepcopy(result)})

    def _finish_jobs(self):
        for agent, job in list(self.jobs.items()):
            if job.finishes > self.tick:
                continue
            subtask = next(s for s in self.task["subtasks"] if s["id"] == job.subtask_id)
            self.completed.add(job.subtask_id)
            for key, value in subtask["outputs"].items():
                self.inventory[key] = self.inventory.get(key, 0) + value if isinstance(value, int) else value
            self._log(agent, {"type": subtask["action"], "subtask_id": job.subtask_id},
                      {"ok": True, "completed": job.subtask_id, "outputs": subtask["outputs"]})
            self.inbox[agent].append({"tick": self.tick, "completed": job.subtask_id,
                                      "outputs": deepcopy(subtask["outputs"])})
            del self.jobs[agent]

    def _apply_events(self):
        for event in self.task["environment_events"]:
            trigger = event["trigger"]
            event_id = event["event_id"]
            if event_id not in self.event_state and self.tick >= trigger["tick"]:
                self.event_state[event_id] = deepcopy(event["effects"])

    def step_batch(self, actions):
        if self.done:
            return {"error": "episode_finished"}
        if not isinstance(actions, dict):
            raise ValueError("actions must be a mapping")
        results = {}
        jobs_by_subtask = {j.subtask_id for j in self.jobs.values()}
        specs = {s["id"]: s for s in self.task["subtasks"]}
        for agent, action in sorted(actions.items()):
            self.actions_used += 1
            if agent not in self.agents:
                results[agent] = {"ok": False, "error": "unknown_agent"}
                continue
            if agent in self.jobs:
                results[agent] = {"ok": False, "error": "agent_busy"}
                continue
            if not isinstance(action, dict) or not isinstance(action.get("type"), str):
                results[agent] = {"ok": False, "error": "malformed_action"}
                continue
            typ = action["type"]
            if typ == "message":
                recipient, body = action.get("recipient"), action.get("body", "")
                ok = recipient in self.agents and isinstance(body, str) and len(body) <= 2000
                result = {"ok": ok, "error": None if ok else "invalid_message"}
                if ok:
                    self.pending_messages.append((self.tick + 1, recipient, {"sender": agent, "body": body}))
                results[agent] = result
                self._log(agent, action, result)
                continue
            if typ == "handoff" and action.get("subtask_id") is None:
                recipient = action.get("recipient")
                ok = recipient in self.agents and recipient != agent
                result = {"ok": ok, "error": None if ok else "invalid_recipient"}
                results[agent] = result
                self._log(agent, action, result)
                continue
            if typ == "submit":
                checks = self.check()
                self.submitted = True
                result = {"ok": all(checks.values()), "checks": checks}
                results[agent] = result
                self._log(agent, action, result)
                continue
            subtask_id = action.get("subtask_id")
            if typ not in ACTIONS or subtask_id not in specs:
                results[agent] = {"ok": False, "error": "unsupported_action_or_subtask"}
                self._log(agent, action, results[agent])
                continue
            if subtask_id in self.completed or subtask_id in jobs_by_subtask:
                results[agent] = {"ok": False, "error": "duplicate_subtask"}
                self._log(agent, action, results[agent])
                continue
            spec = specs[subtask_id]
            missing = [dep for dep in spec["depends_on"] if dep not in self.completed]
            if missing:
                results[agent] = {"ok": False, "error": "missing_dependency", "dependencies": missing}
                self._log(agent, action, results[agent])
                continue
            expected = spec["action"]
            if typ != expected:
                results[agent] = {"ok": False, "error": "action_mismatch", "expected": expected}
                self._log(agent, action, results[agent])
                continue
            duration = spec["estimated_ticks"] + 2
            self.jobs[agent] = _Job(agent, subtask_id, self.tick + duration)
            results[agent] = {"ok": True, "finishes_at": self.tick + duration}
            self._log(agent, action, results[agent])
        self.tick += 1
        self._apply_events()
        for delivery, recipient, message in list(self.pending_messages):
            if delivery <= self.tick:
                self.inbox[recipient].append(message)
                self.pending_messages.remove((delivery, recipient, message))
        self._finish_jobs()
        return results

    def advance(self, ticks=1):
        for _ in range(ticks):
            if self.done:
                break
            self.step_batch({})

    def observe(self, agent):
        if agent not in self.agents:
            raise ValueError("unknown_agent")
        messages = self.inbox[agent]
        self.inbox[agent] = []
        return {"tick": self.tick, "deadline": self.task["deadline_ticks"], "agent": agent,
                "busy_until": self.jobs[agent].finishes if agent in self.jobs else None,
                "completed_subtasks": sorted(self.completed), "available_subtasks": sorted(
                    s["id"] for s in self.task["subtasks"] if s["id"] not in self.completed),
                "inventory": deepcopy(self.inventory), "events": deepcopy(self.event_state),
                "messages": deepcopy(messages), "done": self.done}

    def check(self):
        def outputs_satisfied(outputs):
            for key, required in outputs.items():
                actual = self.inventory.get(key, 0)
                if isinstance(required, (int, float)) and isinstance(actual, (int, float)):
                    if actual < required:
                        return False
                elif actual != required:
                    return False
            return True
        return {"all_subtasks": len(self.completed) == len(self.task["subtasks"]),
                "required_outputs": all(
                    outputs_satisfied(s["outputs"])
                    for s in self.task["subtasks"]),
                "no_failure": self.failure_reason is None}

    def score(self):
        checks = self.check()
        success = bool(self.submitted and all(checks.values()) and self.tick <= self.task["deadline_ticks"])
        return {"success": success, "reward": int(success),
                "completion_ticks": self.tick if success else None,
                "deadline": self.task["deadline_ticks"], "checks": checks,
                "diagnostic_progress": sum(checks.values()) / len(checks),
                "reason": self.failure_reason or ("success" if success else "timeout" if self.done else "running"),
                "trace": deepcopy(self.trace), "actions_used": self.actions_used}


def run_reference(task, controller="parallel", workers=None):
    """Execute a task's dependency graph with a deterministic controller."""
    env = SymbolicMinecraftEnv(task, workers=workers)
    agents = env.agents
    for group in task["parallel_groups"]:
        if controller == "serial":
            for subtask_id in group:
                subtask = next(s for s in task["subtasks"] if s["id"] == subtask_id)
                env.step_batch({agents[0]: {"type": subtask["action"], "subtask_id": subtask_id}})
                while env.jobs and not env.done:
                    env.step_batch({})
            continue
        for start in range(0, len(group), len(agents)):
            actions = {}
            for i, subtask_id in enumerate(group[start:start + len(agents)]):
                agent = agents[i]
                subtask = next(s for s in task["subtasks"] if s["id"] == subtask_id)
                actions[agent] = {"type": subtask["action"], "subtask_id": subtask_id}
            env.step_batch(actions)
            while env.jobs and not env.done:
                env.step_batch({})
    env.step_batch({agents[0]: {"type": "submit"}})
    return env.score()
