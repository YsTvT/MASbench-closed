"""Shared virtual clock, trace, message transport and success-first scoring."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from statistics import mean, median


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Job:
    agent: str
    action: dict
    finishes: int
    lock: str


class Simulation:
    """All actions in a batch read the state at dispatch, then finish on a clock.

    Resource writes are reserved until completion. Completion events are applied
    after source events, so optimistic versions reject stale results. A controller
    and a MAS receive the same worker pool; only decision architecture differs.
    """

    def __init__(self, case, workers=4, action_budget=800, message_budget=16000):
        if not isinstance(workers, int) or not 1 <= workers <= 16:
            raise ValueError("workers must be an integer in [1,16]")
        self.case = deepcopy(case)
        self.agents = ["worker_%d" % i for i in range(workers)]
        self.action_budget = action_budget
        self.message_budget = message_budget
        self.tick = 0
        self.jobs = {}
        self.trace = []
        self.inbox = {a: [] for a in self.agents}
        self.pending_messages = []
        self.actions_used = self.message_chars = 0
        self.submitted = False
        self.terminal_reason = None
        self._reset_state()

    @property
    def done(self):
        return self.submitted or self.tick >= self.case["deadline"] or self.terminal_reason is not None

    def _reset_state(self):
        raise NotImplementedError

    def _prepare(self, agent, action):
        raise NotImplementedError

    def _finish(self, agent, action):
        raise NotImplementedError

    def _events(self):
        pass

    def _log(self, agent, action, result):
        self.trace.append({"tick": self.tick, "agent": agent, "action": deepcopy(action), "result": deepcopy(result)})

    def step_batch(self, actions):
        """Dispatch actions and advance exactly one tick; omitted workers idle."""
        if not isinstance(actions, dict):
            raise ValueError("batch must map worker IDs to action objects")
        if self.done:
            return {"error": "episode_finished"}
        results = {}
        locks = {j.lock for j in self.jobs.values() if j.lock}
        for agent, action in sorted(actions.items()):
            if agent not in self.agents:
                results[agent] = {"ok": False, "error": "unknown_worker"}
                continue
            if agent in self.jobs:
                results[agent] = {"ok": False, "error": "worker_busy"}
                continue
            if not isinstance(action, dict) or not isinstance(action.get("type"), str):
                results[agent] = {"ok": False, "error": "malformed_action"}
                continue
            self.actions_used += 1
            if self.actions_used > self.action_budget:
                self.terminal_reason = "action_budget_exhausted"
                results[agent] = {"ok": False, "error": self.terminal_reason}
                break
            if action["type"] == "message":
                body = action.get("body", "")
                recipient = action.get("recipient")
                if recipient not in self.agents or not isinstance(body, str) or len(body) > 2000:
                    result = {"ok": False, "error": "invalid_message"}
                elif self.message_chars + len(body) > self.message_budget:
                    result = {"ok": False, "error": "message_budget_exhausted"}
                else:
                    self.message_chars += len(body)
                    self.pending_messages.append((self.tick + 1, recipient, {"sender": agent, "body": body}))
                    result = {"ok": True, "delivers_at": self.tick + 1}
                results[agent] = result
                self._log(agent, action, result)
                continue
            try:
                prepared, duration, lock = self._prepare(agent, deepcopy(action))
                if lock and lock in locks:
                    raise ValueError("resource_busy")
                if duration < 1:
                    raise ValueError("invalid_duration")
            except (ValueError, KeyError, TypeError) as exc:
                result = {"ok": False, "error": str(exc)}
                results[agent] = result
                self._log(agent, action, result)
                continue
            self.jobs[agent] = Job(agent, prepared, self.tick + duration, lock)
            locks.add(lock)
            results[agent] = {"ok": True, "finishes_at": self.tick + duration}
        self.tick += 1
        self._events()
        for delivery, recipient, message in list(self.pending_messages):
            if delivery <= self.tick:
                self.inbox[recipient].append(message)
                self.pending_messages.remove((delivery, recipient, message))
        # Every action was prepared against pre-batch state. Read results have their
        # own captured versions; writes verify those versions at completion.
        for agent, job in list(self.jobs.items()):
            if job.finishes <= self.tick:
                try:
                    result = self._finish(agent, job.action)
                except (ValueError, KeyError, TypeError) as exc:
                    result = {"ok": False, "error": str(exc)}
                self._log(agent, {k: v for k, v in job.action.items() if not k.startswith("_")}, result)
                self.inbox[agent].append({"tick": self.tick, "result": deepcopy(result)})
                del self.jobs[agent]
        return results

    def advance(self, ticks=1):
        if not isinstance(ticks, int) or not 1 <= ticks <= 1000:
            raise ValueError("ticks must be in [1,1000]")
        for _ in range(ticks):
            if self.done:
                break
            self.step_batch({})

    def observe(self, agent):
        if agent not in self.agents:
            raise ValueError("unknown_worker")
        messages = self.inbox[agent]
        self.inbox[agent] = []
        return {"tick": self.tick, "deadline": self.case["deadline"], "worker": agent,
                "busy_until": self.jobs[agent].finishes if agent in self.jobs else None,
                "messages": deepcopy(messages), "state": self.public_state(), "done": self.done}

    def score(self):
        checks = self.check()
        success = self.submitted and all(checks.values()) and self.tick <= self.case["deadline"] and not self.terminal_reason
        return {"success": bool(success), "reward": int(bool(success)),
                "completion_ticks": self.tick if success else None,
                "checks": checks, "diagnostic_progress": sum(checks.values()) / max(1, len(checks)),
                "reason": self.terminal_reason or ("submitted" if self.submitted else "timeout" if self.done else "running")}


def summarize(runs):
    runs = list(runs)
    unassessed = [r for r in runs if r.get("evaluation_status", "evaluated") != "evaluated"]
    evaluated = [r for r in runs if r.get("evaluation_status", "evaluated") == "evaluated"]
    if any(not isinstance(r.get("success"), bool) for r in evaluated):
        raise ValueError("evaluated runs require boolean success")
    successes = [r for r in evaluated if r["success"]]
    times = [r["completion_ticks"] for r in successes]
    if any(t is None or t < 0 for t in times):
        raise ValueError("successful run requires completion time")
    n = len(evaluated)
    # Wilson interval: undefined for an empty dataset, not a fictitious zero.
    if n:
        p, z = len(successes) / n, 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [max(0, center - radius), min(1, center + radius)]
    else:
        p, interval = None, None
    normalized = []
    for row in successes:
        deadline = row.get("deadline")
        if deadline is None or deadline <= 0:
            raise ValueError("successful run requires a positive deadline")
        normalized.append(min(1.0, max(0.0, row["completion_ticks"] / deadline)))
    # Failures contribute zero by construction. The conditional time statistic
    # remains interpretable because it is calculated only over successful runs.
    conditional_time = mean(times) if times else None
    conditional_normalized_time = mean(normalized) if normalized else None
    time_score = (1.0 - conditional_normalized_time) if conditional_normalized_time is not None else 0.0
    composite = (p * time_score) if p is not None else None
    return {"runs": len(runs), "evaluated_runs": n, "unassessed_runs": len(unassessed),
            "successes": len(successes), "success_rate": p,
            "sr": p, "success_rate_95ci": interval,
            "successful_mean_ticks": conditional_time,
            "successful_median_ticks": median(times) if times else None,
            "successful_mean_normalized_time": conditional_normalized_time,
            "successful_time_score": time_score,
            "composite_score": composite,
            "metric_definition": "SR first; time conditioned on success; composite=SR*(1-mean(success_time/deadline))"}


def paired_speedup(first, second):
    a = {r["task_id"]: r for r in first}
    b = {r["task_id"]: r for r in second}
    if len(a) != len(first) or len(b) != len(second):
        raise ValueError("duplicate task IDs; pair individual replicate seeds separately")
    pairs = [(a[k], b[k]) for k in a.keys() & b.keys() if a[k]["success"] and b[k]["success"]]
    return {"joint_success_tasks": len(pairs), "mean_speedup": mean(x["completion_ticks"] / y["completion_ticks"] for x, y in pairs) if pairs else None}
