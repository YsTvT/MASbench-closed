"""Bench2: long-horizon enterprise workflow with parallel dependent jobs."""

from copy import deepcopy
from .core import Simulation, digest


CATALOG = {
    "report": {"inputs": ["data"], "outputs": "report", "duration": 3},
    "dashboard": {"inputs": ["data"], "outputs": "dashboard", "duration": 4},
    "approval": {"inputs": ["report", "dashboard"], "outputs": "approval", "duration": 2},
    "release": {"inputs": ["approval"], "outputs": "release", "duration": 3},
}


def make_workflow(seed=0, difficulty=1):
    if difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2 or 3")
    branches = ["report", "dashboard"]
    if difficulty >= 2:
        branches += ["quality"]
    if difficulty >= 3:
        branches += ["security"]
    tasks = {}
    for i, kind in enumerate(branches):
        if kind in ("quality", "security"):
            tasks[kind] = {"kind": "report", "requires": ["data"], "artifact": kind}
        else:
            tasks[kind] = {"kind": kind, "requires": ["data"], "artifact": kind}
    tasks["approval"] = {"kind": "approval", "requires": branches, "artifact": "approval"}
    tasks["release"] = {"kind": "release", "requires": ["approval"], "artifact": "release"}
    return {"task_id": "workflow_%03d_d%d" % (seed, difficulty), "seed": seed,
            # Calibrated against the executable serial controller: d1/d2 remain
            # feasible, while the deepest graph exposes the parallelism gap.
            "deadline": 14 + 2 * (difficulty - 1), "difficulty": difficulty,
            "tasks": tasks, "required": ["release"], "input_digest": digest({"seed": seed, "difficulty": difficulty})}


class WorkflowEnv(Simulation):
    def _reset_state(self):
        self.artifacts = {"data": {"version": self.case["input_digest"]}}
        self.versions = {"data": 1}
        self.approvals = set()
        self.submitted = False

    def public_state(self):
        return {"artifacts": sorted(self.artifacts), "finished": sorted(self.approvals),
                "tasks": {k: v["artifact"] if k in self.artifacts else "pending" for k, v in self.case["tasks"].items()}}

    def _prepare(self, agent, action):
        typ, task = action.get("type"), action.get("task")
        if typ == "submit":
            return action, 1, None
        if typ == "inspect":
            return action, 1, None
        if typ == "handoff":
            return action, 1, None
        if task not in self.case["tasks"]:
            raise ValueError("unknown_task")
        spec = self.case["tasks"][task]
        if typ not in ("execute", "approve"):
            raise ValueError("unsupported_action")
        if typ == "execute" and spec["kind"] == "approval":
            raise ValueError("approval_requires_approve")
        if typ == "approve" and spec["kind"] != "approval":
            raise ValueError("only_approval_can_be_approved")
        for req in spec["requires"]:
            if req not in self.artifacts:
                raise ValueError("missing_precondition:%s" % req)
        duration = CATALOG.get(spec["kind"], CATALOG["report"])["duration"]
        lock = "artifact:%s" % spec["artifact"]
        action["_versions"] = {x: self.versions.get(x, 0) for x in spec["requires"]}
        return action, duration, lock

    def _finish(self, agent, action):
        typ = action["type"]
        if typ in ("inspect", "handoff"):
            return {"ok": True}
        if typ == "submit":
            checks = self.check()
            self.submitted = True
            return {"ok": all(checks.values()), "checks": checks}
        task, spec = action["task"], self.case["tasks"][action["task"]]
        for key, version in action.get("_versions", {}).items():
            if self.versions.get(key, 0) != version:
                return {"ok": False, "error": "stale_precondition:%s" % key}
        if typ == "approve":
            self.approvals.add(task)
        self.artifacts[spec["artifact"]] = {"task": task, "producer": agent, "digest": digest(action)}
        self.versions[spec["artifact"]] = self.versions.get(spec["artifact"], 0) + 1
        return {"ok": True, "artifact": spec["artifact"]}

    def check(self):
        checks = {"release_artifact": "release" in self.artifacts,
                  "approval": "approval" in self.artifacts}
        for task, spec in self.case["tasks"].items():
            if task != "release" and spec["artifact"] != "approval":
                checks["branch:" + task] = spec["artifact"] in self.artifacts
        return checks


def serial_baseline(case):
    """TheAgentCompany/OpenHands-style single controller: correct but serial."""
    env = WorkflowEnv(case, workers=1)
    order = [k for k in case["tasks"] if k not in ("approval", "release")] + ["approval", "release"]
    for task in order:
        env.step_batch({"worker_0": {"type": "approve" if task == "approval" else "execute", "task": task}})
        env.advance(3)
    env.step_batch({"worker_0": {"type": "submit"}})
    env.advance(1)
    return env.score()
