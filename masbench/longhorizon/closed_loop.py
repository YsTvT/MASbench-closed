"""Closed-loop scripted controllers and evaluator for Bench2/Bench3."""

import copy
import json
from pathlib import Path

from ..controllers import Controller
from .core import summarize, paired_speedup
from .workflow import WorkflowEnv, make_workflow
from .incident import IncidentEnv, make_incident, FAULTS


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError("unsupported trace value: %r" % (type(value),))


class _WorkflowPlanner:
    def reset(self, case, env):
        self.case, self.env = case, env
        self.branches = [k for k in case["tasks"] if k not in ("approval", "release")]
        self.assigned = set()
        self.approval_sent = self.release_sent = self.submitted = False

    def act(self, observations):
        state = next(iter(observations.values()))["state"]
        free = [a for a, o in observations.items() if o["busy_until"] is None]
        actions = {}
        for agent in free:
            pending = [x for x in self.branches if x not in self.assigned and x not in state["artifacts"]]
            if pending:
                task = pending[0]
                self.assigned.add(task)
                actions[agent] = {"type": "execute", "task": task}
                continue
            if not self.approval_sent and "approval" not in state["artifacts"] and all(x in state["artifacts"] for x in self.branches):
                self.approval_sent = True
                actions[agent] = {"type": "approve", "task": "approval"}
                continue
            if not self.release_sent and "approval" in state["artifacts"]:
                self.release_sent = True
                actions[agent] = {"type": "execute", "task": "release"}
                continue
            if not self.submitted and "release" in state["artifacts"]:
                self.submitted = True
                actions[agent] = {"type": "submit"}
        return actions

    def finish(self):
        return None


class _IncidentPlanner:
    def reset(self, case, env):
        self.case, self.env = case, env
        self.signals = list(case.get("query_signals", case["signals"]))
        self.assigned_queries = set()
        self.diagnose_sent = self.mitigate_sent = self.verify_sent = self.communicate_sent = self.submit_sent = False

    def act(self, observations):
        state = next(iter(observations.values()))["state"]
        free = [a for a, o in observations.items() if o["busy_until"] is None]
        actions = {}
        for agent in free:
            missing = [s for s in self.signals if s not in self.assigned_queries and s not in state["evidence"]]
            if missing:
                signal = missing[0]
                self.assigned_queries.add(signal)
                actions[agent] = {"type": "query", "signal": signal}
            elif not state["root_cause"] and not self.diagnose_sent and set(self.signals).issubset(state["evidence"]):
                self.diagnose_sent = True
                root = {"cache_miss": "cache_config", "db_pool": "db_pool", "queue_lag": "worker_scale"}[self.case["fault"]]
                actions[agent] = {"type": "diagnose", "root_cause": root}
            elif state["root_cause"] and not state["mitigated"] and not self.mitigate_sent:
                self.mitigate_sent = True
                actions[agent] = {"type": "mitigate", "fix": FAULTS[self.case["fault"]]["fix"]}
            elif state["mitigated"] and not state["verified"] and not self.verify_sent:
                self.verify_sent = True
                actions[agent] = {"type": "verify", "checks": set(self.signals)}
            elif state["verified"] and not state["communicated"] and not self.communicate_sent:
                self.communicate_sent = True
                actions[agent] = {"type": "communicate", "body": "incident resolved"}
            elif state["communicated"] and not self.submit_sent:
                self.submit_sent = True
                actions[agent] = {"type": "submit"}
        return actions

    def finish(self):
        return None


class ClosedLoopRunner:
    def __init__(self, env, controller, max_ticks=None):
        self.env, self.controller = env, controller
        self.max_ticks = max_ticks or env.case["deadline"]

    def run(self):
        self.controller.reset(self.env.case, self.env)
        while not self.env.done and self.env.tick < self.max_ticks:
            observations = {agent: self.env.observe(agent) for agent in self.env.agents}
            actions = self.controller.act(observations)
            self.env.step_batch(actions)
        result = self.env.score()
        result.update({"trace": copy.deepcopy(self.env.trace), "actions_used": self.env.actions_used,
                       "message_chars": self.env.message_chars, "ticks": self.env.tick})
        self.controller.finish()
        return result


def run_closed_loop(case, family, method, workers=4, deadline_scale=1.0):
    scaled = copy.deepcopy(case)
    scaled["deadline"] = max(1, int(round(case["deadline"] * deadline_scale)))
    n_workers = workers if method == "mas" else 1
    planner = _WorkflowPlanner() if family == "workflow" else _IncidentPlanner()
    env = WorkflowEnv(scaled, workers=n_workers) if family == "workflow" else IncidentEnv(scaled, workers=n_workers)
    result = ClosedLoopRunner(env, planner).run()
    result.update({"task_id": case["task_id"], "family": family, "method": method,
                   "difficulty": case["difficulty"], "deadline": scaled["deadline"]})
    return result


def run_closed_loop_suite(cases, out_dir, deadline_scale=1.0, workers=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        family = case["family"]
        for method in ("serial", "mas"):
            rows.append(run_closed_loop(case, family, method, workers, deadline_scale))
    (out_dir / "runs.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True, default=_json_default) for r in rows) + "\n")
    grouped = {}
    for family in ("workflow", "incident"):
        serial = [r for r in rows if r["family"] == family and r["method"] == "serial"]
        mas = [r for r in rows if r["family"] == family and r["method"] == "mas"]
        grouped[family + "/serial"] = summarize(serial)
        grouped[family + "/mas"] = summarize(mas)
        grouped[family + "/paired"] = paired_speedup(serial, mas)
    report = {"deadline_scale": deadline_scale, "workers": workers, "runs": len(rows),
              "groups": grouped, "protocol": "closed-loop; success-first; time conditioned on success"}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Closed-loop deterministic run", "", "| Group | SR | Composite | Successful mean ticks |", "|---|---:|---:|---:|"]
    for name in ("workflow/serial", "workflow/mas", "incident/serial", "incident/mas"):
        value = grouped[name]
        lines.append("| %s | %.3f | %.3f | %s |" % (name, value["sr"], value["composite_score"], value["successful_mean_ticks"]))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def cases_from_jsonl(data_dir, split="test"):
    path = Path(data_dir) / ("longhorizon_%s.jsonl" % split)
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_smoke_cases():
    return [make_workflow(0, 1) | {"family": "workflow"}, make_workflow(1, 3) | {"family": "workflow"},
            make_incident(0, 1) | {"family": "incident"}, make_incident(1, 3) | {"family": "incident"}]
