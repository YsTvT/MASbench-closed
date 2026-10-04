"""Real-environment RL contracts for Bench2 (workflow) and Bench3 (incident).

The deterministic :mod:`workflow` and :mod:`incident` state machines are useful
for unit tests and baselines, but they must never be used as a substitute for a
real train/test rollout.  This module defines the small adapter boundary used
by a real workflow backend (for example a sandboxed company service or an
incident simulator running in its own container).  An adapter is required to
declare ``real_rollout=True``; otherwise both the writer and the learner reject
the episode.

The terminal checker is authoritative.  Intermediate checker values may be
used only as training shaping and are never converted into a success label.
The public score follows the benchmark rule: failures have SR/reward zero and
no completion time; time is considered only for successful episodes.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence


TRAJECTORY_SCHEMA = "masbench.longhorizon.trajectory.v1"
SCORE_SCHEMA = "masbench.longhorizon.rl_score.v1"


class RealLongHorizonAdapter(Protocol):
    """Adapter implemented by a real Bench2/Bench3 environment.

    ``reset`` must reset an isolated environment and return either an initial
    observation mapping or ``None``.  If it returns ``None``, ``agents`` must
    be populated after reset.  ``score`` is the final checker result and must
    include ``evaluation_status`` and, for successful runs,
    ``completion_ticks``.
    """

    backend_name: str
    family: str
    real_rollout: bool

    def reset(self, case: Mapping[str, Any]) -> Any:
        ...

    def observe(self, agent: str) -> Mapping[str, Any]:
        ...

    def step_batch(self, actions: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        ...

    def score(self) -> Mapping[str, Any]:
        ...


def load_factory(spec: str):
    """Load a ``module:object`` factory used by the CLI runners."""
    if ":" not in spec:
        raise ValueError(f"factory must use module:object syntax: {spec!r}")
    module, name = spec.split(":", 1)
    value: Any = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _normalise_score(raw: Mapping[str, Any], *, case: Mapping[str, Any],
                     backend: str, family: str) -> Dict[str, Any]:
    status = raw.get("evaluation_status", "evaluated")
    if status not in {"evaluated", "unassessed"}:
        raise ValueError(f"invalid evaluation_status={status!r}")
    if status == "evaluated":
        if not isinstance(raw.get("success"), bool):
            raise ValueError("evaluated score requires boolean success")
        success = bool(raw["success"])
        completion = raw.get("completion_ticks")
        if success and (not isinstance(completion, (int, float)) or completion < 0):
            raise ValueError("successful score requires non-negative completion_ticks")
        if not success and completion is not None:
            raise ValueError("failed score must omit completion_ticks")
        reward: Optional[int] = 1 if success else 0
    else:
        success, reward, completion = None, None, None
    deadline = case.get("deadline_ticks", case.get("deadline"))
    if not isinstance(deadline, (int, float)) or deadline <= 0:
        raise ValueError("case requires positive deadline/deadline_ticks")
    topology_id = case.get("topology_id") or f"{family}:{case.get('schema_version', 'v1')}"
    return {
        "schema_version": SCORE_SCHEMA,
        "evaluation_status": status,
        "real_rollout": True,
        "backend": backend,
        "family": family,
        "task_id": case["task_id"],
        "split": case["split"],
        "topology_id": topology_id,
        "deadline": deadline,
        "success": success,
        "reward": reward,
        "completion_ticks": completion,
        "checks": deepcopy(raw.get("checks", {})),
        "raw_score": deepcopy(dict(raw)),
    }


def _progress(score: Mapping[str, Any]) -> float:
    checks = score.get("checks", {})
    if not isinstance(checks, Mapping) or not checks:
        return 0.0
    values = [v for v in checks.values() if isinstance(v, bool)]
    return sum(values) / len(values) if values else 0.0


@dataclass
class TrajectoryStep:
    episode_id: str
    tick: int
    agent: str
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    result: Mapping[str, Any]
    reward: float
    done: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealLongHorizonRLEnv:
    """Recorder and reward boundary around a real Bench2/Bench3 adapter."""

    def __init__(self, adapter: RealLongHorizonAdapter, *, max_ticks: Optional[int] = None):
        if not bool(getattr(adapter, "real_rollout", False)):
            raise ValueError("Bench2/Bench3 RL requires adapter.real_rollout=True")
        family = str(getattr(adapter, "family", ""))
        if family not in {"workflow", "incident"}:
            raise ValueError("adapter.family must be workflow or incident")
        self.adapter = adapter
        self.backend = str(getattr(adapter, "backend_name", adapter.__class__.__name__))
        self.family = family
        self.max_ticks = max_ticks
        self.case: Optional[Mapping[str, Any]] = None
        self.agents: list[str] = []
        self.tick = 0
        self.episode_id = ""
        self.steps: list[Dict[str, Any]] = []
        self._last_progress = 0.0
        self._checker_done = False

    def reset(self, case: Mapping[str, Any], episode_id: str) -> Dict[str, Mapping[str, Any]]:
        if case.get("split") not in {"train", "dev", "test"}:
            raise ValueError("case split must be train, dev, or test")
        self.case, self.episode_id, self.tick = case, str(episode_id), 0
        self.steps, self._last_progress = [], 0.0
        self._checker_done = False
        returned = self.adapter.reset(case)
        if isinstance(returned, Mapping) and returned:
            self.agents = [str(agent) for agent in returned]
            observations = {str(a): deepcopy(dict(o)) for a, o in returned.items()}
        else:
            supplied = getattr(self.adapter, "agents", None)
            if supplied is None:
                supplied = [f"worker_{i}" for i in range(int(case.get("required_agents", 1)))]
            self.agents = [str(agent) for agent in supplied]
            observations = {agent: deepcopy(dict(self.adapter.observe(agent))) for agent in self.agents}
        if not self.agents:
            raise ValueError("real adapter must expose at least one agent")
        return observations

    @property
    def done(self) -> bool:
        return bool((self.max_ticks is not None and self.tick >= self.max_ticks)
                    or getattr(self.adapter, "done", False) or self._checker_done)

    def step(self, actions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
        if self.case is None:
            raise RuntimeError("reset must be called before step")
        if self.done:
            raise RuntimeError("episode reached terminal condition")
        if not isinstance(actions, Mapping):
            raise ValueError("actions must be a mapping")
        results = self.adapter.step_batch(actions) or {}
        if not isinstance(results, Mapping):
            raise ValueError("adapter.step_batch must return a mapping")
        try:
            live_score = self.adapter.score()
            progress = _progress(live_score)
            self._checker_done = live_score.get("evaluation_status") == "evaluated"
        except Exception:
            progress = self._last_progress
        shaping = max(-1.0, min(1.0, progress - self._last_progress))
        self._last_progress = progress
        observations = {agent: deepcopy(dict(self.adapter.observe(agent))) for agent in self.agents}
        for agent in self.agents:
            action = deepcopy(dict(actions.get(agent, {"type": "observe"})))
            result = deepcopy(dict(results.get(agent, {"ok": False, "error": "missing_result"})))
            valid = 1.0 if result.get("ok") is True else -1.0
            self.steps.append(TrajectoryStep(self.episode_id, self.tick, agent,
                                              observations[agent], action, result,
                                              0.05 * shaping + 0.01 * valid,
                                              False).to_dict())
        self.tick += 1
        return observations

    def finish(self) -> Dict[str, Any]:
        if self.case is None:
            raise RuntimeError("reset must be called before finish")
        score = _normalise_score(self.adapter.score(), case=self.case,
                                 backend=self.backend, family=self.family)
        terminal = score["reward"] if score["evaluation_status"] == "evaluated" else 0
        return {
            "schema_version": TRAJECTORY_SCHEMA,
            "episode_id": self.episode_id,
            "task_id": self.case["task_id"],
            "split": self.case["split"],
            "family": self.family,
            "topology_id": score["topology_id"],
            "backend": self.backend,
            "real_rollout": True,
            "steps": self.steps,
            "tick": self.tick,
            "terminal_reward": terminal,
            "shaped_return": sum(step["reward"] for step in self.steps),
            "score": score,
        }


class TrajectoryWriter:
    """Persist only real trajectories and terminal checker scores."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, episode: Mapping[str, Any]) -> Path:
        if episode.get("real_rollout") is not True:
            raise ValueError("refusing to persist non-real Bench2/Bench3 trajectory")
        target = self.root / "episodes" / f"{episode['episode_id']}.trajectory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(episode, indent=2, sort_keys=True) + "\n")
        tmp.replace(target)
        self.write_score(episode["episode_id"], episode["score"])
        return target

    def write_score(self, episode_id: str, score: Mapping[str, Any]) -> Path:
        if score.get("real_rollout") is not True:
            raise ValueError("refusing to persist non-real Bench2/Bench3 score")
        target = self.root / "scores" / str(episode_id) / "score.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
        tmp.replace(target)
        return target


def group_advantages(episodes: Sequence[Mapping[str, Any]], *, key: str = "task_id") -> list[float]:
    groups: Dict[str, list[int]] = defaultdict(list)
    returns = [float(e.get("terminal_reward", 0.0)) for e in episodes]
    for i, episode in enumerate(episodes):
        groups[str(episode.get(key, episode.get("episode_id", i)))].append(i)
    advantages = [0.0] * len(episodes)
    for indices in groups.values():
        values = [returns[i] for i in indices]
        avg = sum(values) / len(values)
        scale = math.sqrt(sum((v - avg) ** 2 for v in values) / len(values)) or 1.0
        for i in indices:
            advantages[i] = (returns[i] - avg) / scale
    return advantages


class OnPolicyGRPOLearner:
    """Policy-independent train-batch coordinator for real trajectories."""

    def __init__(self, policy, *, algorithm: str = "grpo"):
        if algorithm not in {"grpo", "reinforce"}:
            raise ValueError("algorithm must be grpo or reinforce")
        self.policy, self.algorithm = policy, algorithm

    def train_batch(self, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not episodes:
            raise ValueError("empty RL batch")
        for episode in episodes:
            if episode.get("real_rollout") is not True:
                raise ValueError("RL update requires real rollout trajectories")
            if episode.get("split") != "train":
                raise ValueError("RL update accepts train split only")
            if episode.get("score", {}).get("evaluation_status") != "evaluated":
                raise ValueError("RL update requires evaluated checker score")
        # REINFORCE uses the terminal return directly (a one-candidate batch
        # must still produce a learning signal).  GRPO centers candidates
        # within the same task to remove the task-level baseline.
        advantages = ([float(e.get("terminal_reward", 0.0)) for e in episodes]
                      if self.algorithm == "reinforce" else group_advantages(episodes))
        update = self.policy.update(episodes, advantages)
        return {"algorithm": self.algorithm, "episodes": len(episodes),
                "successes": sum(e["score"].get("success") is True for e in episodes),
                "mean_terminal_reward": sum(float(e.get("terminal_reward", 0)) for e in episodes) / len(episodes),
                "advantages": advantages, "policy_update": dict(update or {})}


def load_score_rows(root: Path, *, split: Optional[str] = None) -> list[Dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(Path(root).rglob("score.json"))
    rows = []
    for path in paths:
        row = json.loads(path.read_text())
        if row.get("real_rollout") is not True:
            raise ValueError(f"non-real score is not valid for Bench2/3 eval: {path}")
        if row.get("evaluation_status") != "evaluated":
            raise ValueError(f"unassessed score cannot enter Bench2/3 eval: {path}")
        if split and row.get("split") != split:
            continue
        rows.append(row)
    return rows
