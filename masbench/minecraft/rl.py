"""Real-Minecraft RL episode and learner contracts.

This module is deliberately backend agnostic.  A VillagerAgent/Mineflayer
adapter is injected by the caller; the benchmark code never fabricates a
world transition.  SymbolicMinecraftEnv can be used by tests and a dry run,
but every emitted record is marked ``real_rollout=False`` and is rejected by
the real-evaluation path.

The learner implements the *policy side* of an on-policy REINFORCE/GRPO
update.  A policy implementation (for example a Qwen LoRA policy) supplies
``sample_action`` and ``update``.  Keeping this interface small lets the
remote runner use the same trajectory/checker protocol without requiring
Transformers or a GPU in this repository's CPU test environment.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence


TRAJECTORY_SCHEMA = "masbench.minecraft.trajectory.v1"
SCORE_SCHEMA = "masbench.minecraft.rl_score.v1"


class RLPolicy(Protocol):
    """Policy hook used by the real runner.

    ``sample_action`` receives one agent's observation and must return a
    typed VillagerAgent action.  ``update`` is called once per train batch
    with trajectory dictionaries and GRPO/REINFORCE advantages.  A policy
    can wrap a local Qwen LoRA model; no API-only model is silently treated as
    trainable.
    """

    def sample_action(self, observation: Mapping[str, Any], agent: str, *, deterministic: bool = False) -> Mapping[str, Any]:
        ...

    def update(self, trajectories: Sequence[Mapping[str, Any]], advantages: Sequence[float]) -> Mapping[str, Any]:
        ...

    def save_checkpoint(self, path: Path, metadata: Mapping[str, Any]) -> None:
        ...


class RealMinecraftAdapter(Protocol):
    """Minimal adapter implemented by the real VillagerAgent bridge."""

    backend_name: str
    real_rollout: bool

    def reset(self, task: Mapping[str, Any]) -> Any:
        ...

    def observe(self, agent: str) -> Mapping[str, Any]:
        ...

    def step_batch(self, actions: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        ...

    def score(self) -> Mapping[str, Any]:
        ...


def _import_object(spec: str) -> Any:
    """Import ``package.module:object`` for the CLI factories."""
    if ":" not in spec:
        raise ValueError(f"factory must use module:object syntax: {spec!r}")
    module_name, object_name = spec.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for part in object_name.split("."):
        value = getattr(value, part)
    return value


def load_factory(spec: str):
    """Public factory loader used by train/eval CLIs."""
    return _import_object(spec)


def _normalise_score(raw: Mapping[str, Any], *, task: Mapping[str, Any], real: bool, backend: str) -> Dict[str, Any]:
    """Validate and normalize the one authoritative terminal score."""
    status = raw.get("evaluation_status", "evaluated")
    if status not in {"evaluated", "unassessed"}:
        raise ValueError(f"invalid evaluation_status={status!r}")
    success = raw.get("success")
    if status == "evaluated" and not isinstance(success, bool):
        raise ValueError("evaluated score must contain boolean success")
    if status == "evaluated":
        success = bool(success)
        # Failed episodes must not receive time credit.  Completion time is
        # only defined on a successful terminal checker result.
        if success and raw.get("completion_ticks") is None:
            raise ValueError("successful score must contain completion_ticks")
        if not success and raw.get("completion_ticks") is not None:
            raise ValueError("failed score must omit completion_ticks")
        reward = 1 if success else 0
    else:
        success = None
        reward = None
    return {
        "schema_version": SCORE_SCHEMA,
        "evaluation_status": status,
        "real_rollout": bool(real),
        "backend": backend,
        "task_id": task["task_id"],
        "split": task["split"],
        "topology_id": task["topology_id"],
        "deadline": task["deadline_ticks"],
        "success": success,
        "reward": reward,
        "completion_ticks": raw.get("completion_ticks") if success else None,
        "checks": deepcopy(raw.get("checks", {})),
        "raw_score": deepcopy(dict(raw)),
    }


def _progress(score: Mapping[str, Any]) -> float:
    """Return checker progress for training-only dense shaping."""
    checks = score.get("checks", {})
    if not isinstance(checks, Mapping) or not checks:
        return 0.0
    # VillagerAgent's real construction checker reports fractional hit rates
    # (for example ``block_hit_rate=0.5``), while the symbolic checker reports
    # booleans.  Converting every value with ``bool`` made any non-zero rate
    # look like complete progress and produced a one-step shaping spike.  Only
    # finite numeric values and booleans are meaningful progress signals; clip
    # rates to the checker contract [0, 1] and ignore diagnostic strings.
    values = []
    for value in checks.values():
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(max(0.0, min(1.0, float(value))))
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
    source: str = "policy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrajectoryWriter:
    """Write immutable JSON records and atomic terminal score files."""

    def __init__(self, root: Path, *, require_real: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.require_real = require_real

    def write(self, episode: Mapping[str, Any]) -> Path:
        if self.require_real and not episode.get("real_rollout", False):
            raise ValueError("refusing to persist a symbolic episode as real rollout")
        episode_id = str(episode["episode_id"])
        target = self.root / f"{episode_id}.trajectory.json"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(episode, indent=2, sort_keys=True) + "\n")
        tmp.replace(target)
        return target

    def write_score(self, episode_id: str, score: Mapping[str, Any]) -> Path:
        if self.require_real and not score.get("real_rollout", False):
            raise ValueError("refusing to persist a symbolic score as real rollout")
        target = self.root / str(episode_id) / "score.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
        tmp.replace(target)
        return target


class RealMinecraftRLEnv:
    """Episode recorder around a real VillagerAgent adapter.

    The adapter owns Mineflayer, server reset and checker access.  This class
    only records observations/actions and computes training-only dense reward.
    The terminal reward remains the checker result and is never inferred from
    model output.
    """

    def __init__(self, adapter: RealMinecraftAdapter, *, require_real: bool = True,
                 max_ticks: Optional[int] = None):
        self.adapter = adapter
        self.require_real = require_real
        self.real_rollout = bool(getattr(adapter, "real_rollout", False))
        self.backend = str(getattr(adapter, "backend_name", adapter.__class__.__name__))
        if require_real and not self.real_rollout:
            raise ValueError("RealMinecraftRLEnv requires adapter.real_rollout=True")
        self.max_ticks = max_ticks
        self.task: Optional[Mapping[str, Any]] = None
        self.agents = []
        self.tick = 0
        self.episode_id = ""
        self.steps = []
        self._last_progress = 0.0

    def reset(self, task: Mapping[str, Any], episode_id: str) -> Dict[str, Mapping[str, Any]]:
        self.task = task
        self.episode_id = episode_id
        self.tick = 0
        self.steps = []
        self._last_progress = 0.0
        self.adapter.reset(task)
        roles = task.get("agent_roles", [])
        self.agents = [str(role["agent_id"]) for role in roles]
        return {agent: deepcopy(dict(self.adapter.observe(agent))) for agent in self.agents}

    @property
    def done(self) -> bool:
        return bool(self.max_ticks is not None and self.tick >= self.max_ticks)

    def step(self, actions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
        if self.task is None:
            raise RuntimeError("reset must be called before step")
        if self.done:
            raise RuntimeError("episode reached max_ticks")
        results = self.adapter.step_batch(actions)
        try:
            current_progress = _progress(self.adapter.score())
        except Exception:
            # A checker is authoritative only at episode end; a temporary
            # unavailable checker cannot fabricate success or failure.
            current_progress = self._last_progress
        shaping = max(-1.0, min(1.0, current_progress - self._last_progress))
        self._last_progress = current_progress
        observations = {agent: deepcopy(dict(self.adapter.observe(agent))) for agent in self.agents}
        for agent in self.agents:
            action = deepcopy(dict(actions.get(agent, {"type": "observe"})))
            result = deepcopy(dict(results.get(agent, {"ok": False, "error": "missing_result"})))
            valid = 1.0 if result.get("ok") is True else -1.0
            reward = 0.05 * shaping + 0.01 * valid
            self.steps.append(TrajectoryStep(self.episode_id, self.tick, agent,
                                              observations[agent], action, result,
                                              reward, False).to_dict())
        self.tick += 1
        return observations

    def finish(self) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("reset must be called before finish")
        raw = self.adapter.score()
        score = _normalise_score(raw, task=self.task, real=self.real_rollout, backend=self.backend)
        # Keep the terminal learning signal identical to the benchmark
        # contract and to the veRL worker: checker success is 1, an evaluated
        # failure is 0, and an unassessed checker is held out of training.
        # A previous ``-1`` failure signal disagreed with the official worker
        # and made the standalone real runner optimize a different objective.
        terminal = float(score["reward"]) if score["evaluation_status"] == "evaluated" else 0.0
        return {
            "schema_version": TRAJECTORY_SCHEMA,
            "episode_id": self.episode_id,
            "task_id": self.task["task_id"],
            "split": self.task["split"],
            "topology_id": self.task["topology_id"],
            "backend": self.backend,
            "real_rollout": self.real_rollout,
            "steps": self.steps,
            "tick": self.tick,
            "terminal_reward": terminal,
            "shaped_return": sum(step["reward"] for step in self.steps),
            "score": score,
        }


def group_advantages(episodes: Sequence[Mapping[str, Any]], *, key: str = "task_id") -> list[float]:
    """Compute centered, normalized GRPO advantages per task/candidate group."""
    groups: Dict[str, list[int]] = defaultdict(list)
    returns = []
    for index, episode in enumerate(episodes):
        returns.append(float(episode.get("terminal_reward", 0.0)))
        groups[str(episode.get(key, episode.get("episode_id", index)))].append(index)
    advantages = [0.0] * len(episodes)
    for indices in groups.values():
        values = [returns[i] for i in indices]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance) or 1.0
        for i in indices:
            advantages[i] = (returns[i] - mean_value) / scale
    return advantages


class OnPolicyGRPOLearner:
    """Policy-independent GRPO/REINFORCE batch coordinator."""

    def __init__(self, policy: RLPolicy, *, algorithm: str = "grpo"):
        algorithm = algorithm.lower()
        if algorithm not in {"grpo", "reinforce"}:
            raise ValueError("algorithm must be grpo or reinforce")
        self.policy = policy
        self.algorithm = algorithm

    def train_batch(self, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not episodes:
            raise ValueError("empty RL batch")
        for episode in episodes:
            if not episode.get("real_rollout", False):
                raise ValueError("RL update requires real_rollout trajectories")
            score = episode.get("score", {})
            if score.get("evaluation_status") != "evaluated":
                raise ValueError("RL update requires an evaluated checker score")
        # A single-candidate REINFORCE batch must retain its terminal signal;
        # only GRPO needs a within-task centered baseline.
        if self.algorithm == "grpo":
            counts = defaultdict(int)
            for episode in episodes:
                counts[str(episode.get("task_id", episode.get("episode_id")))] += 1
            if any(count < 2 for count in counts.values()):
                raise ValueError("GRPO requires at least two candidates for every task group")
        advantages = ([float(e.get("terminal_reward", 0.0)) for e in episodes]
                      if self.algorithm == "reinforce" else group_advantages(episodes))
        update = self.policy.update(episodes, advantages)
        group_sizes = defaultdict(int)
        for episode in episodes:
            group_sizes[str(episode.get("task_id", episode.get("episode_id")))] += 1
        return {"algorithm": self.algorithm, "episodes": len(episodes),
                "successes": sum(e["score"].get("success") is True for e in episodes),
                "mean_terminal_reward": sum(e.get("terminal_reward", 0.0) for e in episodes) / len(episodes),
                "group_sizes": sorted(group_sizes.values()),
                "advantages": advantages, "policy_update": dict(update or {})}


def load_score_rows(root: Path, *, require_real: bool = True) -> list[Dict[str, Any]]:
    """Read only evaluated, real score files for test/eval aggregation."""
    paths = [root] if root.is_file() else sorted(Path(root).rglob("score.json"))
    rows = []
    for path in paths:
        row = json.loads(path.read_text())
        if require_real and not row.get("real_rollout", False):
            raise ValueError(f"synthetic score is not valid for real eval: {path}")
        if row.get("evaluation_status") != "evaluated":
            raise ValueError(f"unassessed score cannot enter test/eval: {path}")
        rows.append(row)
    return rows
