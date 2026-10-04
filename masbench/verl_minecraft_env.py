"""veRL environment manager for real VillagerAgent Minecraft rollouts.

Each GRPO candidate receives its own copied Minecraft server directory, world
snapshot, and HTTP ports.  A global world lock would deadlock veRL's group
workers because they advance in lock-step.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_VALID_ACTION_TYPES = frozenset({
    "observe", "scanNearbyEntities", "move", "navigateTo", "gather", "mine",
    "MineBlock", "place", "placeBlock", "craft", "craftBlock", "cook",
    "SmeltingCooking", "withdraw", "withdrawItem", "attack", "trade",
    "handoff", "handoverBlock", "deposit", "storeItem", "submit",
})
_COORD_ACTION_TYPES = frozenset({
    "move", "navigateTo", "gather", "mine", "MineBlock", "place", "placeBlock",
})


def _action_is_valid(value: Mapping[str, Any]) -> bool:
    action_type = value.get("type")
    if action_type not in _VALID_ACTION_TYPES:
        return False
    if action_type in _COORD_ACTION_TYPES:
        for key in ("x", "y", "z"):
            coordinate = value.get(key)
            if isinstance(coordinate, bool):
                return False
            try:
                if not math.isfinite(float(coordinate)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _typed_action(value: Any, invalid: Mapping[str, Any]) -> dict[str, Any]:
    if (isinstance(value, Mapping)
            and isinstance(value.get("type"), str)
            and _action_is_valid(value)):
        return dict(value)
    return dict(invalid)


def _parse_joint(text: str) -> dict[str, dict[str, Any]]:
    text = str(text)
    start, end = text.find("{"), text.rfind("}")
    invalid = {"type": "observe", "_valid": False}
    if start < 0 or end <= start:
        return {"agent_0": invalid.copy(), "agent_1": invalid.copy()}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"agent_0": invalid.copy(), "agent_1": invalid.copy()}
    if isinstance(obj, Mapping) and isinstance(obj.get("type"), str):
        return {"agent_0": _typed_action(obj, invalid), "agent_1": {"type": "observe"}}
    if not isinstance(obj, Mapping):
        return {"agent_0": invalid.copy(), "agent_1": invalid.copy()}
    out: dict[str, dict[str, Any]] = {}
    for key, value in obj.items():
        key = {"Alice": "agent_0", "Bob": "agent_1"}.get(str(key), str(key))
        if key not in {"agent_0", "agent_1"}:
            continue
        out[key] = _typed_action(value, invalid)
    return out or {"agent_0": invalid.copy(), "agent_1": invalid.copy()}


def _joint_action_is_valid(text: str) -> bool:
    """Validate the JSON action before veRL applies invalid-action penalties.

    The old projection only checked for a pair of braces.  A malformed JSON
    completion therefore received ``is_action_valid=True`` even though the
    worker had to replace it with an observe action.  Keep the permissive
    single-agent form accepted by ``_parse_joint`` for backwards
    compatibility, but require at least one recognized agent action with a
    string ``type``.
    """
    text = str(text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        obj = json.loads(text[start:end + 1])
    except (TypeError, json.JSONDecodeError):
        return False
    if isinstance(obj, Mapping) and isinstance(obj.get("type"), str):
        return _action_is_valid(obj)
    if not isinstance(obj, Mapping):
        return False
    recognized = False
    for key, value in obj.items():
        key = {"Alice": "agent_0", "Bob": "agent_1"}.get(str(key), str(key))
        if key in {"agent_0", "agent_1"} and isinstance(value, Mapping):
            if _action_is_valid(value):
                recognized = True
    return recognized


class MinecraftWorker:
    def __init__(self, seed: int, env_kwargs: Mapping[str, Any] | None = None,
                 candidate_id: int = 0):
        self.seed = int(seed)
        self.candidate_id = int(candidate_id)
        self.env_kwargs = dict(env_kwargs or {})
        self.task: dict[str, Any] | None = None
        self.adapter = None
        self.turn = 0
        self.max_turns = int(self.env_kwargs.get("max_turns", 32))
        self.done = False
        self.final_text = ""
        self.final_info: dict[str, Any] = {}

    def _candidate_task(self, task: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(task, (str, bytes, bytearray)):
            try:
                task = json.loads(task)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Minecraft env_kwargs must be a task mapping or JSON object") from exc
        if not isinstance(task, Mapping):
            raise TypeError(f"Minecraft task must be a mapping, got {type(task).__name__}")
        # RLHFDataset/agent_system releases differ in whether they unwrap the
        # parquet ``env_kwargs`` column before calling reset.  Accept the
        # direct task, a JSON-encoded env_kwargs field, or a mapping wrapper,
        # while retaining any top-level task overrides.
        if "env_kwargs" in task:
            payload = task.get("env_kwargs")
            if isinstance(payload, (list, tuple)) and len(payload) == 1:
                payload = payload[0]
            if isinstance(payload, (str, bytes, bytearray)):
                try:
                    payload = json.loads(payload)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("Minecraft env_kwargs contains invalid JSON") from exc
            if isinstance(payload, Mapping):
                merged = dict(payload)
                merged.update({key: value for key, value in task.items()
                               if key != "env_kwargs" and key not in {"prompt", "extra_info"}})
                task = merged
        elif "task" in task and isinstance(task.get("task"), Mapping):
            merged = dict(task["task"])
            merged.update({key: value for key, value in task.items() if key != "task"})
            task = merged
        result = dict(task)
        base_server = (result.get("base_server_root") or self.env_kwargs.get("server_root")
                       or result.get("server_root"))
        source_snapshot = (result.get("source_world_snapshot")
                           or self.env_kwargs.get("world_snapshot")
                           or result.get("world_snapshot"))
        if base_server:
            result["base_server_root"] = str(base_server)
            result["server_root"] = str(Path(base_server) / ".masbench_candidates"
                                         / f"candidate_{self.candidate_id:03d}")
        if source_snapshot:
            source_snapshot = Path(source_snapshot)
            result["source_world_snapshot"] = str(source_snapshot)
            result["world_snapshot"] = str(source_snapshot.parent
                                           / f"{source_snapshot.name}_candidate_{self.candidate_id:03d}")
        # Existing source-task port fields are source defaults, never a
        # license for candidates to share a listening port.
        result["server_port"] = int(self.env_kwargs.get("server_port_base", 25565)) + self.candidate_id
        result["agent_port_base"] = int(self.env_kwargs.get("agent_port_base", 5000)) + 10 * self.candidate_id
        base_villager = (result.get("base_villageragent_root")
                         or self.env_kwargs.get("villageragent_root")
                         or result.get("villageragent_root"))
        if base_villager:
            result["base_villageragent_root"] = str(base_villager)
            result["villageragent_root"] = str(Path(base_villager) / ".masbench_candidates"
                                               / f"candidate_{self.candidate_id:03d}")
        base_name = str(result.get("task_name") or result.get("task_id") or "minecraft")
        result["task_name"] = f"{base_name}__candidate_{self.candidate_id:03d}"
        result.setdefault("max_turns", self.max_turns)
        return result

    def reset(self, task: Mapping[str, Any] | None = None):
        self.close()
        self.task = self._candidate_task(task or self.env_kwargs.get("task", {}))
        self.adapter = None
        self.turn = 0
        self.done = False
        self.final_text = ""
        self.final_info = {}
        return self._pending_observation(), {"real_rollout": True, "reset_pending": True}

    def _pending_observation(self):
        task = self.task or {}
        return json.dumps({"task_id": task.get("task_id"), "goal": task.get("goal"),
                           "agent_roles": task.get("agent_roles", []),
                           "instruction": "Emit one JSON object mapping agent_0 and agent_1 to executable actions. First action should observe.",
                           "real_rollout": True}, ensure_ascii=False, sort_keys=True)

    def _ensure_started(self):
        if self.adapter is not None:
            return
        from masbench.minecraft.villageragent_real import make_adapter
        task = self.task or {}
        episode_id = f"{task.get('task_id', 'minecraft')}__verl__{self.seed}"
        self.adapter = make_adapter(task, episode_id)
        self.adapter.reset(task)

    def step(self, text_action: str):
        if self.done:
            return self.final_text, 0.0, True, dict(self.final_info)
        self._ensure_started()
        actions = _parse_joint(text_action)
        # A malformed action or a transient Mineflayer tool exception is an
        # invalid transition for this candidate.  It must not tear down the
        # Ray actor (and therefore abort the whole GRPO update).  Dispatch
        # each agent independently so one failing tool does not suppress the
        # other agent's action; the checker remains the only terminal reward
        # authority.
        result = {}
        for agent, action in sorted(actions.items()):
            try:
                piece = self.adapter.step_batch({agent: action})
                if isinstance(piece, Mapping):
                    result.update(piece)
                else:
                    result[agent] = {"ok": False, "error": "tool_invalid_result"}
            except Exception as exc:  # tool failures are per-action penalties
                result[agent] = {"ok": False,
                                 "error": f"tool_exception:{type(exc).__name__}",
                                 "detail": str(exc)[:240]}
        self.turn += 1
        observations = {}
        for agent in ("agent_0", "agent_1"):
            try:
                observations[agent] = self.adapter.observe(agent)
            except Exception as exc:
                # Preserve a typed observation so the policy can recover on
                # the next turn.  This is intentionally not checker evidence.
                observations[agent] = {
                    "agent": agent,
                    "task_id": (self.task or {}).get("task_id"),
                    "observation_error": f"{type(exc).__name__}:{str(exc)[:240]}",
                }
        terminal = any(a.get("type") == "submit" for a in actions.values()) or self.turn >= self.max_turns
        if terminal:
            try:
                score = self.adapter.score()
            except Exception as exc:
                score = {"evaluation_status": "unassessed", "checks": {},
                         "reason": f"checker_exception:{type(exc).__name__}",
                         "detail": str(exc)[:240], "real_rollout": True}
        else:
            score = None
        text = json.dumps({"observations": observations, "last_results": result,
                           "score": score, "turn": self.turn}, ensure_ascii=False, sort_keys=True)
        if terminal:
            status = score.get("evaluation_status") if score else "unassessed"
            if os.environ.get("MASBENCH_REQUIRE_EVALUATED", "1") == "1" and status != "evaluated":
                self.adapter.close(); self.adapter = None
                raise RuntimeError(f"real checker did not produce evaluated evidence: {score}")
            success = bool(score and score.get("success"))
            info = {"real_rollout": True, "evaluation_status": status,
                    "success": success, "won": success, "checker": score}
            self.adapter.close(); self.adapter = None
            self.done, self.final_text, self.final_info = True, text, dict(info)
            # Keep the environment reward equal to the benchmark terminal
            # contract (1 for checker success, 0 for an evaluated failure).
            # Invalid-action handling is exposed separately via
            # ``is_action_valid`` in the manager; it must not turn a failed
            # world checker into a shaped success signal.
            return text, float(success), True, info
        return text, 0.0, False, {"real_rollout": True, "evaluation_status": "running"}

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


def _make_worker(seed, env_kwargs, candidate_id, resources_per_worker=None):
    import ray
    resources = dict(resources_per_worker or {"num_cpus": 1, "num_gpus": 0})
    return ray.remote(**resources)(MinecraftWorker).remote(seed, env_kwargs, candidate_id)


class MinecraftMultiProcessEnv:
    def __init__(self, seed: int, env_num: int, group_n: int,
                 resources_per_worker: Mapping[str, Any], is_train: bool = True,
                 env_kwargs: Mapping[str, Any] | None = None,
                 candidate_offset: int = 0):
        import ray
        del is_train
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.group_n, self.env_num = int(group_n), int(env_num)
        # Train and validation managers are constructed together by
        # ``make_envs``.  Give them disjoint candidate ids so their server
        # roots and ports cannot collide if validation runs while training.
        self.candidate_offset = int(candidate_offset)
        self.env_kwargs = dict(env_kwargs or {})
        self.workers = [_make_worker(seed + i, self.env_kwargs,
                                     self.candidate_offset + i, resources_per_worker)
                        for i in range(self.env_num * self.group_n)]

    def reset(self, kwargs=None):
        if kwargs is None:
            tasks = [self.env_kwargs.get("task", {}) for _ in self.workers]
        elif isinstance(kwargs, np.ndarray):
            tasks = list(kwargs.tolist())
        elif isinstance(kwargs, (list, tuple)):
            tasks = list(kwargs)
        else:
            tasks = [kwargs]
        if len(tasks) == self.env_num and self.group_n > 1:
            tasks = [task for task in tasks for _ in range(self.group_n)]
        if len(tasks) != len(self.workers):
            raise ValueError(f"env_kwargs count {len(tasks)} != worker count {len(self.workers)}")
        results = __import__("ray").get([w.reset.remote(task) for w, task in zip(self.workers, tasks)])
        return [r[0] for r in results], [r[1] for r in results]

    def step(self, actions):
        if len(actions) != len(self.workers):
            raise ValueError(f"action count {len(actions)} != worker count {len(self.workers)}")
        results = __import__("ray").get([w.step.remote(a) for w, a in zip(self.workers, actions)])
        return ([r[0] for r in results], [r[1] for r in results],
                np.asarray([r[2] for r in results], dtype=bool), [r[3] for r in results])

    def close(self):
        __import__("ray").get([w.close.remote() for w in self.workers])


def build_minecraft_envs(seed, env_num, group_n, resources_per_worker,
                         is_train=True, env_kwargs=None, candidate_offset=0):
    return MinecraftMultiProcessEnv(seed, env_num, group_n, resources_per_worker,
                                    is_train=is_train, env_kwargs=env_kwargs,
                                    candidate_offset=candidate_offset)


def minecraft_projection(actions):
    return actions, [_joint_action_is_valid(a) for a in actions]


try:
    from agent_system.environments.base import EnvironmentManagerBase as _EnvironmentManagerBase
except Exception:
    class _EnvironmentManagerBase:  # type: ignore[no-redef]
        pass


class MinecraftEnvironmentManager(_EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        try:
            super().__init__(envs, projection_f, config)
        except TypeError:
            self.envs, self.projection_f, self.config = envs, projection_f, config

    def reset(self, kwargs=None):
        obs, infos = self.envs.reset(kwargs=kwargs)
        return {"text": obs, "image": None, "anchor": obs}, infos

    def step(self, text_actions):
        actions, valid = self.projection_f(text_actions)
        obs, rewards, dones, infos = self.envs.step(actions)
        for i, info in enumerate(infos):
            info["is_action_valid"] = valid[i]
        return ({"text": obs, "image": None, "anchor": obs},
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def success_evaluator(self, total_infos=None, total_batch_list=None,
                          episode_rewards=None, episode_lengths=None, **kwargs):
        del total_batch_list, episode_rewards, episode_lengths, kwargs
        # agent_system versions in the wild pass either ``[episode_infos]``
        # (nested) or a flat list of step infos.  Normalize both forms while
        # preserving one success value per candidate episode.  A flat list of
        # terminal records is also supported for validation/eval callers.
        tolist = getattr(total_infos, "tolist", None)
        if callable(tolist):
            total_infos = tolist()
        if total_infos is None:
            rows = []
        elif isinstance(total_infos, Mapping):
            rows = [[total_infos]]
        else:
            raw_rows = list(total_infos)
            normalized_rows = []
            for item in raw_rows:
                item_tolist = getattr(item, "tolist", None)
                normalized_rows.append(item_tolist() if callable(item_tolist) else item)
            raw_rows = normalized_rows
            if raw_rows and all(isinstance(item, Mapping) for item in raw_rows):
                terminal_count = sum(
                    item.get("evaluation_status") in {"evaluated", "unassessed"}
                    for item in raw_rows
                )
                rows = ([[item] for item in raw_rows]
                        if terminal_count > 1 else [raw_rows])
            else:
                rows = [item if isinstance(item, (list, tuple)) else [item]
                        for item in raw_rows]
        successes = []
        for trajectory in rows:
            sequence = trajectory if isinstance(trajectory, (list, tuple)) else [trajectory]
            terminal = next((item for item in reversed(sequence)
                             if isinstance(item, Mapping) and item.get("evaluation_status")
                             in {"evaluated", "unassessed"}), {})
            if terminal.get("evaluation_status") != "evaluated":
                raise RuntimeError(f"Minecraft checker evidence is not evaluated: {terminal}")
            successes.append(float(bool(terminal.get("won", terminal.get("success", False)))))
        return {"success_rate": np.asarray(successes, dtype=np.float32)}

    def close(self):
        self.envs.close()
