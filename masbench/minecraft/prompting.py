"""Frozen prompt-based controllers for real Minecraft evaluation.

This module deliberately sits beside the RL path rather than inside it.  A
closed model is queried at every environment turn, but its weights never
change and no rollout is passed to veRL.  The real VillagerAgent adapter and
its checker remain the only source of environment transitions and success.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..controllers import ModelAdapter


PROMPT_VERSION = "minecraft-villageragent-typed-prompt-v1"

_PROMPT_REDACT_KEYS = frozenset({"task_id", "task_name", "split"})

# Keep this schema permissive for tool arguments.  The action ``type`` is
# constrained, while arguments such as coordinates, item names and targets are
# supplied by the real VillagerAgent tool contract.
ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "tool": {"type": "string"},
        "tool_input": {"type": "object"},
        "action": {"type": "object"},
    },
    "required": [],
    "additionalProperties": True,
}


ACTION_ALIASES = {
    "navigateTo": "move",
    "scanNearbyEntities": "observe",
    "MineBlock": "gather",
    "mine": "gather",
    "placeBlock": "place",
    "craftBlock": "craft",
    "SmeltingCooking": "cook",
    "handoverBlock": "handoff",
    "storeItem": "deposit",
    "Final Answer": "submit",
    "final_answer": "submit",
    "finish": "submit",
    "done": "submit",
}


SYSTEM_PROMPT = """You are one agent in a real Minecraft VillagerAgent task.
Use the observation, task goal, action history, teammate state and tool schema
to choose exactly one executable action for this turn. Return one JSON object
and no prose. The object must contain a `type` field. Valid action types are
observe, move, gather, craft, cook, place, attack, trade, handoff, deposit,
submit. Put the concrete VillagerAgent tool arguments at the top level of the
same object. Do not invent success, do not emit a final answer as prose, and do
not submit until the world really satisfies the goal. If an earlier action
failed, inspect the returned error and change the next action; do not repeat a
blocked navigation target forever. When an agent is busy, use observe or wait
for its result. This benchmark calls the real VillagerAgent tools directly:
move/navigateTo and gather/MineBlock require integer x, y and z coordinates;
observe/scanNearbyEntities may use item_name, radius and item_num; place and
craft/cook must include the concrete block or item arguments expected by the
tool. Do not emit an abstract subtask-only action when a real tool argument is
required. The action must be safe to execute immediately by the real adapter."""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task_view(task: Mapping[str, Any]) -> dict[str, Any]:
    """Keep all public task context while avoiding mutable caller state."""
    fields = (
        # Evaluation metadata (task IDs, names and split labels) is kept in
        # traces but never shown to the model.  Otherwise a prompt can reveal
        # that an example is held out or let a controller memorize task IDs.
        "family", "subfamily", "difficulty",
        "goal", "deadline_ticks", "world_seed", "world_config", "initial_inventory",
        "agent_roles", "action_space", "action_specs", "subtasks", "parallel_groups",
        "concrete_success_checks",
    )
    return {key: deepcopy(task[key]) for key in fields if key in task}


def _redact_prompt_metadata(value: Any) -> Any:
    """Remove benchmark identity labels from all model-visible state."""
    if isinstance(value, Mapping):
        return {
            str(key): _redact_prompt_metadata(item)
            for key, item in value.items()
            if str(key) not in _PROMPT_REDACT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_redact_prompt_metadata(item) for item in value]
    return deepcopy(value)


def build_messages(task: Mapping[str, Any], agent: str,
                   observation: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
                   peer_observations: Mapping[str, Any] | None = None,
                   *, max_history: int = 12,
                   prompt_version: str = PROMPT_VERSION) -> list[dict[str, str]]:
    """Build a reproducible system/user prompt matching the original loop.

    The task description, relevant world observation, teammate state, action
    history and tool knowledge are explicit in the user message.  A frozen
    ``prompt_version`` is stored with every episode so prompt changes cannot be
    silently compared as the same baseline.
    """
    payload = {
        "prompt_version": prompt_version,
        "agent": agent,
        "task": _task_view(task),
        "observation": _redact_prompt_metadata(dict(observation)),
        "teammate_observations": _redact_prompt_metadata(dict(peer_observations or {})),
        "recent_action_history": _redact_prompt_metadata(list(history[-max_history:])),
        "output_contract": {
            "one_action_per_turn": True,
            "json_only": True,
            "valid_types": [
                "observe", "move", "gather", "craft", "cook", "place",
                "attack", "trade", "handoff", "deposit", "submit",
            ],
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _json_text(payload)},
    ]


def _decode_raw(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise ValueError("model response must be a mapping or JSON string")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not valid JSON") from exc


def parse_action(raw: Any, *, agent: str,
                 action_space: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize original VillagerAgent tool-call shapes to MASbench actions.

    Original VillagerAgent prompts often return ``{"tool": ..., "tool_input":
    ...}``, while MASbench's adapter accepts ``{"type": ..., ...}``.  Both
    forms are accepted here.  Invalid output becomes a real ``observe`` action
    with private diagnostics; the adapter strips private keys before invoking a
    tool, and the episode trace retains the parser error.
    """
    allowed = set(action_space or (
        "observe", "move", "gather", "craft", "cook", "place", "attack",
        "trade", "handoff", "deposit", "submit",
    ))
    try:
        value = _decode_raw(raw)
        if not isinstance(value, Mapping):
            raise ValueError("response must be a JSON object")
        # Accept a joint response when a provider ignored the per-agent prompt.
        if agent in value and isinstance(value[agent], Mapping):
            value = value[agent]
        if isinstance(value.get("action"), Mapping):
            value = value["action"]
        if isinstance(value.get("tool_calls"), list) and value["tool_calls"]:
            first = value["tool_calls"][0]
            if isinstance(first, Mapping):
                function = first.get("function", first)
                if isinstance(function, Mapping) and function.get("name"):
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    value = {"tool": function["name"], "arguments": arguments}
                else:
                    value = function
        if "name" in value and "type" not in value and "tool" not in value:
            value = {"tool": value.get("name"),
                     "arguments": value.get("arguments", value.get("input", {}))}
        if "tool" in value:
            name = value.get("tool")
            arguments = value.get("tool_input", value.get("arguments", {}))
            if not isinstance(arguments, Mapping):
                raise ValueError("tool_input must be an object")
            value = dict(arguments)
            value["type"] = name
        elif "action_type" in value and "type" not in value:
            value = dict(value)
            value["type"] = value.pop("action_type")
        typ = value.get("type")
        if not isinstance(typ, str) or not typ.strip():
            raise ValueError("missing action type")
        typ = ACTION_ALIASES.get(typ, typ)
        if typ not in allowed:
            raise ValueError(f"unsupported action type: {typ}")
        action = {str(key): deepcopy(item) for key, item in value.items()
                  if key not in {"tool", "tool_input", "arguments", "tool_calls", "action"}}
        action["type"] = typ
        # The real VillagerAgent schemas for navigation, mining and placement
        # require explicit coordinates.  An abstract action must not reach
        # Pydantic inside the tool bridge and terminate the episode; turn it
        # into an observation so the model receives the concrete error on the
        # next turn and can recover from the current world state.
        if typ in {"move", "gather", "place"} and any(key not in action for key in ("x", "y", "z")):
            return {"type": "observe", "_valid": False,
                    "_parser_error": f"{typ} requires integer x, y, z"}, {
                "valid": False, "error": f"{typ} requires integer x, y, z",
                "raw": deepcopy(raw),
            }
        action["_valid"] = True
        return action, {"valid": True, "raw": deepcopy(raw)}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"type": "observe", "_valid": False, "_parser_error": str(exc)}, {
            "valid": False, "error": str(exc), "raw": deepcopy(raw),
        }


class PromptMinecraftController:
    """One frozen closed-model controller for one real episode."""

    def __init__(self, model: ModelAdapter, *, max_history: int = 12,
                 prompt_version: str = PROMPT_VERSION):
        self.model = model
        self.max_history = max_history
        self.prompt_version = prompt_version
        self.task: Mapping[str, Any] | None = None
        self.episode_id = ""
        self.history: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []

    def reset(self, task: Mapping[str, Any], episode_id: str):
        self.task = task
        self.episode_id = episode_id
        self.history = []
        self.decisions = []

    def act(self, observations: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        if self.task is None:
            raise RuntimeError("controller.reset must be called before act")
        actions: dict[str, dict[str, Any]] = {}
        decisions: list[dict[str, Any]] = []
        allowed = self.task.get("action_space")
        for agent in sorted(observations):
            observation = observations[agent]
            peers = {name: value for name, value in observations.items() if name != agent}
            messages = build_messages(self.task, agent, observation, self.history, peers,
                                      max_history=self.max_history,
                                      prompt_version=self.prompt_version)
            cache_key = f"{self.episode_id}:tick:{observation.get('tick', len(self.history))}:agent:{agent}"
            raw = self.model.generate(messages, response_schema=ACTION_RESPONSE_SCHEMA,
                                       cache_key=cache_key)
            action, diagnostic = parse_action(raw, agent=agent, action_space=allowed)
            actions[agent] = action
            decision = {"agent": agent, "tick": observation.get("tick"),
                        "messages": messages, "raw_output": deepcopy(raw),
                        "action": deepcopy(action), "diagnostic": diagnostic}
            decisions.append(decision)
            self.decisions.append(decision)
            self.history.append({"agent": agent, "tick": observation.get("tick"),
                                 "action": deepcopy(action),
                                 "result": None,
                                 "parser_valid": diagnostic.get("valid", False)})
        return actions, decisions

    def record_results(self, results: Mapping[str, Any]):
        """Attach adapter feedback to the latest per-agent history entries."""
        by_agent = {entry["agent"]: entry for entry in reversed(self.history)}
        for agent, result in results.items():
            if agent in by_agent:
                by_agent[agent]["result"] = deepcopy(result)


def normalize_score(raw: Mapping[str, Any], task: Mapping[str, Any], episode_id: str,
                    *, model_metadata: Mapping[str, Any], prompt_version: str) -> dict[str, Any]:
    """Normalize the checker result to the shared success-first score shape."""
    status = raw.get("evaluation_status")
    if status is None:
        status = "evaluated" if isinstance(raw.get("success"), bool) else "unassessed"
    if status not in {"evaluated", "unassessed"}:
        raise ValueError(f"invalid checker evaluation_status={status!r}")
    success = bool(raw["success"]) if status == "evaluated" else None
    completion = raw.get("completion_ticks") if success else None
    return {
        "schema_version": "masbench.minecraft.prompt_score.v1",
        "evaluation_status": status,
        "real_rollout": True,
        "backend": str(raw.get("backend", "villageragent_mineflayer")),
        "model_type": "closed_api",
        "weights_updated": bool(model_metadata.get("weights_updated", False)),
        "controller_trained": bool(model_metadata.get("controller_trained", False)),
        "test_frozen": bool(model_metadata.get("test_frozen", True)),
        "model": dict(model_metadata),
        "prompt_version": prompt_version,
        "episode_id": episode_id,
        "task_id": task["task_id"],
        "split": task["split"],
        "topology_id": task.get("topology_id"),
        "deadline": task["deadline_ticks"],
        "success": success,
        "reward": int(success) if status == "evaluated" else None,
        "completion_ticks": completion,
        "checks": deepcopy(raw.get("checks", {})),
        "raw_score": deepcopy(dict(raw)),
    }


class PromptMinecraftRunner:
    """Execute an API-only episode without invoking the RL learner."""

    def __init__(self, adapter_factory, model: ModelAdapter, *, max_ticks: int | None = None,
                 max_history: int = 12, prompt_version: str = PROMPT_VERSION,
                 model_metadata: Mapping[str, Any] | None = None):
        self.adapter_factory = adapter_factory
        self.model = model
        self.max_ticks = max_ticks
        self.max_history = max_history
        self.prompt_version = prompt_version
        self.model_metadata = dict(model_metadata or {})

    def run_episode(self, task: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
        adapter = self.adapter_factory(task, episode_id)
        if not getattr(adapter, "real_rollout", False):
            raise ValueError("closed-model Minecraft evaluation requires a real_rollout adapter")
        controller = PromptMinecraftController(self.model, max_history=self.max_history,
                                                prompt_version=self.prompt_version)
        controller.reset(task, episode_id)
        agents = [str(role["agent_id"]) for role in task.get("agent_roles", [])]
        if not agents:
            raise ValueError("task has no agent_roles")
        horizon = int(self.max_ticks or task["deadline_ticks"])
        if horizon <= 0:
            raise ValueError("max_ticks must be positive")
        steps: list[dict[str, Any]] = []
        submitted = False
        try:
            adapter.reset(task)
            observations = {agent: deepcopy(dict(adapter.observe(agent))) for agent in agents}
            for tick in range(horizon):
                actions, decisions = controller.act(observations)
                results = adapter.step_batch(actions)
                controller.record_results(results)
                next_observations = {agent: deepcopy(dict(adapter.observe(agent))) for agent in agents}
                steps.append({"tick": tick, "observations": deepcopy(observations),
                              "actions": deepcopy(actions), "results": deepcopy(dict(results)),
                              "decisions": deepcopy(decisions)})
                observations = next_observations
                submitted = submitted or any(action.get("type") == "submit"
                                              for action in actions.values())
                if submitted:
                    break
            raw_score = adapter.score()
            score = normalize_score(raw_score, task, episode_id,
                                    model_metadata=self.model_metadata,
                                    prompt_version=self.prompt_version)
            return {
                "schema_version": "masbench.minecraft.prompt_episode.v1",
                "episode_id": episode_id,
                "task_id": task["task_id"],
                "split": task["split"],
                "topology_id": task.get("topology_id"),
                "backend": str(getattr(adapter, "backend_name", adapter.__class__.__name__)),
                "real_rollout": True,
                "model_type": "closed_api",
                "weights_updated": bool(self.model_metadata.get("weights_updated", False)),
                "controller_trained": bool(self.model_metadata.get("controller_trained", False)),
                "test_frozen": bool(self.model_metadata.get("test_frozen", True)),
                "prompt_version": self.prompt_version,
                "model": deepcopy(self.model_metadata),
                "steps": steps,
                "ticks": len(steps),
                "submitted": submitted,
                "score": score,
            }
        finally:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()


def write_episode(root: str | Path, episode: Mapping[str, Any]) -> tuple[Path, Path]:
    """Persist the detailed trace and the separate authoritative score."""
    root = Path(root)
    episode_dir = root / str(episode["episode_id"])
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episode_dir / "episode.json"
    score_path = episode_dir / "score.json"
    for path, payload in ((episode_path, episode), (score_path, episode["score"])):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    return episode_path, score_path
