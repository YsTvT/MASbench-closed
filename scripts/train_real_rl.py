#!/usr/bin/env python3
"""Train a policy from real Minecraft rollouts.

The default command intentionally fails unless a real adapter and a trainable
policy factory are supplied.  ``--dry-run`` uses the symbolic adapter only to
test the trajectory plumbing; its records are marked ``real_rollout=false``
and are never passed to the learner or benchmark evaluator.

Factories use ``module:object`` syntax.  An adapter factory receives
``(task, episode_id)`` and returns an object implementing
``RealMinecraftAdapter``.  A policy factory receives no arguments and returns
an object implementing ``RLPolicy`` (a Qwen LoRA wrapper can live in a
private/remote module without adding GPU dependencies to MASbench).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from masbench.minecraft.env import SymbolicMinecraftEnv
from masbench.minecraft.rl import (OnPolicyGRPOLearner, RealMinecraftRLEnv,
                                   TrajectoryWriter, load_factory)
from masbench.minecraft.tasks import load_tasks


class _SymbolicAdapter:
    backend_name = "symbolic"
    real_rollout = False

    def __init__(self):
        self.env = None

    def reset(self, task):
        self.env = SymbolicMinecraftEnv(task)

    def observe(self, agent):
        return self.env.observe(agent)

    def step_batch(self, actions):
        return self.env.step_batch(actions)

    def score(self):
        return self.env.score()


class _DryRunPolicy:
    """Deterministic graph policy for plumbing tests, never used for RL."""

    def __init__(self, task):
        self.task = task

    def sample_action(self, observation: Mapping[str, Any], agent: str, *, deterministic=False):
        if observation.get("done"):
            return {"type": "submit"}
        completed = set(observation.get("completed_subtasks", []))
        if len(completed) == len(self.task["subtasks"]):
            return {"type": "submit"}
        available = []
        for subtask in self.task["subtasks"]:
            if subtask["id"] in completed:
                continue
            if all(dep in completed for dep in subtask.get("depends_on", [])):
                available.append(subtask)
        if not available:
            return {"type": "observe"}
        # Give each root branch to a different agent; later dependent nodes
        # are assigned to the first agent.  This is only a contract test.
        index = 0 if agent.endswith("0") else 1
        subtask = available[index % len(available)]
        return {"type": subtask["action"], "subtask_id": subtask["id"]}

    def update(self, trajectories, advantages):
        raise RuntimeError("dry-run policy must never receive an RL update")

    def save_checkpoint(self, path, metadata):
        raise RuntimeError("dry-run policy has no checkpoint")


def _run_episode(task, adapter, policy, episode_id, max_ticks):
    env = RealMinecraftRLEnv(adapter, require_real=bool(getattr(adapter, "real_rollout", False)),
                             max_ticks=max_ticks)
    observations = env.reset(task, episode_id)
    submitted = False
    try:
        for _ in range(max_ticks):
            actions = {}
            for agent, observation in observations.items():
                action = policy.sample_action(observation, agent, deterministic=False)
                if not isinstance(action, Mapping) or not isinstance(action.get("type"), str):
                    raise ValueError(f"policy emitted invalid action for {agent}: {action!r}")
                actions[agent] = dict(action)
            observations = env.step(actions)
            submitted = submitted or any(a.get("type") == "submit" for a in actions.values())
            if submitted:
                break
        return env.finish()
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


def _write_json(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run(args):
    tasks = load_tasks(Path(args.data_dir) / "minecraft_tasks.jsonl", split="train")
    if args.dry_run:
        task = tasks[0]
        adapter = _SymbolicAdapter()
        policy = _DryRunPolicy(task)
        episode = _run_episode(task, adapter, policy, "dry_run_000", args.max_ticks)
        episode["real_rollout"] = False
        episode["dry_run"] = True
        writer = TrajectoryWriter(Path(args.out), require_real=False)
        writer.write(episode)
        writer.write_score(episode["episode_id"], episode["score"] | {"real_rollout": False})
        report = {"mode": "dry-run", "real_rollout": False,
                  "episodes": 1, "message": "symbolic plumbing only; no RL update"}
        _write_json(Path(args.out) / "manifest.json", report)
        return report

    if not args.adapter_factory or not args.policy_factory:
        raise SystemExit("real training requires --adapter-factory and --policy-factory; use --dry-run for plumbing")
    if args.algorithm == "grpo" and args.candidates < 2:
        raise SystemExit("strict GRPO requires at least 2 stochastic candidates per task")
    if args.algorithm == "grpo" and args.batch_size % args.candidates != 0:
        raise SystemExit("GRPO batch-size must be a multiple of candidates so task groups stay intact")
    adapter_factory = load_factory(args.adapter_factory)
    policy = load_factory(args.policy_factory)()
    writer = TrajectoryWriter(Path(args.out), require_real=True)
    learner = OnPolicyGRPOLearner(policy, algorithm=args.algorithm)
    episodes = []
    updates = []
    for task in tasks[: args.max_tasks]:
        task_group = []
        for candidate in range(args.candidates):
            episode_id = f"{task['task_id']}__rl__{candidate:02d}"
            adapter = adapter_factory(task, episode_id)
            episode = _run_episode(task, adapter, policy, episode_id, args.max_ticks)
            writer.write(episode)
            writer.write_score(episode_id, episode["score"])
            episodes.append(episode)
            task_group.append(episode)
            if len(episodes) >= args.batch_size:
                update = learner.train_batch(episodes)
                updates.append(update)
                _write_json(Path(args.out) / f"update_{len(updates):04d}.json", update)
                episodes = []
    if episodes:
        updates.append(learner.train_batch(episodes))
        _write_json(Path(args.out) / f"update_{len(updates):04d}.json", updates[-1])
    metadata = {"schema_version": "masbench.minecraft.rl_run.v1", "mode": "train",
                "real_rollout": True, "algorithm": args.algorithm,
                "train_tasks": len(tasks[: args.max_tasks]), "updates": len(updates)}
    policy.save_checkpoint(Path(args.out) / "checkpoint-final", metadata)
    _write_json(Path(args.out) / "manifest.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="MASbench/data")
    parser.add_argument("--out", default="MASbench/results/minecraft_rl_train")
    parser.add_argument("--adapter-factory", help="module:factory returning a real VillagerAgent adapter")
    parser.add_argument("--policy-factory", help="module:factory returning a trainable Qwen/LoRA policy")
    parser.add_argument("--algorithm", choices=("reinforce", "grpo"), default="reinforce")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tasks <= 0:
        args.max_tasks = 10**9
    print(json.dumps(run(args), indent=2, sort_keys=True))
