#!/usr/bin/env python3
"""Collect real Bench2/Bench3 train rollouts and perform policy updates.

This command deliberately has no symbolic fallback.  The adapter factory must
connect to the real workflow/incident backend and set ``real_rollout=True``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from masbench.longhorizon.rl import (OnPolicyGRPOLearner, RealLongHorizonRLEnv,
                                     TrajectoryWriter, load_factory)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = {"bench2": "bench2_workflow", "bench3": "bench3_incident"}


def _load_cases(bench: str, split: str):
    path = ROOT / "data" / DATA_DIRS[bench] / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode(case: Mapping[str, Any], adapter, policy, episode_id: str, max_ticks: int):
    env = RealLongHorizonRLEnv(adapter, max_ticks=max_ticks)
    observations = env.reset(case, episode_id)
    for _ in range(max_ticks):
        actions = {}
        for agent, observation in observations.items():
            action = policy.sample_action(observation, agent, deterministic=False)
            if not isinstance(action, Mapping) or not isinstance(action.get("type"), str):
                raise ValueError(f"policy emitted invalid action for {agent}: {action!r}")
            actions[agent] = dict(action)
        observations = env.step(actions)
        if env.done:
            break
    return env.finish()


def run(args):
    if args.split != "train":
        raise ValueError("training data must use split=train")
    adapter_factory = load_factory(args.adapter_factory)
    policy = load_factory(args.policy_factory)()
    writer = TrajectoryWriter(Path(args.out))
    learner = OnPolicyGRPOLearner(policy, algorithm=args.algorithm)
    episodes, updates = [], []
    benches = ("bench2", "bench3") if args.bench == "all" else (args.bench,)
    for bench in benches:
        for case in _load_cases(bench, "train")[:args.max_tasks]:
            for candidate in range(args.candidates):
                eid = f"{case['task_id']}__rl__{candidate:02d}"
                adapter = adapter_factory(case, eid)
                episode = _episode(case, adapter, policy, eid, args.max_ticks)
                writer.write(episode)
                episodes.append(episode)
                if len(episodes) >= args.batch_size:
                    update = learner.train_batch(episodes)
                    updates.append(update)
                    _write(Path(args.out) / f"update_{len(updates):04d}.json", update)
                    episodes = []
    if episodes:
        update = learner.train_batch(episodes)
        updates.append(update)
        _write(Path(args.out) / f"update_{len(updates):04d}.json", update)
    metadata = {"schema_version": "masbench.longhorizon.rl_run.v1", "mode": "train",
                "real_rollout": True, "algorithm": args.algorithm, "bench": args.bench,
                "train_split": "train", "updates": len(updates)}
    policy.save_checkpoint(Path(args.out) / "checkpoint-final", metadata)
    _write(Path(args.out) / "manifest.json", metadata)
    return metadata


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=("bench2", "bench3", "all"), default="all")
    p.add_argument("--split", choices=("train",), default="train")
    p.add_argument("--adapter-factory", required=True)
    p.add_argument("--policy-factory", required=True)
    p.add_argument("--out", default="MASbench/results/bench23_rl_train")
    p.add_argument("--algorithm", choices=("reinforce", "grpo"), default="reinforce")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--max-tasks", type=int, default=10**9)
    p.add_argument("--max-ticks", type=int, default=128)
    print(json.dumps(run(p.parse_args()), indent=2, sort_keys=True))
