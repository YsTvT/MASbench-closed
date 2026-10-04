#!/usr/bin/env python3
"""Run frozen policy on a real Bench2/Bench3 test batch and aggregate scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from masbench.longhorizon.rl import RealLongHorizonRLEnv, TrajectoryWriter, load_factory, load_score_rows
from masbench.metrics import success_first

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = {"bench2": "bench2_workflow", "bench3": "bench3_incident"}


def _cases(bench):
    path = ROOT / "data" / DATA_DIRS[bench] / "test.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(case, adapter, policy, eid, ticks):
    env = RealLongHorizonRLEnv(adapter, max_ticks=ticks)
    observations = env.reset(case, eid)
    for _ in range(ticks):
        actions = {}
        for agent, observation in observations.items():
            action = policy.sample_action(observation, agent, deterministic=True)
            if not isinstance(action, Mapping) or not isinstance(action.get("type"), str):
                raise ValueError(f"policy emitted invalid action for {agent}: {action!r}")
            actions[agent] = dict(action)
        observations = env.step(actions)
        if env.done:
            break
    return env.finish()


def run(args):
    adapter_factory = load_factory(args.adapter_factory)
    policy = load_factory(args.policy_factory)()
    if args.checkpoint and not hasattr(policy, "load_checkpoint"):
        raise ValueError("policy factory does not support load_checkpoint")
    if args.checkpoint:
        policy.load_checkpoint(Path(args.checkpoint))
    writer = TrajectoryWriter(Path(args.out))
    benches = ("bench2", "bench3") if args.bench == "all" else (args.bench,)
    for bench in benches:
        for case in _cases(bench)[:args.max_tasks]:
            eid = f"{case['task_id']}__eval"
            episode = _run(case, adapter_factory(case, eid), policy, eid, args.max_ticks)
            writer.write(episode)
    rows = load_score_rows(Path(args.out) / "scores", split="test")
    groups = {}
    for family in ("workflow", "incident"):
        family_rows = [row for row in rows if row.get("family") == family]
        if family_rows:
            groups[family] = success_first(family_rows)
    summary = {"schema_version": "masbench.longhorizon.rl_eval.v1", "real_rollout": True,
               "split": "test", "bench": args.bench, "groups": groups, "score_files": len(rows)}
    _write(Path(args.out) / "eval_summary.json", summary)
    return summary


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=("bench2", "bench3", "all"), default="all")
    p.add_argument("--adapter-factory", required=True)
    p.add_argument("--policy-factory", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--out", default="MASbench/results/bench23_rl_test")
    p.add_argument("--max-tasks", type=int, default=10**9)
    p.add_argument("--max-ticks", type=int, default=128)
    print(json.dumps(run(p.parse_args()), indent=2, sort_keys=True))
