#!/usr/bin/env python3
"""Run a frozen Qwen/LoRA checkpoint through the real Minecraft checker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from masbench.minecraft.splits import SplitLeakageError, audit_split_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", choices=("test",), default="test",
                        help="real Qwen evaluation is test-only")
    parser.add_argument("--train-tasks", default=None,
                        help="optional train JSONL for leakage audit; defaults to a train.jsonl sibling")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-turns", type=int, default=0,
                        help="override horizon; 0 uses each task's deadline_ticks")
    parser.add_argument("--villager-root", default=os.environ.get("VILLAGERAGENT_ROOT", "vendor/VillagerAgent"))
    parser.add_argument("--server-root", default=os.environ.get("MASBENCH_SERVER_ROOT"))
    parser.add_argument("--world-snapshot", default=os.environ.get("MASBENCH_WORLD_SNAPSHOT"))
    parser.add_argument("--server-port", type=int, default=25600)
    parser.add_argument("--agent-port-base", type=int, default=6000)
    args = parser.parse_args()

    from masbench.minecraft.qwen_lora_policy import QwenLoRAPolicy
    from masbench.minecraft.villageragent_real import make_adapter

    tasks_path = Path(args.tasks)
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if not tasks:
        raise SystemExit("test task file is empty")
    bad_split = [str(task.get("task_id", "<unknown>")) for task in tasks
                 if task.get("split") != args.split]
    if bad_split:
        raise SystemExit(f"{tasks_path} contains non-test rows: {bad_split[:8]}")
    split_audit = None
    train_path = Path(args.train_tasks) if args.train_tasks else tasks_path.with_name("train.jsonl")
    if not train_path.is_file():
        raise SystemExit(
            "test evaluation requires --train-tasks (or a train.jsonl sibling); "
            "refusing an unaudited test run"
        )
    try:
        split_audit = audit_split_files(train_path, tasks_path)
    except SplitLeakageError as exc:
        raise SystemExit(f"MINECRAFT SPLIT AUDIT FAILED: {exc}") from exc
    policy = QwenLoRAPolicy(args.model, checkpoint=args.checkpoint, max_new_tokens=128)
    records = []
    for index, task in enumerate(tasks):
        task = dict(task)
        task.setdefault("villageragent_root", args.villager_root)
        if args.server_root:
            task.setdefault("server_root", args.server_root)
        if args.world_snapshot:
            task.setdefault("world_snapshot", args.world_snapshot)
        task.setdefault("server_port", args.server_port + index)
        task.setdefault("agent_port_base", args.agent_port_base + 10 * index)
        task_name = str(task.get("task_name") or task.get("task_id"))
        episode_id = f"{task_name}__qwen_eval"
        adapter = make_adapter(task, episode_id)
        actions = []
        score = {"evaluation_status": "unassessed", "success": None,
                 "real_rollout": True, "reason": "episode_interrupted"}
        try:
            adapter.reset(task)
            horizon = args.max_turns or int(task.get("deadline_ticks", 128))
            for turn in range(horizon):
                obs = {agent: adapter.observe(agent) for agent in ("agent_0", "agent_1")}
                joint = {}
                for agent in ("agent_0", "agent_1"):
                    joint[agent] = policy.sample_action(obs[agent], agent)
                actions.append({"turn": turn, "actions": joint})
                adapter.step_batch(joint)
                if any(action.get("type") == "submit" for action in joint.values()):
                    break
            score = adapter.score()
        finally:
            adapter.close()
        records.append({"task_id": task.get("task_id"), "task_idx": task.get("task_idx"),
                        "task_name": task_name, "model": args.model,
                        "checkpoint": args.checkpoint, "actions": actions,
                        "score": score})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                              for row in records) + "\n")
    print(json.dumps({"output": str(out), "episodes": len(records),
                      "successes": sum(bool(row["score"].get("success")) for row in records),
                      "split_audit": split_audit},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
