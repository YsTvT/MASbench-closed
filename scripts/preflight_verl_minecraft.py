#!/usr/bin/env python3
"""Fail-fast checks before starting a real veRL Minecraft run."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="MASbench/data/minecraft_real_pilot/train.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.task_file).read_text().splitlines() if line.strip()]
    assert rows and all(row.get("task_idx") is not None for row in rows), "pilot tasks require task_idx"
    assert all(row.get("split") == "train" for row in rows), "veRL pilot must use train split"
    env_manager = importlib.import_module("agent_system.environments.env_manager")
    assert getattr(env_manager.make_envs, "__name__", "") == "_make_envs", \
        "veRL Minecraft sitecustomize hook is not active"
    policy = importlib.import_module("masbench.minecraft.qwen_lora_policy")
    assert hasattr(policy, "QwenLoRAPolicy")
    adapter = importlib.import_module("masbench.minecraft.villageragent_real")
    assert adapter.VillagerAgentRealAdapter.real_rollout is True
    print(json.dumps({"ok": True, "tasks": len(rows), "task_idx": sorted({r["task_idx"] for r in rows}),
                      "algorithm": "grpo", "group_size": 4, "policy": "qwen2.5-7b-lora"}, indent=2))


if __name__ == "__main__":
    main()

