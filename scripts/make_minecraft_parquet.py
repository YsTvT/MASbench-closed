#!/usr/bin/env python3
"""Convert the real VillagerAgent task JSONL into veRL RL parquet files.

veRL's RLHFDataset reads a parquet row with a chat ``prompt`` and carries the
``env_kwargs`` object through the multi-turn collector.  The object below is
the complete real task, so the Minecraft worker never falls back to a symbolic
task or an implicit task index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_row(task: dict, *, split: str, index: int, server_root: str,
              snapshot_root: str, villager_root: str,
              server_port_base: int = 25600, agent_port_base: int = 6000) -> dict:
    prompt = (
        "You are controlling two cooperating Minecraft builders in the real "
        "VillagerAgent environment. Follow the goal and emit exactly one JSON "
        "object mapping agent_0 and agent_1 to executable typed actions. "
        "Start with observe, then use move/gather/place/craft/trade and submit "
        "only after the construction checker can verify the goal.\n\n"
        f"Goal: {task.get('goal', '')}\n"
        f"Agent roles: {json.dumps(task.get('agent_roles', []), ensure_ascii=False)}\n"
        f"Required checks: {json.dumps(task.get('success_checks', []), ensure_ascii=False)}"
    )
    # Keep the task payload in env_kwargs.  The real bridge consumes task_idx,
    # server_root and world_snapshot from this object on every reset.
    env_kwargs = dict(task)
    env_kwargs.update({
        "split": split,
        "base_server_root": server_root,
        "source_world_snapshot": snapshot_root,
        "villageragent_root": villager_root,
        "server_port_base": int(server_port_base),
        "agent_port_base": int(agent_port_base),
        "real_backend": "villageragent_mineflayer",
    })
    return {
        "data_source": "minecraft_real_villageragent",
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"ground_truth": "1", "style": "real_checker"},
        "ability": "minecraft",
        # JSON keeps heterogeneous task schemas representable in one parquet
        # column; the worker decodes this string before every reset.
        "env_kwargs": json.dumps(env_kwargs, ensure_ascii=False, sort_keys=True),
        "extra_info": {
            "index": index,
            "task_id": task.get("task_id"),
            "task_idx": task.get("task_idx"),
            "split": split,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--villager-root", required=True)
    parser.add_argument("--server-port-base", type=int, default=25600)
    parser.add_argument("--agent-port-base", type=int, default=6000)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no tasks in {args.input}")
    if any(row.get("task_idx") is None for row in rows):
        raise SystemExit("every real task must contain task_idx")
    records = [build_row(row, split=args.split, index=i,
                         server_root=args.server_root,
                         snapshot_root=args.snapshot_root,
                         villager_root=args.villager_root,
                         server_port_base=args.server_port_base,
                         agent_port_base=args.agent_port_base)
               for i, row in enumerate(rows)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out)
    print(json.dumps({"output": str(out), "rows": len(records),
                      "columns": table.column_names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
