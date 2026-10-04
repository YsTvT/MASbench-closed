#!/usr/bin/env python3
"""Run a frozen closed-model prompt baseline on real Minecraft.

This is intentionally separate from ``train_real_rl.py``.  It keeps the same
VillagerAgent/Mineflayer adapter, world reset and checker, but replaces the
trainable Qwen policy with one API decision per agent and environment turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masbench.controllers import CachedModelAdapter, JsonContractAdapter
from masbench.metrics import success_first
from masbench.minecraft.prompting import PromptMinecraftRunner, write_episode
from masbench.minecraft.rl import load_factory
from masbench.minecraft.splits import SplitLeakageError, audit_split_files
from masbench.minecraft.tasks import load_tasks


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run(args):
    task_path = Path(args.tasks_file) if args.tasks_file else Path(args.data_dir) / "minecraft_tasks.jsonl"
    split_audit = None
    if args.split == "test":
        # A held-out real run must name the training file that was used to
        # select the controller.  Derive the conventional sibling for the
        # pilot layout, but fail before API/Minecraft work if it is absent.
        train_path = (Path(args.train_tasks_file) if args.train_tasks_file
                      else task_path.with_name("train.jsonl"))
        if not train_path.is_file():
            raise SystemExit(
                "test evaluation requires --train-tasks-file (or a train.jsonl "
                f"sibling of {task_path}); refusing an unaudited test run"
            )
        try:
            split_audit = audit_split_files(train_path, task_path)
        except SplitLeakageError as exc:
            raise SystemExit(f"MINECRAFT SPLIT AUDIT FAILED: {exc}") from exc
    tasks = load_tasks(task_path, split=args.split)
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in wanted]
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise SystemExit(f"no Minecraft tasks found for split={args.split!r}")
    if not args.adapter_factory or not args.model_factory:
        raise SystemExit("closed-model evaluation requires --adapter-factory and --model-factory")

    adapter_factory = load_factory(args.adapter_factory)
    raw_model = load_factory(args.model_factory)()
    model_metadata = dict(getattr(raw_model, "metadata", {}) or {})
    model_metadata.update({
        "model_label": args.model_label,
        "model_version": args.model_version,
        "temperature": args.temperature if args.temperature is not None else model_metadata.get("temperature"),
        "max_tokens": args.max_tokens if args.max_tokens is not None else model_metadata.get("max_tokens"),
        "n": 1,
        "prompt_based": True,
        "weights_updated": False,
        "controller_trained": bool(args.controller_trained),
        "test_frozen": args.split == "test",
    })
    model = JsonContractAdapter(raw_model)
    if args.cache_dir:
        model = CachedModelAdapter(model, args.cache_dir)
    runner = PromptMinecraftRunner(
        adapter_factory,
        model,
        max_ticks=args.max_ticks,
        max_history=args.max_history,
        prompt_version=args.prompt_version,
        model_metadata=model_metadata,
    )

    # The real VillagerAgent adapter temporarily changes cwd to its vendor
    # checkout while invoking tools. Resolve output before the first episode so
    # traces and score files always land in the requested MASbench directory.
    out = Path(args.out).resolve()
    rows = []
    episodes = 0
    for task in tasks:
        for repeat in range(args.repeats):
            episode_id = f"{task['task_id']}__closed_prompt__{repeat:02d}"
            episode = runner.run_episode(task, episode_id)
            write_episode(out, episode)
            rows.append(episode["score"])
            episodes += 1

    summary = success_first(rows)
    summary.update({
        "schema_version": "masbench.minecraft.prompt_eval.v1",
        "real_rollout": True,
        "model_type": "closed_api",
        "weights_updated": False,
        "prompt_version": args.prompt_version,
        "split": args.split,
        "tasks": len(tasks),
        "episodes": episodes,
        "repeats": args.repeats,
        "split_audit": split_audit,
    })
    _write_json(out / "summary.json", {"summary": summary, "rows": rows})
    _write_json(out / "manifest.json", {
        "schema_version": "masbench.minecraft.prompt_manifest.v1",
        "mode": "closed_api_prompt_eval",
        "real_rollout": True,
        "model": model_metadata,
        "prompt_version": args.prompt_version,
        "split": args.split,
        "task_count": len(tasks),
        "episodes": episodes,
        "split_audit": split_audit,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="MASbench/data")
    parser.add_argument("--tasks-file", default=None,
                        help="optional JSONL task file, e.g. data/minecraft_real_pilot/train.jsonl")
    parser.add_argument("--train-tasks-file", default=None,
                        help="training JSONL used for test leakage audit; required for split=test")
    parser.add_argument("--out", default="MASbench/results/minecraft_closed_prompt")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--prompt-version", default="minecraft-villageragent-typed-prompt-v1")
    parser.add_argument("--model-label", default="closed-api")
    parser.add_argument("--model-version", default="unspecified")
    parser.add_argument("--controller-trained", action="store_true",
                        help="mark that the prompt/controller was selected on a separate train split")
    parser.add_argument("--temperature", type=float, default=None,
                        help="metadata only; the injected model factory owns the API setting")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="metadata only; the injected model factory owns the API setting")
    parser.add_argument("--cache-dir", default=None,
                        help="optional deterministic response cache; use a fresh cache per prompt version")
    parser.add_argument("--adapter-factory", help="module:factory(task, episode_id) -> real adapter")
    parser.add_argument("--model-factory", help="module:factory() -> ModelAdapter")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
