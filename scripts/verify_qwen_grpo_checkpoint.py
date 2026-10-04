#!/usr/bin/env python3
"""Verify that a veRL Qwen LoRA run produced a usable actor checkpoint.

The official veRL layout has changed between releases (for example,
``global_step_1/actor/huggingface`` versus ``actor/huggingface``), so this
check searches below the run directory instead of hard-coding one layout.
It deliberately requires PEFT adapter metadata and adapter weights: a folder
containing only the base Qwen model is not a successful GRPO checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)


def _adapter_candidates(root: Path) -> list[Path]:
    if root.is_file():
        return []
    paths = []
    for config in root.rglob("adapter_config.json"):
        if config.is_file() and not any(part.startswith(".") for part in config.parts):
            paths.append(config.parent)
    return sorted(paths)


def _weight_for(directory: Path) -> Path | None:
    for name in _WEIGHT_NAMES:
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    # Some PEFT save hooks shard adapter weights.  Accept a complete set of
    # non-empty shards while keeping the check specific to adapter files.
    shards = sorted(directory.glob("adapter_model-*.safetensors"))
    if shards and all(path.stat().st_size > 0 for path in shards):
        return shards[0]
    return None


def verify_checkpoint(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {root}")
    candidates = _adapter_candidates(root)
    if not candidates:
        raise ValueError(f"no adapter_config.json found below {root}")
    failures = []
    for directory in candidates:
        config_path = directory / "adapter_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{config_path}: invalid JSON ({exc})")
            continue
        if not isinstance(config, dict) or config.get("peft_type") not in {None, "LORA", "IA3", "ADALORA"}:
            failures.append(f"{config_path}: unsupported PEFT config")
            continue
        weight = _weight_for(directory)
        if weight is None:
            failures.append(f"{directory}: adapter weights are missing or empty")
            continue
        relative = directory.relative_to(root)
        step = next((part for part in relative.parts if part.startswith("global_step_")), None)
        return {
            "ok": True,
            "checkpoint_root": str(root),
            "adapter_dir": str(directory),
            "adapter_config": str(config_path),
            "adapter_weight": str(weight),
            "global_step": step,
            "base_model_name_or_path": config.get("base_model_name_or_path"),
        }
    details = "; ".join(failures) if failures else "no complete adapter candidate"
    raise ValueError(details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify_checkpoint(args.checkpoint)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
