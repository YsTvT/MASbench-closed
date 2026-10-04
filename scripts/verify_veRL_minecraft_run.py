#!/usr/bin/env python3
"""Fail closed unless a real veRL Minecraft run produced a usable checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--test-records", required=True)
    parser.add_argument("--train-log", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    ckpt_files = list(result_dir.rglob("adapter_model.safetensors")) + list(result_dir.rglob("adapter_model.bin"))
    if not ckpt_files:
        raise SystemExit(f"no LoRA adapter checkpoint under {result_dir}")

    test_path = Path(args.test_records)
    records = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("test records are empty")
    for row in records:
        score = row.get("score", {})
        if score.get("evaluation_status") != "evaluated":
            raise SystemExit(f"test record is not checker-evaluated: {row.get('task_id')}")

    log = Path(args.train_log).read_text(errors="replace")
    bad = ("Traceback (most recent call last)", "ActorDiedError", "ConfigAttributeError")
    if any(marker in log for marker in bad):
        raise SystemExit("training log contains a fatal marker")
    print(json.dumps({"ok": True, "checkpoint": str(ckpt_files[0]),
                      "test_episodes": len(records),
                      "evaluated": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
