#!/usr/bin/env python3
"""Evaluate persisted real-Minecraft RL scores.

Only checker-evaluated, real-world records are accepted.  This command never
turns a missing checker result into SR=0 and never consumes symbolic replays.
Actual server execution is performed by the train runner's injected adapter;
this CLI is the immutable test/eval aggregation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masbench.metrics import success_first
from masbench.minecraft.rl import load_score_rows


def run(scores: str, out: str):
    rows = load_score_rows(Path(scores), require_real=True)
    summary = success_first(rows)
    summary.update({"schema_version": "masbench.minecraft.rl_eval.v1",
                    "real_rollout": True, "score_files": len(rows)})
    payload = {"summary": summary, "rows": rows}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", help="directory containing real score.json files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.scores, args.out), indent=2, sort_keys=True))
