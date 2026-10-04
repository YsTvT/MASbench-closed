#!/usr/bin/env python3
"""Aggregate persisted real-rollout score records.

This command deliberately excludes records whose checker did not run.  Such
records remain visible through ``unassessed_runs`` and never become SR=0.
"""
import argparse
import json
from pathlib import Path

from masbench.metrics import success_first


def load_scores(root):
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("score.json"))
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload.setdefault("source_file", str(path))
        status = payload.get("evaluation_status", "evaluated")
        if status not in {"evaluated", "unassessed"}:
            raise ValueError(f"unknown evaluation_status={status!r}: {path}")
        if status == "evaluated" and not isinstance(payload.get("success"), bool):
            raise ValueError(f"evaluated score requires boolean success: {path}")
        if payload.get("success") and payload.get("completion_ticks") is None:
            raise ValueError(f"successful score requires completion_ticks: {path}")
        if not payload.get("success") and payload.get("completion_ticks") is not None:
            raise ValueError(f"failed/unassessed score must omit completion_ticks: {path}")
        rows.append(payload)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", help="A score.json file or directory containing score.json files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = load_scores(args.scores)
    summary = success_first(rows)
    summary.update({"schema_version": "masbench.eval.v1", "score_files": len(rows)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
