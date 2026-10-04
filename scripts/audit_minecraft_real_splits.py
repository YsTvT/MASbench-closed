#!/usr/bin/env python3
"""Audit real Minecraft split files before training or evaluation.

This command is intentionally strict.  It is safe to run on a pilot file,
but it exits non-zero when a task appears in more than one split or when the
two model evaluators do not receive the same held-out task set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masbench.minecraft.splits import (SplitLeakageError, assert_same_eval_set,
                                       audit_split_files, load_task_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--open-test", default=None,
                        help="optional second evaluator's test JSONL")
    parser.add_argument("--closed-test", default=None,
                        help="optional closed evaluator's test JSONL")
    args = parser.parse_args()
    try:
        report = audit_split_files(args.train, args.test, val_path=args.val)
        if bool(args.open_test) != bool(args.closed_test):
            raise SplitLeakageError("--open-test and --closed-test must be supplied together")
        if args.open_test:
            open_rows = load_task_rows(args.open_test)
            closed_rows = load_task_rows(args.closed_test)
            report["same_eval_set"] = assert_same_eval_set(open_rows, closed_rows)
        print(json.dumps({"ok": True, **report}, ensure_ascii=False, sort_keys=True))
    except SplitLeakageError as exc:
        raise SystemExit(f"MINECRAFT SPLIT AUDIT FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
