"""Run deterministic Bench2/Bench3 baselines on materialized data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from masbench.longhorizon.closed_loop import _json_default, run_closed_loop
from masbench.longhorizon.core import paired_speedup, summarize


ROOT = Path(__file__).resolve().parents[1]
FAMILY_DIR = {"bench2": "bench2_workflow", "bench3": "bench3_incident"}
FAMILY_NAME = {"bench2": "workflow", "bench3": "incident"}


def load_cases(bench: str, split: str) -> list[dict]:
    path = ROOT / "data" / FAMILY_DIR[bench] / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(bench: str, split: str, out_dir: Path, deadline_scale: float, workers: int) -> dict:
    family = FAMILY_NAME[bench]
    rows = []
    for original in load_cases(bench, split):
        for method in ("serial", "mas"):
            case = copy.deepcopy(original)
            case["deadline"] = max(1, int(round(original["deadline"] * deadline_scale)))
            rows.append(run_closed_loop(case, family, method, workers=workers))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=_json_default) + "\n" for r in rows))
    serial = [r for r in rows if r["method"] == "serial"]
    mas = [r for r in rows if r["method"] == "mas"]
    report = {
        "benchmark": bench,
        "family": family,
        "split": split,
        "deadline_scale": deadline_scale,
        "workers": workers,
        "protocol": "success-first; failed-run time is null and excluded",
        "groups": {"serial": summarize(serial), "mas": summarize(mas), "paired": paired_speedup(serial, mas)},
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=("bench2", "bench3", "all"), default="all")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--out", default="MASbench/results/bench23")
    parser.add_argument("--deadline-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    benches = ("bench2", "bench3") if args.bench == "all" else (args.bench,)
    reports = {}
    for bench in benches:
        reports[bench] = run(bench, args.split, Path(args.out) / bench / args.split, args.deadline_scale, args.workers)
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
