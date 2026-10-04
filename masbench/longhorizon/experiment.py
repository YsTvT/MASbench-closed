"""Reproducible Bench2/3 experiment harness.

The harness fixes the model-independent part of the comparison: worker count,
action budget, message budget, seed list, deadline scale and deterministic
checker. It emits JSONL with one row per task and a Markdown summary.
"""

import argparse
import copy
import json
from pathlib import Path

from .core import summarize, paired_speedup
from .workflow import WorkflowEnv, make_workflow
from .incident import IncidentEnv, make_incident, FAULTS
from .closed_loop import run_closed_loop, _json_default


def _tick_until(env, ticks):
    env.advance(max(1, ticks))


def run_workflow(case, method, workers=4):
    return run_closed_loop(case, "workflow", method, workers=workers)


def run_incident(case, method, workers=4):
    return run_closed_loop(case, "incident", method, workers=workers)


def make_cases(seeds=30):
    for seed in range(seeds):
        for difficulty in (1, 2, 3):
            yield make_workflow(seed, difficulty)
            yield make_incident(seed, difficulty)


def load_cases(data_dir, splits=("test",)):
    data_dir = Path(data_dir)
    for split in splits:
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        name = "longhorizon_%s.jsonl" % split
        with (data_dir / name).open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def run_suite(out_dir, seeds=30, deadline_scale=1.0, data_dir=None, splits=("test",)):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = load_cases(data_dir, splits=splits) if data_dir else make_cases(seeds)
    for original in cases:
        for method in ("serial", "mas"):
            case = copy.deepcopy(original)
            case["deadline"] = max(1, int(round(case["deadline"] * deadline_scale)))
            runner = run_workflow if original["task_id"].startswith("workflow") else run_incident
            rows.append(runner(case, method))
    with (out_dir / "runs.jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    grouped = {}
    for family in ("workflow", "incident"):
        for method in ("serial", "mas"):
            group = [r for r in rows if r["family"] == family and r["method"] == method]
            grouped[family + "/" + method] = summarize(group)
        serial = [r for r in rows if r["family"] == family and r["method"] == "serial"]
        mas = [r for r in rows if r["family"] == family and r["method"] == "mas"]
        grouped[family + "/paired"] = paired_speedup(serial, mas)
    report = {"seeds": seeds, "data_dir": str(data_dir) if data_dir else None, "splits": list(splits), "deadline_scale": deadline_scale, "groups": grouped,
              "protocol": "success-first; time is conditioned on successful runs"}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# MASbench long-horizon experiment", "", "| Group | SR | Composite | Successful mean ticks | Successful median ticks |", "|---|---:|---:|---:|---:|"]
    for name, value in grouped.items():
        if name.endswith("/paired"):
            continue
        lines.append("| %s | %.3f | %.3f | %s | %s |" % (name, value["sr"], value["composite_score"], value["successful_mean_ticks"], value["successful_median_ticks"]))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="results/longhorizon"); p.add_argument("--seeds", type=int, default=30); p.add_argument("--deadline-scale", type=float, default=1.0); p.add_argument("--data-dir", default=None); p.add_argument("--split", nargs="+", choices=("train", "dev", "test"), default=["test"])
    args = p.parse_args()
    print(json.dumps(run_suite(args.out, args.seeds, args.deadline_scale, args.data_dir, tuple(args.split)), indent=2, sort_keys=True))
