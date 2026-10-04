import argparse
import json
from pathlib import Path

from masbench.metrics import success_first
from masbench.minecraft.env import run_reference
from masbench.minecraft.tasks import load_tasks


def run(data_dir, out_dir):
    tasks = load_tasks(Path(data_dir) / "minecraft_tasks.jsonl")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        for method in ("serial", "mas"):
            score = run_reference(task, "serial" if method == "serial" else "parallel")
            score.update({"task_id": task["task_id"], "family": task["subfamily"],
                          "method": method, "split": task["split"], "topology_id": task["topology_id"],
                          "deadline": task["deadline_ticks"], "backend": "symbolic"})
            rows.append(score)
    with (out_dir / "runs.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    groups = {}
    for family in sorted({t["subfamily"] for t in tasks}):
        for method in ("serial", "mas"):
            groups[family + "/" + method] = success_first(
                r for r in rows if r["family"] == family and r["method"] == method)
    report = {"backend": "symbolic", "tasks": len(tasks), "runs": len(rows), "groups": groups,
              "protocol": "symbolic sanity only; real Minecraft execution remains an experiment"}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Minecraft symbolic sanity run", "", "| Group | SR | Composite | Successful mean ticks |", "|---|---:|---:|---:|"]
    for name, value in groups.items():
        lines.append("| %s | %.3f | %.3f | %s |" % (name, value["sr"], value["composite_score"], value["successful_mean_ticks"]))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="MASbench/data")
    parser.add_argument("--out", default="MASbench/results/minecraft_symbolic")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.out), indent=2, sort_keys=True))
