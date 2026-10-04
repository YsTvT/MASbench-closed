"""Run every no-GPU deterministic audit/sanity check for the three benches."""

import json
from pathlib import Path

from masbench.longhorizon.closed_loop import cases_from_jsonl, run_closed_loop_suite
from masbench.minecraft.tasks import load_tasks
from masbench.minecraft.env import run_reference
from masbench.metrics import success_first


root = Path("MASbench")
data = root / "data"
report = {"status": "ok", "backends": {}}

long_cases = list(cases_from_jsonl(data, "test"))
long_out = root / "results" / "longhorizon" / "closed_loop_test_1_0"
report["backends"]["workflow_incident_symbolic"] = run_closed_loop_suite(long_cases, long_out, 1.0)

mc_cases = load_tasks(data / "minecraft_tasks.jsonl", split="test")
mc_rows = []
for task in mc_cases:
    for method in ("serial", "mas"):
        score = run_reference(task, "serial" if method == "serial" else "parallel")
        score.update({"task_id": task["task_id"], "family": task["subfamily"],
                      "method": method, "deadline": task["deadline_ticks"]})
        mc_rows.append(score)
groups = {}
for family in sorted({t["subfamily"] for t in mc_cases}):
    for method in ("serial", "mas"):
        groups[family + "/" + method] = success_first(
            r for r in mc_rows if r["family"] == family and r["method"] == method)
report["backends"]["minecraft_symbolic_test"] = {"tasks": len(mc_cases), "groups": groups}
(root / "results" / "sanity_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps({"status": "ok", "workflow_incident_test": len(long_cases), "minecraft_test": len(mc_cases)}, sort_keys=True))
