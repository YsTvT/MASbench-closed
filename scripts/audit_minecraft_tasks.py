import json
from pathlib import Path
from masbench.minecraft.tasks import load_tasks

root = Path(__file__).resolve().parents[1] / "data"
rows = load_tasks(root / "minecraft_tasks.jsonl")
assert len(rows) == 300
assert len({r["task_id"] for r in rows}) == len(rows)
assert {r["split"] for r in rows} == {"train", "dev", "test"}
assert {r["difficulty"] for r in rows} == {1, 2, 3}
assert {r["subfamily"] for r in rows} == {"construction", "cooking", "escape", "mas_hard"}
topology_splits = {}
for row in rows:
    topology_splits.setdefault(row["topology_id"], set()).add(row["split"])
assert all(len(splits) == 1 for splits in topology_splits.values()), "topology leakage across splits"
expected = {"construction": 100, "cooking": 100, "escape": 25, "mas_hard": 75}
assert {f: sum(r["subfamily"] == f for r in rows) for f in expected} == expected
for family in expected:
    assert {r["difficulty"] for r in rows if r["subfamily"] == family} == {1, 2, 3}
    family_rows = [r for r in rows if r["subfamily"] == family]
    serial_over = sum(r["serial_estimated_ticks"] > r["deadline_ticks"] for r in family_rows) / len(family_rows)
    assert 0.10 <= serial_over <= 0.40, "deadline does not expose a calibrated serial/MAS window for %s" % family
    assert all(r["critical_path_ticks"] <= r["deadline_ticks"] for r in family_rows)
for row in rows:
    graph = {s["id"]: set(s["depends_on"]) for s in row["subtasks"]}
    grouped = [node for group in row["parallel_groups"] for node in group]
    assert len(grouped) == len(set(grouped)) == len(graph)
    assert set(grouped) == set(graph)
    seen = set()
    while graph:
        ready = {node for node, deps in graph.items() if deps <= seen}
        assert ready, "dependency cycle in %s" % row["task_id"]
        seen |= ready
        for node in ready:
            del graph[node]
family_splits = {f: {s: sum(r["subfamily"] == f and r["split"] == s for r in rows) for s in ("train", "dev", "test")} for f in expected}
plan_path = root / "minecraft_reference_plans.jsonl"
plans = [json.loads(line) for line in plan_path.open() if line.strip()]
assert len(plans) == len(rows) * 3
assert {p["controller"] for p in plans} == {"serial", "parallel", "handoff"}
assert {p["task_id"] for p in plans} == {r["task_id"] for r in rows}
assert all(p["actions"][-1]["type"] == "submit" for p in plans)
episode_path = root / "minecraft_episodes.jsonl"
episodes = [json.loads(line) for line in episode_path.open() if line.strip()]
assert len(episodes) == len(rows) * 3 * 2
assert {e["variant"] for e in episodes} == {"oracle", "stale_handoff"}
assert all(e["synthetic_oracle_replay"] is True for e in episodes)
assert sum(e["success"] for e in episodes) == len(rows) * 3
print(json.dumps({"status": "ok", "count": len(rows), "reference_plans": len(plans), "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}, "family_splits": family_splits}, sort_keys=True))
