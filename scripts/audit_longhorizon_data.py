"""Audit expanded Bench2/3 JSONL data before a paper run."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data"
rows = []
by_split = {}
for name in ("longhorizon_train.jsonl", "longhorizon_dev.jsonl", "longhorizon_test.jsonl"):
    split_name = name.rsplit("_", 1)[-1].split(".", 1)[0]
    with (root / name).open() as f:
        split_rows = [json.loads(line) for line in f if line.strip()]
    by_split[split_name] = split_rows
    rows.extend(split_rows)
assert rows, "empty dataset"
ids = [r["task_id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate task_id"
assert {r["split"] for r in rows} == {"train", "dev", "test"}
for split, split_rows in by_split.items():
    assert all(r["split"] == split for r in split_rows), "split/file mismatch"
assert {r["family"] for r in rows} == {"workflow", "incident"}
for row in rows:
    assert row["difficulty"] in (1, 2, 3)
    assert row["deadline"] > 0
    assert row.get("generator_version") == "longhorizon-v2-stratified"
    if row["family"] == "workflow":
        tasks = row["tasks"]
        assert "approval" in tasks and "release" in tasks
        branches = [k for k in tasks if k not in ("approval", "release")]
        assert len(branches) >= 2
        assert set(tasks["approval"]["requires"]) == set(branches)
        assert tasks["release"]["requires"] == ["approval"]
        assert len({tasks[k]["artifact"] for k in branches}) == len(branches)
        assert all(tasks[k]["requires"] == ["data"] for k in branches)
        assert row["required"] == ["release"]
    else:
        assert len(row["signals"]) >= 2
        assert set(row["signals"]).issubset(set(row["query_signals"]))
        assert len(row["query_signals"]) == 2 * (1 + len(row["distractors"]))
        assert row["fault"] not in row["distractors"]
        assert len(set(row["signals"])) == len(row["signals"])
        assert set(row["signals"]) == set({
            "cache_miss": ["latency_high", "cache_hit_low"],
            "db_pool": ["error_rate_high", "db_pool_exhausted"],
            "queue_lag": ["queue_lag", "worker_cpu_low"],
        }[row["fault"]])
        assert row["required"] == ["root_cause", "mitigated", "verified"]
families = {f: sum(r["family"] == f for r in rows) for f in ("workflow", "incident")}
splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
family_splits = {s: {f: sum(r["split"] == s and r["family"] == f for r in rows)
                     for f in ("workflow", "incident")} for s in splits}
for split_rows in by_split.values():
    assert {r["difficulty"] for r in split_rows} == {1, 2, 3}, "difficulty not represented in split"
print(json.dumps({"status": "ok", "count": len(rows), "families": families,
                  "splits": splits, "family_splits": family_splits}, sort_keys=True))
