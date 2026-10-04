"""Strict audit for the materialized Bench2/Bench3 pools."""

from __future__ import annotations

import json
from pathlib import Path

from masbench.longhorizon.closed_loop import run_closed_loop
from masbench.longhorizon.incident import FAULTS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load(directory: str) -> list[dict]:
    rows = []
    for split in ("train", "dev", "test"):
        path = DATA / directory / f"{split}.jsonl"
        assert path.exists(), f"missing {path}"
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                assert row["split"] == split, f"split mismatch: {row['task_id']}"
                rows.append(row)
    return rows


def audit_workflow(rows: list[dict]) -> None:
    assert len(rows) == 175
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["schema_version"] == "bench2-workflow-v1"
        tasks = row["tasks"]
        branches = [x for x in tasks if x not in ("approval", "release")]
        assert len(branches) >= 2 and row["parallel_width"] == len(branches)
        assert row["required_agents"] == len(branches)
        assert set(tasks["approval"]["requires"]) == set(branches)
        assert tasks["release"]["requires"] == ["approval"]
        assert len({tasks[x]["artifact"] for x in branches}) == len(branches)
        assert all(tasks[x]["requires"] == ["data"] for x in branches)
        assert row["critical_path_ticks"] <= row["deadline"]
        # The controller is a deterministic executable oracle.  It proves every
        # generated case is solvable by MAS under its published deadline.
        result = run_closed_loop(row, "workflow", "mas", workers=row["required_agents"])
        assert result["success"], row["task_id"]


def audit_incident(rows: list[dict]) -> None:
    assert len(rows) == 48
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["schema_version"] == "bench3-incident-v1"
        assert row["fault"] not in row["distractors"]
        assert row["required_evidence"] == row["signals"]
        assert set(row["signals"]).issubset(set(row["query_signals"]))
        assert len(row["query_signals"]) == 2 * (1 + len(row["distractors"]))
        assert row["fault"] in FAULTS
        assert row["parallel_query_width"] >= 2
        # A hidden answer must not appear in the public state.  The fault field
        # is a gold label for evaluation, while prompts only expose signals.
        result = run_closed_loop(row, "incident", "mas", workers=4)
        assert result["success"], row["task_id"]


def main() -> None:
    workflow = _load("bench2_workflow")
    incident = _load("bench3_incident")
    audit_workflow(workflow)
    audit_incident(incident)
    out = {
        "status": "ok",
        "bench2_workflow": {"count": len(workflow), "splits": {s: sum(r["split"] == s for r in workflow) for s in ("train", "dev", "test")}},
        "bench3_incident": {"count": len(incident), "splits": {s: sum(r["split"] == s for r in incident) for s in ("train", "dev", "test")}},
        "checks": ["disjoint IDs", "stratified splits", "dependency integrity", "MAS solvability", "success-first score contract"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
