"""Materialize the Bench2/Bench3 data contract into separate benchmark pools.

The original ``longhorizon_*.jsonl`` files are kept for backward compatibility.
This generator writes one directory per benchmark so an evaluator can freeze a
single benchmark without filtering a mixed file.  Cases remain generated from
the same deterministic constructors used by the environments; no labels are
sampled or inferred from a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from masbench.longhorizon.incident import FAULTS, make_incident
from masbench.longhorizon.workflow import CATALOG, make_workflow


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _split(seed: int, total: int) -> str:
    # A fixed, stratified policy.  Keeping the rule here makes regeneration
    # byte-for-byte reproducible and prevents accidental train/test leakage.
    train_end = int(total * 0.60)
    dev_end = train_end + int(total * 0.15)
    return "train" if seed < train_end else "dev" if seed < dev_end else "test"


def _workflow_row(seed: int) -> dict:
    difficulty = seed % 3 + 1
    row = make_workflow(seed, difficulty)
    branches = [name for name in row["tasks"] if name not in ("approval", "release")]
    branch_durations = {
        name: CATALOG[row["tasks"][name]["kind"] if row["tasks"][name]["kind"] in CATALOG else "report"]["duration"]
        for name in branches
    }
    scenarios = ("quarterly finance close", "regional inventory refresh", "customer health review",
                 "privacy export readiness", "capacity planning update", "release risk review")
    row.update({
        "schema_version": "bench2-workflow-v1",
        "split": _split(seed, 175),
        "scenario": scenarios[seed % len(scenarios)],
        "parallel_width": len(branches),
        "branch_durations": branch_durations,
        "critical_path_ticks": max(branch_durations.values()) + CATALOG["approval"]["duration"] + CATALOG["release"]["duration"],
        "required_agents": len(branches),
        "success_contract": ["every branch artifact", "approval artifact", "release artifact"],
        "generator_version": "bench23-v1",
    })
    return row


def _incident_row(seed: int) -> dict:
    difficulty = seed % 3 + 1
    row = make_incident(seed, difficulty)
    services = {
        "cache_miss": ("catalog-api", "p95 latency", "cache configuration drift"),
        "db_pool": ("checkout-api", "5xx error rate", "database connection pool exhaustion"),
        "queue_lag": ("notification-worker", "queue age", "worker capacity mismatch"),
    }
    service, primary_metric, causal_summary = services[row["fault"]]
    row.update({
        "schema_version": "bench3-incident-v1",
        "split": _split(seed, 48),
        "service": service,
        "severity": ("SEV-3", "SEV-2", "SEV-1")[difficulty - 1],
        "primary_metric": primary_metric,
        "causal_summary_hidden": causal_summary,
        "query_budget": len(row["query_signals"]),
        "required_evidence": list(row["signals"]),
        "parallel_query_width": min(4, len(row["query_signals"])),
        "success_contract": ["correct root cause", "correct mitigation", "post-mitigation verification", "operator communication"],
        "generator_version": "bench23-v1",
    })
    return row


def _write_family(name: str, rows: list[dict], total: int) -> dict:
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = out / f"{split}.jsonl"
        subset = [r for r in rows if r["split"] == split]
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in subset))
    manifest = {
        "schema_version": "bench23-manifest-v1",
        "benchmark": name,
        "count": len(rows),
        "expected_count": total,
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")},
        "task_ids": [r["task_id"] for r in rows],
        "protocol": "success-first; completion time is conditioned on success",
        "generator_version": "bench23-v1",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {k: manifest[k] for k in ("benchmark", "count", "expected_count", "splits")}


def main() -> None:
    workflow = [_workflow_row(seed) for seed in range(175)]
    incident = [_incident_row(seed) for seed in range(48)]
    summary = {
        "bench2": _write_family("bench2_workflow", workflow, 175),
        "bench3": _write_family("bench3_incident", incident, 48),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
