#!/usr/bin/env python3
"""Build leakage-safe splits from the real VillagerAgent blueprints.

``env/build_judger.py`` selects one entry from
``data/building_blue_print.json`` by list index (``--idx``), renders that entry
into Minecraft, and scores the actual blocks.  Every output row below keeps
that blueprint and its source hashes; ``task_idx`` is the exact index passed
to the checker.  No target structure or success threshold is fabricated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REPOSITORY = "https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON source {path}: {exc}") from exc


def _unwrap_blueprints(value: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = next((value.get(k) for k in ("blueprints", "building_blue_print", "data", "items")
                        if isinstance(value.get(k), list)), None)
        if records is None:
            raise SystemExit(f"{source} has no blueprint list")
    else:
        raise SystemExit(f"{source} must contain a JSON list of blueprints")
    result: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("blocks"), list) or not record["blocks"]:
            raise SystemExit(f"blueprint index {idx} in {source} is not renderable (missing blocks)")
        if not isinstance(record.get("size"), (list, tuple)) or len(record["size"]) != 3:
            raise SystemExit(f"blueprint index {idx} in {source} has invalid size")
        result.append(record)
    if len(result) < 3:
        raise SystemExit(f"{source} contains only {len(result)} blueprints; need at least three for held-out splits")
    return result


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _description_for(descriptions: Any, idx: int) -> list[Any]:
    if not isinstance(descriptions, dict):
        return []
    value = descriptions.get(f"task_{idx}", [])
    return value if isinstance(value, list) else [value]


def _split_groups(records: list[dict[str, Any]], train_frac: float, val_frac: float) -> dict[str, str]:
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise SystemExit("train-frac and val-frac must be in (0,1) and sum to < 1")
    groups: dict[str, list[int]] = {}
    for row in records:
        groups.setdefault(row["blueprint_sha256"], []).append(int(row["task_idx"]))
    ordered = sorted(groups.values(), key=lambda indices: min(indices))
    n = len(ordered)
    train_end = max(1, min(n - 2, int(n * train_frac)))
    val_end = max(train_end + 1, min(n - 1, train_end + int(n * val_frac)))
    result: dict[str, str] = {}
    for position, indices in enumerate(ordered):
        split = "train" if position < train_end else "val" if position < val_end else "test"
        for idx in indices:
            result[str(idx)] = split
    return result


def _safe_name(name: Any, idx: int) -> str:
    text = re.sub(r"[^a-z0-9._-]+", "_", str(name or f"blueprint_{idx}").strip().lower()).strip("._-")
    return text or f"blueprint_{idx}"


def build_rows(*, villager_root: Path, blueprint_path: Path, description_path: Path | None,
               repository: str, train_frac: float, val_frac: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_blueprints = blueprint_path.read_bytes()
    blueprints = _unwrap_blueprints(json.loads(raw_blueprints), blueprint_path)
    blueprint_file_hash = hashlib.sha256(raw_blueprints).hexdigest()
    descriptions = _load_json(description_path) if description_path and description_path.is_file() else {}
    checker_path = villager_root / "env" / "build_judger.py"
    if not checker_path.is_file():
        raise SystemExit(f"real checker is missing: {checker_path}")
    checker_text = checker_path.read_text(encoding="utf-8", errors="replace")
    if "blue_prints[select_idx]" not in checker_text:
        raise SystemExit(f"{checker_path} is not the expected checker (missing blue_prints[select_idx])")
    checker_hash = _sha256(checker_path)
    rows: list[dict[str, Any]] = []
    for idx, blueprint in enumerate(blueprints):
        blueprint_hash = hashlib.sha256(_canonical(blueprint).encode("utf-8")).hexdigest()
        name = blueprint.get("name") or f"blueprint_{idx}"
        # ``name`` is the only natural-language task label in the upstream
        # source. Keep it verbatim; do not synthesize material counts or a
        # success predicate from the geometry.
        source_goal = name if isinstance(name, str) else None
        rows.append({
            "task_id": f"villageragent_blueprint_{idx:04d}_{_safe_name(name, idx)}",
            "task_idx": idx,
            "task_name": f"minecraft_real_blueprint_{idx:04d}",
            "split": "unassigned",
            "family": "minecraft",
            "subfamily": "construction",
            "source_alignment": "VillagerAgent-Minecraft-multiagent-framework",
            "real_backend": "villageragent_mineflayer",
            "real_world_reset": "villageragent_build_judger_render",
            "goal": source_goal,
            "action_space": ["observe", "move", "place", "break", "handoff", "submit"],
            "agent_roles": [
                {"agent_id": "agent_0", "role": "builder", "capabilities": ["observe", "move", "place", "break", "message"]},
                {"agent_id": "agent_1", "role": "builder", "capabilities": ["observe", "move", "place", "break", "message"]},
            ],
            "required_agents": 2,
            "checker": {"entrypoint": "env/build_judger.py", "argv": ["--idx", idx],
                        "metrics": ["block_hit_rate", "view_hit_rate", "efficiency", "complexity"],
                        "score_authority": "VillagerAgent build_judger.py"},
            "blueprint": copy.deepcopy(blueprint),
            "blueprint_description": _description_for(descriptions, idx),
            "metadata": {
                "blueprint_name": name,
                "blueprint_index": idx,
                "blueprint_size": list(blueprint["size"]),
                "block_count": len(blueprint["blocks"]),
                "source_path": str(blueprint_path.resolve()),
                "source_sha256": blueprint_file_hash,
                "record_sha256": blueprint_hash,
                "checker_path": str(checker_path.resolve()),
                "checker_sha256": checker_hash,
                "repository": repository,
                "goal_source": "data/building_blue_print.json:name",
            },
            "blueprint_sha256": blueprint_hash,
        })
    split_by_idx = _split_groups(rows, train_frac, val_frac)
    for row in rows:
        row["split"] = split_by_idx[str(row["task_idx"])]
        row["task_name"] = f"minecraft_real_blueprint_{row['task_idx']:04d}_{row['split']}"
        row.pop("blueprint_sha256", None)
    assert len({row["task_idx"] for row in rows}) == len(rows)
    manifest = {
        "source_repository": repository,
        "source_root": str(villager_root.resolve()),
        "source_revision": _git_revision(villager_root),
        "blueprint_path": str(blueprint_path.resolve()),
        "blueprint_sha256": blueprint_file_hash,
        "checker_path": str(checker_path.resolve()),
        "checker_sha256": checker_hash,
        "description_path": str(description_path.resolve()) if description_path and description_path.is_file() else None,
        "description_sha256": _sha256(description_path) if description_path and description_path.is_file() else None,
        "split_policy": {"group_key": "canonical blueprint SHA-256",
                         "task_idx_source": "list index in data/building_blue_print.json",
                         "train_fraction": train_frac, "val_fraction": val_frac,
                         "test_fraction": 1.0 - train_frac - val_frac},
        "counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "task_idx_by_split": {split: [row["task_idx"] for row in rows if row["split"] == split]
                              for split in ("train", "val", "test")},
    }
    return rows, manifest


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--villager-root", "--source-root", required=True, type=Path)
    parser.add_argument("--blueprint-file", type=Path)
    parser.add_argument("--description-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    root = args.villager_root.resolve()
    blueprint_path = (args.blueprint_file or root / "data" / "building_blue_print.json").resolve()
    description_path = (args.description_file or root / "data" / "blueprint_description_all.json").resolve()
    if not blueprint_path.is_file():
        raise SystemExit(f"real blueprint source is missing: {blueprint_path}")
    rows, manifest = build_rows(villager_root=root, blueprint_path=blueprint_path,
                                description_path=description_path, repository=args.repository,
                                train_frac=args.train_frac, val_frac=args.val_frac)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        _write_jsonl(args.output_dir / f"{split}.jsonl", (row for row in rows if row["split"] == split))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
