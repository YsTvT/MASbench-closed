"""Leakage checks shared by real Minecraft train and evaluation runners.

The real VillagerAgent pilot stores a task in JSONL and in a veRL parquet
row.  A split label alone is not sufficient protection: a copied task can
keep the same geometry while changing only ``split`` or ``task_name``.  This
module treats the task content, task id, blueprint index and world seed as
independent identity signals and fails closed when any of them overlap.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class SplitLeakageError(ValueError):
    """Raised when a train/evaluation split cannot be considered held out."""


# These fields label a record or select a runtime instance.  They are removed
# only for the content fingerprint; task_id/task_idx/world_seed are checked
# separately below, so changing one of them cannot hide a duplicate task.
_FINGERPRINT_IGNORED = frozenset({
    "split", "task_id", "task_name", "task_idx", "world_seed",
    "server_root", "base_server_root", "world_snapshot",
    "source_world_snapshot", "villageragent_root", "base_villageragent_root",
    "server_port", "server_port_base", "agent_port_base", "max_turns",
})


def _without_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _FINGERPRINT_IGNORED
        }
    if isinstance(value, (list, tuple)):
        return [_without_identity(item) for item in value]
    return value


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Return a stable hash of task semantics, excluding identity/runtime labels."""
    payload = json.dumps(_without_identity(deepcopy(dict(task))),
                         ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_task_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL split without silently accepting blank files."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise SplitLeakageError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    if not rows:
        raise SplitLeakageError(f"split file is empty: {path}")
    return rows


def _index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, set[Any]]:
    required = ("task_id", "task_idx")
    for row in rows:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise SplitLeakageError(
                f"{label} row {row.get('task_id', '<unknown>')} is missing "
                + ", ".join(missing)
            )
    index = {
        "task_id": {str(row["task_id"]) for row in rows},
        "task_idx": {row["task_idx"] for row in rows},
        "fingerprint": {task_fingerprint(row) for row in rows},
    }
    # Some real blueprint manifests do not materialize a world_seed; task_idx
    # and the semantic fingerprint remain authoritative in that format.
    if all(row.get("world_seed") is not None for row in rows):
        index["world_seed"] = {row["world_seed"] for row in rows}
    return index


def assert_disjoint(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
                    *, reference_label: str = "train", candidate_label: str = "test") -> dict[str, Any]:
    """Fail unless two splits are disjoint by every supported identity signal."""
    left = _index(reference, reference_label)
    right = _index(candidate, candidate_label)
    overlaps = {key: sorted(left[key] & right[key], key=str)
                for key in left if left[key] & right[key]}
    if overlaps:
        details = "; ".join(f"{key}={values}" for key, values in overlaps.items())
        raise SplitLeakageError(
            f"{reference_label}/{candidate_label} leakage detected: {details}"
        )
    return {
        "reference": reference_label,
        "candidate": candidate_label,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "overlaps": {},
    }


def assert_split_label(rows: Iterable[Mapping[str, Any]], expected: str, *, label: str) -> None:
    """Require every row in a file to carry the split it claims to represent."""
    bad = [str(row.get("task_id", "<unknown>")) for row in rows
           if row.get("split") != expected]
    if bad:
        raise SplitLeakageError(
            f"{label} contains rows with split != {expected!r}: {bad[:8]}"
        )


def audit_split_files(train_path: str | Path, test_path: str | Path,
                      *, val_path: str | Path | None = None) -> dict[str, Any]:
    """Audit JSONL train/test (and optional val) files before a real run."""
    train = load_task_rows(train_path)
    test = load_task_rows(test_path)
    assert_split_label(train, "train", label=str(train_path))
    assert_split_label(test, "test", label=str(test_path))
    report = {"train_test": assert_disjoint(train, test)}
    if val_path is not None:
        val = load_task_rows(val_path)
        assert_split_label(val, "val", label=str(val_path))
        report["train_val"] = assert_disjoint(train, val, candidate_label="val")
        report["val_test"] = assert_disjoint(val, test, reference_label="val")
    report["train_file"] = str(train_path)
    report["test_file"] = str(test_path)
    if val_path is not None:
        report["val_file"] = str(val_path)
    return report


def assert_same_eval_set(left_rows: Sequence[Mapping[str, Any]],
                         right_rows: Sequence[Mapping[str, Any]], *,
                         left_label: str = "open", right_label: str = "closed") -> dict[str, Any]:
    """Require two model evaluators to consume the same held-out tasks."""
    left = _index(left_rows, left_label)
    right = _index(right_rows, right_label)
    mismatches = {
        key: {"only_in_left": sorted(left[key] - right[key], key=str),
              "only_in_right": sorted(right[key] - left[key], key=str)}
        for key in left
        if left[key] != right[key]
    }
    if mismatches:
        raise SplitLeakageError(
            f"{left_label}/{right_label} test sets differ: {mismatches}"
        )
    return {"left": left_label, "right": right_label, "count": len(left_rows),
            "task_ids": sorted(left["task_id"])}
