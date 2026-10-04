import json
from pathlib import Path
from masbench.longhorizon.workflow import make_workflow
from masbench.longhorizon.incident import make_incident

root = Path(__file__).resolve().parents[1] / "data"
root.mkdir(exist_ok=True)
rows = []
targets = {"workflow": 175, "incident": 48}
# Keep the original-scale total while making the learning and evaluation
# partitions genuinely disjoint.  The split is stratified within each family
# so every partition contains multiple difficulty levels.
split_ratios = (0.60, 0.15, 0.25)
for family, fn in (("workflow", make_workflow), ("incident", make_incident)):
    n = targets[family]
    train_end = int(n * split_ratios[0])
    dev_end = train_end + int(n * split_ratios[1])
    for seed in range(targets[family]):
        difficulty = seed % 3 + 1
        row = fn(seed, difficulty)
        row["family"] = family
        if seed < train_end:
            row["split"] = "train"
        elif seed < dev_end:
            row["split"] = "dev"
        else:
            row["split"] = "test"
        row["generator_version"] = "longhorizon-v2-stratified"
        rows.append(row)
for split in ("train", "dev", "test"):
    with (root / ("longhorizon_%s.jsonl" % split)).open("w") as f:
        for row in rows:
            if row["split"] == split:
                f.write(json.dumps(row, sort_keys=True) + "\n")
manifest = {"count": len(rows), "families": targets,
            "dev": sum(r["split"] == "dev" for r in rows),
            "train": sum(r["split"] == "train" for r in rows),
            "test": sum(r["split"] == "test" for r in rows),
            "split_policy": "stratified 60/15/25 by family and seed",
            "protocol": "success-first; completion time is defined only for successful runs"}
(root / "longhorizon_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, sort_keys=True))
