import json
import subprocess
import sys
import unittest
from pathlib import Path


class MaterializedBench23Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_materialized_pool_counts_and_disjoint_ids(self):
        expected = {"bench2_workflow": 175, "bench3_incident": 48}
        all_ids = set()
        for directory, count in expected.items():
            rows = []
            for split in ("train", "dev", "test"):
                path = self.root / "data" / directory / (split + ".jsonl")
                self.assertTrue(path.exists(), path)
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            self.assertEqual(len(rows), count)
            ids = {row["task_id"] for row in rows}
            self.assertEqual(len(ids), count)
            self.assertTrue(all_ids.isdisjoint(ids))
            all_ids.update(ids)

    def test_materialized_pool_strict_audit(self):
        audit = self.root / "scripts" / "audit_bench23_data.py"
        subprocess.run([sys.executable, str(audit)], check=True, cwd=self.root.parent)


if __name__ == "__main__":
    unittest.main()
