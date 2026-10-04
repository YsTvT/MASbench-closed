import json
import subprocess
import sys
import unittest
from pathlib import Path


class ExpandedDataTests(unittest.TestCase):
    def test_expanded_data_has_original_scale_targets(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "data" / "longhorizon_test.jsonl").is_file():
            self.skipTest("held-out test data is kept in the private evaluation workspace")
        rows = []
        for name in ("longhorizon_train.jsonl", "longhorizon_dev.jsonl", "longhorizon_test.jsonl"):
            with (root / "data" / name).open() as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        self.assertEqual(sum(r["family"] == "workflow" for r in rows), 175)
        self.assertEqual(sum(r["family"] == "incident" for r in rows), 48)
        subprocess.run([sys.executable, str(root / "scripts" / "audit_longhorizon_data.py")], check=True)


if __name__ == "__main__":
    unittest.main()
