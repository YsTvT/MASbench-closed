import unittest
from pathlib import Path

from masbench.minecraft.tasks import load_tasks
from masbench.minecraft.env import run_reference


class MinecraftTaskTests(unittest.TestCase):
    def test_graph_data_is_valid_and_split(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "data" / "minecraft_tasks.jsonl").is_file():
            self.skipTest("private benchmark task/oracle data is not in the public source snapshot")
        rows = load_tasks(root / "data" / "minecraft_tasks.jsonl")
        self.assertEqual(len(rows), 300)
        self.assertEqual({r["split"] for r in rows}, {"train", "dev", "test"})
        self.assertEqual({r["subfamily"] for r in rows}, {"construction", "cooking", "escape", "mas_hard"})
        self.assertTrue(all(any(len(g) >= 2 for g in r["parallel_groups"]) for r in rows))
        with (root / "data" / "minecraft_reference_plans.jsonl").open() as handle:
            plans = [line for line in handle if line.strip()]
        self.assertEqual(len(plans), 900)
        with (root / "data" / "minecraft_episodes.jsonl").open() as handle:
            episodes = [line for line in handle if line.strip()]
        self.assertEqual(len(episodes), 1800)

    def test_symbolic_adapter_has_parallelism_gap(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "data" / "minecraft_tasks.jsonl").is_file():
            self.skipTest("private benchmark task data is not in the public source snapshot")
        rows = load_tasks(root / "data" / "minecraft_tasks.jsonl")
        serial = run_reference(rows[0], "serial")
        parallel = run_reference(rows[0], "parallel")
        self.assertTrue(parallel["success"])
        self.assertLess(parallel["completion_ticks"], serial["completion_ticks"] or 10**9)


if __name__ == "__main__":
    unittest.main()
