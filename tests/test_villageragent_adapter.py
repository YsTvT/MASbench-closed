import unittest

from masbench.minecraft.tasks import load_tasks
from masbench.minecraft.villageragent import VillagerAgentAdapter


class FakeVillagerClient:
    def reset_world(self, world_config, seed, inventory, roles):
        self.state = {"completed_subtasks": [], "inventory": {}, "predicates": {}}

    def observe(self, agent):
        return self.state

    def execute_action(self, agent, action):
        sid = action.get("subtask_id")
        if sid:
            self.state["completed_subtasks"].append(sid)
        return {"ok": True}

    def world_snapshot(self):
        return self.state

    def advance(self, ticks):
        return None


class VillagerAdapterTests(unittest.TestCase):
    def test_reset_observe_step_contract(self):
        task = load_tasks("MASbench/data/minecraft_tasks.jsonl", split="train")[0]
        client = FakeVillagerClient()
        adapter = VillagerAgentAdapter(client, ["agent_0", "agent_1"])
        adapter.reset(task)
        client.state["predicates"] = {c["predicate"]: True for c in task["concrete_success_checks"]}
        client.state["inventory"] = {
            key: value
            for subtask in task["subtasks"]
            for key, value in subtask["outputs"].items()
        }
        self.assertEqual(adapter.observe("agent_0")["tick"], 0)
        for subtask in task["subtasks"]:
            adapter.step_batch({"agent_0": {"type": subtask["action"], "subtask_id": subtask["id"]}})
        adapter.step_batch({"agent_0": {"type": "submit"}})
        self.assertTrue(adapter.score()["success"])


if __name__ == "__main__":
    unittest.main()
