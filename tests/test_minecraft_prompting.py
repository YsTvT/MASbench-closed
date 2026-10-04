import tempfile
import unittest

from masbench.controllers import ModelAdapter
from masbench.minecraft.prompting import (PromptMinecraftRunner, build_messages,
                                          parse_action, write_episode)


class _Model(ModelAdapter):
    metadata = {"provider": "test", "model": "fake"}

    def generate(self, messages, *, response_schema=None, cache_key=None):
        return {"tool": "MineBlock", "tool_input": {"name": "oak_log"}}


class _Adapter:
    backend_name = "fake-real"
    real_rollout = True

    def __init__(self, task, episode_id):
        self.tick = 0

    def reset(self, task):
        self.tick = 0

    def observe(self, agent):
        return {"tick": self.tick, "agent": agent}

    def step_batch(self, actions):
        self.tick += 1
        return {agent: {"ok": True} for agent in actions}

    def score(self):
        return {"evaluation_status": "evaluated", "success": False,
                "completion_ticks": None, "checks": {}}

    def close(self):
        pass


class MinecraftPromptingTests(unittest.TestCase):
    def test_original_tool_shape_is_normalized(self):
        action, diagnostic = parse_action(
            {"tool": "MineBlock", "tool_input": {"name": "oak_log",
                                                       "x": 1, "y": 64, "z": 2}},
            agent="agent_0",
        )
        self.assertEqual(action["type"], "gather")
        self.assertEqual(action["name"], "oak_log")
        self.assertTrue(diagnostic["valid"])

    def test_prompt_contains_task_observation_and_history(self):
        messages = build_messages(
            {"task_id": "task", "goal": "build", "split": "test",
             "deadline_ticks": 10, "action_space": ["observe", "submit"]},
            "agent_0", {"tick": 3, "task_id": "task"},
            [{"action": {"type": "observe", "result": {"task_id": "task"}}}],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn('"task_id"', messages[1]["content"])
        self.assertNotIn('"split"', messages[1]["content"])
        self.assertIn("recent_action_history", messages[1]["content"])

    def test_runner_writes_real_prompt_episode(self):
        task = {"task_id": "prompt-task", "split": "test",
                "topology_id": "prompt-topology", "goal": "build a shelter",
                "deadline_ticks": 4, "action_space": ["observe", "gather", "submit"],
                "agent_roles": [{"agent_id": "agent_0", "role": "builder"}]}
        runner = PromptMinecraftRunner(lambda task, episode_id: _Adapter(task, episode_id),
                                       _Model(), max_ticks=1)
        with tempfile.TemporaryDirectory() as directory:
            episode = runner.run_episode(task, "prompt-test")
            episode_path, score_path = write_episode(directory, episode)
            self.assertTrue(episode["real_rollout"])
            self.assertEqual(episode["score"]["model_type"], "closed_api")
            self.assertTrue(episode_path.exists())
            self.assertTrue(score_path.exists())


if __name__ == "__main__":
    unittest.main()
