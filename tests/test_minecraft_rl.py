import json
import tempfile
import unittest
from pathlib import Path

from masbench.minecraft.rl import (OnPolicyGRPOLearner, TrajectoryWriter,
                                   RealMinecraftRLEnv, _progress,
                                   group_advantages, load_score_rows)
try:
    from masbench.verl_minecraft_env import (MinecraftEnvironmentManager,
                                             MinecraftWorker)
    _VERL_BRIDGE_AVAILABLE = True
except ModuleNotFoundError as exc:
    # The repository's dependency-free unit-test environment does not install
    # numpy/Ray.  The bridge tests run in the dedicated veRL environment.
    if exc.name not in {"numpy", "ray"}:
        raise
    MinecraftEnvironmentManager = MinecraftWorker = None
    _VERL_BRIDGE_AVAILABLE = False


class _Policy:
    def __init__(self):
        self.updated = None

    def update(self, trajectories, advantages):
        self.updated = (trajectories, advantages)
        return {"ok": True}


class _FailedRealAdapter:
    backend_name = "fake-villageragent"
    real_rollout = True

    def reset(self, task):
        self.task = task

    def observe(self, agent):
        return {"agent": agent}

    def step_batch(self, actions):
        return {agent: {"ok": True} for agent in actions}

    def score(self):
        return {"evaluation_status": "evaluated", "success": False,
                "checks": {"block_hit_rate": 0.0, "view_hit_rate": 0.0}}


class _ToolErrorAdapter:
    real_rollout = True

    def step_batch(self, actions):
        del actions
        raise RuntimeError("mineflayer unavailable")

    def observe(self, agent):
        raise ConnectionError(f"bridge unavailable for {agent}")

    def score(self):
        return {"evaluation_status": "evaluated", "success": False,
                "checks": {"block_hit_rate": 0.0, "view_hit_rate": 0.0}}

    def close(self):
        pass


class MinecraftRLContractTests(unittest.TestCase):
    def test_fractional_checker_progress_is_not_booleanized(self):
        self.assertAlmostEqual(
            _progress({"checks": {"block_hit_rate": 0.25, "view_hit_rate": 0.75}}),
            0.5,
        )
        self.assertAlmostEqual(
            _progress({"checks": {"block_hit_rate": 2.0, "diagnostic": "ok"}}),
            1.0,
        )

    def test_real_failure_uses_binary_terminal_reward(self):
        task = {"task_id": "t", "split": "train", "topology_id": "top",
                "deadline_ticks": 4, "agent_roles": [{"agent_id": "agent_0"}]}
        env = RealMinecraftRLEnv(_FailedRealAdapter(), max_ticks=2)
        env.reset(task, "episode-fail")
        env.step({"agent_0": {"type": "observe"}})
        episode = env.finish()
        self.assertEqual(episode["score"]["reward"], 0)
        self.assertEqual(episode["terminal_reward"], 0.0)

    def test_grpo_advantages_are_group_centered(self):
        episodes = [{"task_id": "t", "terminal_reward": 1.0},
                    {"task_id": "t", "terminal_reward": -1.0},
                    {"task_id": "u", "terminal_reward": 1.0}]
        advantages = group_advantages(episodes)
        self.assertAlmostEqual(sum(advantages[:2]), 0.0)
        self.assertEqual(advantages[2], 0.0)  # one candidate has no baseline

    def test_learner_rejects_symbolic_and_unassessed(self):
        learner = OnPolicyGRPOLearner(_Policy())
        with self.assertRaises(ValueError):
            learner.train_batch([{"real_rollout": False,
                                  "score": {"evaluation_status": "evaluated"}}])
        with self.assertRaises(ValueError):
            learner.train_batch([{"real_rollout": True,
                                  "score": {"evaluation_status": "unassessed"}}])

    def test_writer_and_eval_reject_symbolic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TrajectoryWriter(root, require_real=True)
            with self.assertRaises(ValueError):
                writer.write({"episode_id": "synthetic", "real_rollout": False})
            score = root / "score.json"
            score.write_text(json.dumps({"evaluation_status": "evaluated", "real_rollout": False,
                                         "success": True, "completion_ticks": 1,
                                         "deadline": 2}))
            with self.assertRaises(ValueError):
                load_score_rows(score, require_real=True)

    @unittest.skipUnless(_VERL_BRIDGE_AVAILABLE, "dedicated veRL dependencies are unavailable")
    def test_worker_contains_tool_and_observation_errors(self):
        worker = MinecraftWorker(seed=3, env_kwargs={"max_turns": 1}, candidate_id=2)
        worker.task = {"task_id": "tool-error", "task_name": "tool-error",
                       "task_idx": 0, "split": "train"}
        worker.adapter = _ToolErrorAdapter()
        text, reward, done, info = worker.step('{"type":"observe"}')
        payload = json.loads(text)
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["evaluation_status"], "evaluated")
        self.assertIn("tool_exception", payload["last_results"]["agent_0"]["error"])
        self.assertIn("observation_error", payload["observations"]["agent_0"])

    @unittest.skipUnless(_VERL_BRIDGE_AVAILABLE, "dedicated veRL dependencies are unavailable")
    def test_candidate_paths_are_disjoint(self):
        worker = MinecraftWorker(
            seed=1,
            env_kwargs={"server_root": "/srv/mc", "world_snapshot": "/srv/world",
                        "villageragent_root": "/srv/va", "server_port_base": 25600,
                        "agent_port_base": 6000},
            candidate_id=7,
        )
        task = worker._candidate_task({"task_id": "t", "task_name": "task"})
        self.assertEqual(task["server_root"], "/srv/mc/.masbench_candidates/candidate_007")
        self.assertEqual(task["world_snapshot"], "/srv/world_candidate_007")
        self.assertEqual(task["villageragent_root"], "/srv/va/.masbench_candidates/candidate_007")
        self.assertEqual(task["server_port"], 25607)
        self.assertEqual(task["agent_port_base"], 6070)

    @unittest.skipUnless(_VERL_BRIDGE_AVAILABLE, "dedicated veRL dependencies are unavailable")
    def test_candidate_task_unwraps_dataset_env_kwargs(self):
        worker = MinecraftWorker(seed=1, env_kwargs={}, candidate_id=0)
        task = worker._candidate_task({
            "env_kwargs": [json.dumps({"task_id": "dataset-task", "task_idx": 4,
                                        "split": "train"})],
            "prompt": [{"role": "user", "content": "ignored"}],
            "extra_info": {"task_id": "dataset-task"},
        })
        self.assertEqual(task["task_id"], "dataset-task")
        self.assertEqual(task["task_idx"], 4)

    @unittest.skipUnless(_VERL_BRIDGE_AVAILABLE, "dedicated veRL dependencies are unavailable")
    def test_success_evaluator_accepts_nested_and_flat_infos(self):
        manager = object.__new__(MinecraftEnvironmentManager)
        nested = manager.success_evaluator(total_infos=[
            [{"evaluation_status": "running"}, {"evaluation_status": "evaluated", "won": True}],
            [{"evaluation_status": "evaluated", "won": False}],
        ])
        self.assertEqual(nested["success_rate"].tolist(), [1.0, 0.0])
        flat = manager.success_evaluator(total_infos=[
            {"evaluation_status": "running"},
            {"evaluation_status": "evaluated", "won": True},
        ])
        self.assertEqual(flat["success_rate"].tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
