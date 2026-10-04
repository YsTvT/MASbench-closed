import json
import tempfile
import unittest
from pathlib import Path

from masbench.longhorizon.rl import (OnPolicyGRPOLearner, RealLongHorizonRLEnv,
                                     TrajectoryWriter, load_score_rows)


class _RealAdapter:
    backend_name = "fake-real-service"
    family = "workflow"
    real_rollout = True
    agents = ["worker_0"]

    def reset(self, case):
        self.n = 0
        self.finished = False
        return {"worker_0": {"tick": 0, "state": "live"}}

    def observe(self, agent):
        return {"tick": self.n, "state": "live" if not self.finished else "done"}

    def step_batch(self, actions):
        self.n += 1
        if any(a.get("type") == "submit" for a in actions.values()):
            self.finished = True
        return {"worker_0": {"ok": True}}

    def score(self):
        if not self.finished:
            return {"evaluation_status": "unassessed", "checks": {"release": False}}
        return {"evaluation_status": "evaluated", "success": True,
                "completion_ticks": self.n, "checks": {"release": True}}


class _Policy:
    def __init__(self):
        self.updated = False

    def sample_action(self, observation, agent, *, deterministic=False):
        return {"type": "submit"}

    def update(self, trajectories, advantages):
        self.updated = True
        return {"ok": True}

    def save_checkpoint(self, path, metadata):
        Path(path).mkdir(parents=True, exist_ok=True)


class Bench23RealRLTests(unittest.TestCase):
    def case(self, split="train"):
        return {"task_id": "workflow_000_d1", "split": split,
                "schema_version": "bench2-workflow-v1", "deadline": 5,
                "required_agents": 1}

    def test_real_episode_and_train_batch(self):
        adapter = _RealAdapter()
        env = RealLongHorizonRLEnv(adapter, max_ticks=4)
        env.reset(self.case(), "episode-1")
        env.step({"worker_0": {"type": "submit"}})
        episode = env.finish()
        self.assertTrue(episode["real_rollout"])
        self.assertEqual(episode["score"]["reward"], 1)
        policy = _Policy()
        result = OnPolicyGRPOLearner(policy).train_batch([episode])
        self.assertEqual(result["successes"], 1)
        self.assertTrue(policy.updated)

    def test_symbolic_and_test_split_are_rejected_for_train(self):
        policy = _Policy()
        env = RealLongHorizonRLEnv(_RealAdapter(), max_ticks=2)
        env.reset(self.case("test"), "test-1")
        env.step({"worker_0": {"type": "submit"}})
        episode = env.finish()
        with self.assertRaises(ValueError):
            OnPolicyGRPOLearner(policy).train_batch([episode])
        with self.assertRaises(ValueError):
            RealLongHorizonRLEnv(type("Synthetic", (), {"real_rollout": False, "family": "workflow"})())

    def test_writer_and_eval_require_real_evaluated_scores(self):
        env = RealLongHorizonRLEnv(_RealAdapter(), max_ticks=2)
        env.reset(self.case(), "episode-2")
        env.step({"worker_0": {"type": "submit"}})
        episode = env.finish()
        with tempfile.TemporaryDirectory() as tmp:
            writer = TrajectoryWriter(Path(tmp))
            writer.write(episode)
            rows = load_score_rows(Path(tmp) / "scores", split="train")
            self.assertEqual(len(rows), 1)
            p = Path(tmp) / "bad.json"
            p.write_text(json.dumps({"real_rollout": True, "evaluation_status": "unassessed"}))
            with self.assertRaises(ValueError):
                load_score_rows(p)


if __name__ == "__main__":
    unittest.main()
