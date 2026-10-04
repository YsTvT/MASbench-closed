import unittest
from masbench.longhorizon.core import paired_speedup, summarize
from masbench.longhorizon.workflow import WorkflowEnv, make_workflow, serial_baseline
from masbench.longhorizon.incident import IncidentEnv, make_incident, serial_baseline as incident_baseline
from masbench.metrics import success_first
from masbench.longhorizon.prompts import incident_prompt, validate_workflow_output, workflow_prompt
from masbench.longhorizon.closed_loop import run_closed_loop


class LongHorizonTests(unittest.TestCase):
    def test_workflow_parallel_success(self):
        case = make_workflow(3, 3)
        env = WorkflowEnv(case, workers=4)
        # Independent branches are dispatched together; controller can inspect
        # dependencies and reserve every output without race conditions.
        env.step_batch({"worker_0": {"type": "execute", "task": "report"},
                        "worker_1": {"type": "execute", "task": "dashboard"},
                        "worker_2": {"type": "execute", "task": "quality"},
                        "worker_3": {"type": "execute", "task": "security"}})
        env.advance(5)
        env.step_batch({"worker_0": {"type": "approve", "task": "approval"}})
        env.advance(3)
        env.step_batch({"worker_0": {"type": "execute", "task": "release"}})
        env.advance(4)
        env.step_batch({"worker_0": {"type": "submit"}})
        env.advance(1)
        self.assertTrue(env.score()["success"])

    def test_incident_requires_evidence_and_verification(self):
        case = make_incident(1, 2)
        case["deadline"] = 30  # manual trace below intentionally includes idle ticks
        env = IncidentEnv(case, workers=3)
        self.assertEqual(env.step_batch({"worker_0": {"type": "mitigate", "fix": "increase_db_pool"}})["worker_0"]["ok"], False)
        for i, signal in enumerate(case["signals"]):
            env.step_batch({"worker_%d" % i: {"type": "query", "signal": signal}})
        env.advance(3)
        root = {"cache_miss": "cache_config", "db_pool": "db_pool", "queue_lag": "worker_scale"}[case["fault"]]
        fix = {"cache_miss": "restore_cache_config", "db_pool": "increase_db_pool", "queue_lag": "scale_workers"}[case["fault"]]
        env.step_batch({"worker_0": {"type": "diagnose", "root_cause": root}}); env.advance(2)
        env.step_batch({"worker_0": {"type": "mitigate", "fix": fix}}); env.advance(3)
        env.step_batch({"worker_0": {"type": "verify", "checks": set(case["signals"])}}); env.advance(2)
        env.step_batch({"worker_0": {"type": "communicate", "body": "resolved"}}); env.advance(1)
        env.step_batch({"worker_0": {"type": "submit"}}); env.advance(1)
        self.assertTrue(env.score()["success"])

    def test_success_first_summary(self):
        reports = [{"task_id": "a", "success": False, "completion_ticks": None, "deadline": 20},
                   {"task_id": "b", "success": True, "completion_ticks": 10, "deadline": 20}]
        out = summarize(reports)
        self.assertEqual(out["successes"], 1)
        self.assertEqual(out["successful_mean_ticks"], 10)
        self.assertEqual(out["sr"], 0.5)
        self.assertAlmostEqual(out["composite_score"], 0.25)
        self.assertIsNone(reports[0]["completion_ticks"])

    def test_shared_metric_rejects_failed_time(self):
        with self.assertRaises(ValueError):
            success_first([{"success": False, "completion_ticks": 3, "deadline": 10}])
        out = success_first([{"success": False, "completion_ticks": None, "deadline": 10},
                             {"success": True, "completion_ticks": 5, "deadline": 10}])
        self.assertEqual(out["sr"], 0.5)
        self.assertEqual(out["composite_score"], 0.25)

    def test_prompts_hide_incident_answer_and_validate_contract(self):
        case = make_incident(2, 3)
        prompt = incident_prompt(case)
        self.assertNotIn('"fault"', prompt)
        self.assertNotIn('"root":', prompt)
        workflow = make_workflow(2, 3)
        self.assertIn("parallel_tasks", workflow_prompt(workflow))
        branches = [k for k in workflow["tasks"] if k not in ("approval", "release")]
        self.assertEqual(validate_workflow_output(workflow, {"parallel_tasks": branches, "final_action": "release"}), (True, None))
        self.assertEqual(validate_workflow_output(workflow, {"parallel_tasks": branches, "final_action": "released workflow"})[0], False)

    def test_closed_loop_serial_and_mas_complete(self):
        workflow = make_workflow(4, 2)
        serial = run_closed_loop(workflow, "workflow", "serial")
        mas = run_closed_loop(workflow, "workflow", "mas")
        self.assertTrue(serial["success"])
        self.assertTrue(mas["success"])
        self.assertLess(mas["completion_ticks"], serial["completion_ticks"])

    def test_closed_loop_incident_parallel_queries(self):
        case = make_incident(4, 3)
        serial = run_closed_loop(case, "incident", "serial")
        mas = run_closed_loop(case, "incident", "mas")
        self.assertFalse(serial["success"])
        self.assertTrue(mas["success"])
        self.assertIsNone(serial["completion_ticks"])


if __name__ == "__main__":
    unittest.main()
