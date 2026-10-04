"""Model-facing prompts and strict output validators for Bench2/Bench3.

The environment remains the source of truth.  These helpers only define a
reproducible interface for a controller; they never reveal hidden labels from
an incident case.
"""

import json


WORKFLOW_SYSTEM = """You are the coordinator for a long-horizon workflow.
Return exactly one JSON object and no prose.  Schedule every independent root
branch in parallel, then wait for their results before approval and release.
Use only the task names and action values present in the case.  The schema is:
{"parallel_tasks":["task_name", ...], "final_action":"release"}
"""

INCIDENT_SYSTEM = """You are the incident commander for a long-horizon fault.
Return exactly one JSON object and no prose.  Query all useful signals in
parallel, identify the root cause from evidence, apply the corresponding fix,
verify every observed signal, and communicate the resolution.  The schema is:
{"queries":["signal", ...], "root_cause":"label", "fix":"label",
 "verify":["signal", ...], "communication":"short message"}
"""


def workflow_prompt(case):
    branches = [k for k in case["tasks"] if k not in ("approval", "release")]
    public = {"task_id": case["task_id"], "difficulty": case["difficulty"],
              "deadline": case["deadline"], "branches": branches,
              "dependencies": {k: case["tasks"][k]["requires"] for k in case["tasks"]}}
    return WORKFLOW_SYSTEM + "\nCASE:\n" + json.dumps(public, sort_keys=True)


def incident_prompt(case, observation=None):
    # fault, trace_digest and any answer-bearing field are deliberately omitted.
    public = {"task_id": case["task_id"], "difficulty": case["difficulty"],
              "deadline": case["deadline"], "signals": list(case.get("query_signals", case["signals"])),
              "distractor_families": list(case["distractors"])}
    if observation:
        public["observation"] = observation
    return INCIDENT_SYSTEM + "\nCASE:\n" + json.dumps(public, sort_keys=True)


def validate_workflow_output(case, output):
    if not isinstance(output, dict):
        return False, "output_not_object"
    branches = [k for k in case["tasks"] if k not in ("approval", "release")]
    if output.get("parallel_tasks") != branches:
        return False, "parallel_tasks_mismatch"
    if output.get("final_action") != "release":
        return False, "final_action_mismatch"
    return True, None


def validate_incident_output(case, output):
    if not isinstance(output, dict):
        return False, "output_not_object"
    if output.get("queries") != list(case.get("query_signals", case["signals"])):
        return False, "queries_mismatch"
    if not isinstance(output.get("root_cause"), str) or not output["root_cause"]:
        return False, "root_cause_missing"
    if not isinstance(output.get("fix"), str) or not output["fix"]:
        return False, "fix_missing"
    if output.get("verify") != list(case["signals"]):
        return False, "verify_mismatch"
    if not isinstance(output.get("communication"), str) or not output["communication"].strip():
        return False, "communication_missing"
    return True, None
