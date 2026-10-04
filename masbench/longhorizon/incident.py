"""Bench3: AIOps-style fault diagnosis, mitigation and verification."""

from copy import deepcopy
from .core import Simulation, digest


FAULTS = {
    "cache_miss": {"symptoms": ["latency_high", "cache_hit_low"], "root": "cache_config", "fix": "restore_cache_config"},
    "db_pool": {"symptoms": ["error_rate_high", "db_pool_exhausted"], "root": "db_pool", "fix": "increase_db_pool"},
    "queue_lag": {"symptoms": ["queue_lag", "worker_cpu_low"], "root": "worker_scale", "fix": "scale_workers"},
}


def make_incident(seed=0, difficulty=1):
    names = sorted(FAULTS)
    fault = names[seed % len(names)]
    if difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2 or 3")
    # d1 is a clean two-signal diagnosis; d2 adds one distractor family and
    # d3 adds two, so difficulty increases causal ambiguity and query work.
    distractors = [x for x in names if x != fault][:max(0, difficulty - 1)]
    # Harder cases require checking plausible distractor telemetry before a
    # diagnosis is accepted.  The hidden root cause is never exposed.
    query_signals = list(FAULTS[fault]["symptoms"])
    for distractor in distractors:
        query_signals.extend(FAULTS[distractor]["symptoms"])
    return {"task_id": "incident_%03d_d%d" % (seed, difficulty), "seed": seed,
            # Query count grows with difficulty; the deadline is calibrated so
            # serial diagnosis fails on the deepest cases while MAS can batch.
            "deadline": 14 + 3 * (difficulty - 1), "difficulty": difficulty,
            "fault": fault, "signals": FAULTS[fault]["symptoms"],
            "query_signals": query_signals, "distractors": distractors,
            "required": ["root_cause", "mitigated", "verified"],
            "trace_digest": digest({"fault": fault, "seed": seed})}


class IncidentEnv(Simulation):
    def _reset_state(self):
        self.evidence = set()
        self.root_cause = None
        self.mitigated = False
        self.verified = False
        self.communicated = False
        self.submitted = False

    def public_state(self):
        return {"evidence": sorted(self.evidence), "root_cause": self.root_cause,
                "mitigated": self.mitigated, "verified": self.verified, "communicated": self.communicated}

    def _prepare(self, agent, action):
        typ = action.get("type")
        if typ in ("query", "diagnose", "mitigate", "verify", "communicate", "submit"):
            if typ in ("mitigate", "verify") and not self.evidence:
                raise ValueError("evidence_required")
            return action, {"query": 2, "diagnose": 2, "mitigate": 3, "verify": 2, "communicate": 1, "submit": 1}[typ], "incident:%s" % typ if typ not in ("query", "diagnose") else None
        raise ValueError("unsupported_action")

    def _finish(self, agent, action):
        typ = action["type"]
        if typ == "query":
            signal = action.get("signal")
            if signal not in FAULTS[self.case["fault"]]["symptoms"] and signal not in sum((FAULTS[x]["symptoms"] for x in self.case["distractors"]), []):
                return {"ok": False, "error": "unknown_signal"}
            self.evidence.add(signal)
            return {"ok": True, "signal": signal, "value": signal in self.case["signals"]}
        if typ == "diagnose":
            candidate = action.get("root_cause")
            if candidate == FAULTS[self.case["fault"]]["root"]:
                self.root_cause = candidate
                self.evidence.add("diagnosis")
                return {"ok": True}
            return {"ok": False, "error": "wrong_root_cause"}
        if typ == "mitigate":
            if self.root_cause != FAULTS[self.case["fault"]]["root"]:
                return {"ok": False, "error": "diagnosis_required"}
            self.mitigated = action.get("fix") == FAULTS[self.case["fault"]]["fix"]
            return {"ok": self.mitigated}
        if typ == "verify":
            checks = action.get("checks", set(FAULTS[self.case["fault"]]["symptoms"]))
            self.verified = bool(self.mitigated and set(self.case["signals"]).issubset(set(checks)))
            return {"ok": self.verified}
        if typ == "communicate":
            self.communicated = bool(action.get("body"))
            return {"ok": self.communicated}
        if typ == "submit":
            checks = self.check(); self.submitted = True
            return {"ok": all(checks.values()), "checks": checks}

    def check(self):
        return {"root_cause": self.root_cause == FAULTS[self.case["fault"]]["root"],
                "mitigated": self.mitigated, "verified": self.verified, "communicated": self.communicated}


def serial_baseline(case):
    env = IncidentEnv(case, workers=1)
    for signal in case.get("query_signals", case["signals"]):
        env.step_batch({"worker_0": {"type": "query", "signal": signal}}); env.advance(2)
    root, fix = FAULTS[case["fault"]]["root"], FAULTS[case["fault"]]["fix"]
    for action in ({"type": "diagnose", "root_cause": root}, {"type": "mitigate", "fix": fix},
                   {"type": "verify"}, {"type": "communicate", "body": "incident resolved"}, {"type": "submit"}):
        env.step_batch({"worker_0": action}); env.advance(3)
    return env.score()
