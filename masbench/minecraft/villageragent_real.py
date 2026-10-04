"""Real VillagerAgent adapter factory.

This adapter is intentionally thin: the Mineflayer/VillagerAgent processes and
the Minecraft checker remain the authority.  It starts the two agent Flask
bridges and the existing construction checker for an episode, dispatches
typed tool actions through the VillagerAgent tools, and waits for the checker
record before returning the terminal score.

It is meant to be copied/imported on the server where ``env`` and ``pipeline``
from VillagerAgent are importable.  It refuses to run against a symbolic
backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import socket
import fcntl
from typing import Any, Mapping


class VillagerAgentRealAdapter:
    backend_name = "villageragent_mineflayer"
    real_rollout = True

    def __init__(self, task: Mapping[str, Any], episode_id: str,
                 *, root: str = ".", host: str = "127.0.0.1", port: int = 25565,
                 agent_names=("Alice", "Bob"), checker_timeout: int = 120,
                 server_root: str | None = None):
        self.task = dict(task)
        self.episode_id = episode_id
        self.root = Path(root)
        self.host, self.port = host, port
        self.agent_names = list(agent_names)
        self.checker_timeout = checker_timeout
        task_server_root = self.task.get("server_root")
        self.server_root = Path(server_root or task_server_root or os.environ.get(
            "MASBENCH_SERVER_ROOT", "/path/to/minecraft_server"))
        self.snapshot_root = Path(self.task.get("world_snapshot") or os.environ.get(
            "MASBENCH_WORLD_SNAPSHOT", str(self.server_root.parent / "MASbench" / "world_snapshots")))
        source_snapshot = self.task.get("source_world_snapshot")
        self.source_snapshot = Path(source_snapshot) if source_snapshot else None
        self.port = int(self.task.get("server_port", port))
        self.agent_port_base = int(self.task.get("agent_port_base", 5000))
        self._lock_handle = None
        self._server_process = None
        self.processes: list[subprocess.Popen] = []
        self.last_results: dict[str, Any] = {}
        self.task_name = str(task.get("task_name") or episode_id)

    def _kill_children(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes.clear()

    def reset(self, task: Mapping[str, Any]):
        self._kill_children()
        self.task = dict(task)
        if task.get("task_idx") is None and not os.environ.get("MASBENCH_TASK_IDX"):
            raise ValueError("real Minecraft training requires task.task_idx or MASBENCH_TASK_IDX; refusing implicit idx=34")
        self.task_name = str(task.get("task_name") or self.episode_id)
        self.server_root = Path(task.get("server_root") or self.server_root)
        self.snapshot_root = Path(task.get("world_snapshot") or self.snapshot_root)
        source_snapshot = task.get("source_world_snapshot")
        if source_snapshot:
            self.source_snapshot = Path(source_snapshot)
        self.port = int(task.get("server_port", self.port))
        self.agent_port_base = int(task.get("agent_port_base", self.agent_port_base))
        self._prepare_villageragent_root()
        self._acquire_world_lock()
        self._reset_world_snapshot()
        # The checker writes data/score.json; remove stale evidence before each episode.
        for stale in (self.root / "data" / "score.json",
                      self.root / "data" / "action_log.json",
                      self.root / "result" / self.task_name / "score.json"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        env_dir = self.root / "env"
        child_env = os.environ.copy()
        child_env["NODE_PATH"] = str(self.root / "node_modules")
        for idx, name in enumerate(self.agent_names):
            self.processes.append(subprocess.Popen([
                sys.executable, str(env_dir / "minecraft_server.py"),
                "-H", self.host, "-P", str(self.port), "-LP", str(self.agent_port_base + idx),
                "-U", name, "-W", "world", "-D", "False"],
                cwd=self.root, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        task_idx = str(task["task_idx"] if task.get("task_idx") is not None else os.environ["MASBENCH_TASK_IDX"])
        self.processes.append(subprocess.Popen([
            sys.executable, str(env_dir / "build_judger.py"), "--idx", task_idx,
            "--host", self.host, "--port", str(self.port), "--agent_num",
            str(len(self.agent_names)), "--agent_names", ",".join(self.agent_names),
            "--task_name", self.task_name], cwd=self.root, env=child_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        # Flask bridges need time to bind before the first tool call.
        time.sleep(float(os.environ.get("MASBENCH_AGENT_START_WAIT", "8")))
        self.last_results = {}
        return None

    def _prepare_villageragent_root(self):
        """Isolate every bridge's relative data, caches, logs and checker.

        The vendor tools read Alice/Bob endpoints from ``data/url_prefix.json``
        relative to cwd.  Separate Java worlds alone do not isolate actions.
        """
        source = self.task.get("base_villageragent_root")
        if source is not None:
            source = Path(source).resolve()
            self.root = self.root.resolve()
            if source != self.root and not self.root.exists():
                self.root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, self.root, symlinks=True,
                                ignore=shutil.ignore_patterns(
                                    ".masbench_candidates", "node_modules", "result",
                                    ".cache", "__pycache__", "logs", ".git"))
            if (source / "node_modules").exists() and not (self.root / "node_modules").exists():
                (self.root / "node_modules").symlink_to(source / "node_modules",
                                                       target_is_directory=True)
        self.root = self.root.resolve()
        if not (self.root / "env" / "minecraft_client.py").is_file():
            raise RuntimeError(f"VillagerAgent checkout is missing: {self.root}")
        for directory in ("data", "result", ".cache"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        endpoints = {name: f"http://{self.host}:{self.agent_port_base + i}"
                     for i, name in enumerate(self.agent_names)}
        (self.root / "data" / "url_prefix.json").write_text(json.dumps(endpoints))

    def _acquire_world_lock(self):
        if self._lock_handle is not None:
            return
        # A lock inside the snapshot makes it exist before its first copy,
        # causing an empty generated world to replace the authoritative world.
        lock_path = self.snapshot_root.parent / ".locks" / f"{self.snapshot_root.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = open(lock_path, "w")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)

    def _server_pid(self):
        pid_file = self.server_root / ".masbench_server.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                return pid
            except Exception:
                pass
        if os.environ.get("MASBENCH_ALLOW_EXTERNAL_SERVER_RESET") != "1":
            return None
        try:
            out = subprocess.check_output(["pgrep", "-f", "java .*server.jar"], text=True)
            return int(out.splitlines()[0])
        except Exception:
            return None

    def _wait_server(self, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=1):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Minecraft server did not become ready")

    def _reset_world_snapshot(self):
        """Restore one identical world snapshot for every GRPO candidate.

        The first episode snapshots the current server world after a clean
        stop. Later candidates stop/restart the same local server and restore
        that snapshot, so a GRPO group never shares mutated state.
        """
        base_server = Path(self.task.get("base_server_root") or os.environ.get(
            "MASBENCH_SERVER_ROOT", str(self.server_root)))
        if not self.server_root.exists() and base_server.exists() and self.server_root != base_server:
            self.server_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(base_server, self.server_root, symlinks=True,
                            ignore=shutil.ignore_patterns(".masbench_candidates", ".masbench_server.pid"))
        world = self.server_root / "world"
        snapshot = self.snapshot_root
        if self.source_snapshot is not None and not self.source_snapshot.is_dir():
            raise RuntimeError(f"authoritative world snapshot is missing: {self.source_snapshot}")
        if self.source_snapshot is not None and not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.source_snapshot, snapshot)
        if not snapshot.is_dir() or not (snapshot / "level.dat").is_file():
            raise RuntimeError(f"world snapshot lacks level.dat: {snapshot}")
        pid = self._server_pid()
        if pid:
            os.kill(pid, 15)
            deadline = time.time() + 45
            while time.time() < deadline and self._server_pid():
                time.sleep(1)
        if world.exists():
            shutil.rmtree(world)
        shutil.copytree(snapshot, world)
        properties = self.server_root / "server.properties"
        if properties.exists():
            lines = properties.read_text().splitlines()
            found = False
            updated = []
            for line in lines:
                if line.startswith("server-port="):
                    updated.append(f"server-port={self.port}")
                    found = True
                else:
                    updated.append(line)
            if not found:
                updated.append(f"server-port={self.port}")
            properties.write_text("\n".join(updated) + "\n")
        java_xms = os.environ.get("MASBENCH_JAVA_XMS", "1G")
        java_xmx = os.environ.get("MASBENCH_JAVA_XMX", "4G")
        self._server_process = subprocess.Popen(
            ["java", f"-Xms{java_xms}", f"-Xmx{java_xmx}", "-jar", "server.jar", "nogui"],
            cwd=self.server_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (self.server_root / ".masbench_server.pid").write_text(str(self._server_process.pid))
        self._wait_server()

    def _actual_agent(self, agent: str) -> str:
        # Benchmark roles use stable agent_0/agent_1 ids; VillagerAgent
        # exposes the two Mineflayer players as Alice/Bob.
        aliases = {"agent_0": "Alice", "agent_1": "Bob"}
        return aliases.get(str(agent), str(agent))

    def _tool(self, agent: str, action: Mapping[str, Any]):
        # The training process runs from MASbench while VillagerAgent lives in
        # a sibling vendor tree; make its Python package importable explicitly.
        root_str = str(self.root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        # VillagerAgent resolves data/url_prefix.json and action logs relative
        # to its own checkout, so align cwd before invoking a tool.
        os.chdir(self.root)
        os.environ["NODE_PATH"] = str(self.root / "node_modules")
        from env.minecraft_client import Agent
        actual_agent = self._actual_agent(agent)
        typ = str(action.get("type", "observe"))
        if typ == "submit":
            return {"ok": True, "submitted": True}
        name_map = {
            "move": "navigateTo", "navigateTo": "navigateTo",
            "observe": "scanNearbyEntities", "scanNearbyEntities": "scanNearbyEntities",
            "gather": "MineBlock", "mine": "MineBlock", "place": "placeBlock",
            "craft": "craftBlock", "withdraw": "withdrawItem", "withdrawItem": "withdrawItem",
        }
        tool_name = name_map.get(typ, typ)
        tool = getattr(Agent, tool_name, None)
        if tool is None or not hasattr(tool, "invoke"):
            return {"ok": False, "error": f"unsupported_tool:{tool_name}"}
        kwargs = dict(action)
        kwargs.pop("type", None)
        for key in list(kwargs):
            if str(key).startswith("_"):
                kwargs.pop(key, None)
        kwargs["player_name"] = actual_agent
        kwargs.setdefault("emotion", [])
        kwargs.setdefault("murmur", "")
        if tool_name == "scanNearbyEntities":
            kwargs.setdefault("item_name", "")
            kwargs.setdefault("radius", 16)
            kwargs.setdefault("item_num", 0)
        return tool.invoke(kwargs)

    def observe(self, agent: str):
        result = self._tool(agent, {"type": "observe"})
        return {"agent": agent, "tick": len(self.last_results),
                "last_result": result, "task_id": self.task.get("task_id")}

    def step_batch(self, actions: Mapping[str, Mapping[str, Any]]):
        results = {}
        for agent, action in sorted(actions.items()):
            results[agent] = self._tool(agent, action)
        self.last_results = results
        return results

    def score(self):
        # The checker creates data/score.json as [] at startup and appends a
        # record only after its first measured interval.  An empty list is not
        # evidence, so wait for a non-empty record or a result-file fallback.
        paths = [self.root / "result" / self.task_name / "score.json",
                 self.root / "data" / "score.json"]
        deadline = time.time() + self.checker_timeout
        raw = None
        source = None
        while time.time() < deadline and raw is None:
            for candidate in paths:
                if not candidate.exists():
                    continue
                try:
                    parsed = json.loads(candidate.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, list):
                    if not parsed:
                        continue
                    parsed = parsed[-1]
                if isinstance(parsed, dict) and parsed:
                    raw, source = parsed, candidate
                    break
            if raw is None:
                time.sleep(1)
        if raw is None:
            return {"evaluation_status": "unassessed", "checks": {},
                    "reason": "checker_missing_or_empty",
                    "real_rollout": True}
        # The legacy VillagerAgent checker writes a JSON list when multiple
        # intervals are recorded; the normalized runner writes a single dict.
        # Training must accept the latest checker record in either shape.
        if isinstance(raw, list):
            raw = raw[-1] if raw else {}
        if not isinstance(raw, Mapping):
            return {"evaluation_status": "unassessed", "checks": {},
                    "reason": "checker_invalid_payload", "real_rollout": True}
        block = float(raw.get("block_hit_rate", 0.0))
        view = float(raw.get("view_hit_rate", 0.0))
        success = block >= 1.0 and view >= 1.0
        return {"evaluation_status": "evaluated", "success": success,
                "completion_ticks": raw.get("use_time") if success else None,
                "checks": {"block_hit_rate": block, "view_hit_rate": view},
                "raw_checker": raw, "checker_path": str(source),
                "real_rollout": True}

    def close(self):
        self._kill_children()
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
        self._server_process = None
        pid_file = self.server_root / ".masbench_server.pid"
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None


def make_adapter(task, episode_id):
    """Factory consumed by train_real_rl.py on the VillagerAgent server."""
    return VillagerAgentRealAdapter(task, episode_id,
                                    root=task.get("villageragent_root") or os.environ.get("VILLAGERAGENT_ROOT", "."),
                                    server_root=task.get("server_root"),
                                    checker_timeout=int(os.environ.get("MASBENCH_CHECKER_TIMEOUT", "120")))
