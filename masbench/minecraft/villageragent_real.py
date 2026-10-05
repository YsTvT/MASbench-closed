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
import urllib.error
import urllib.request
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
        self._process_roles: dict[int, str] = {}
        self._process_logs: dict[int, Path] = {}
        self._log_handles: list[Any] = []
        self.last_results: dict[str, Any] = {}
        self.task_name = str(task.get("task_name") or episode_id)

    def _log_dir(self) -> Path:
        # Keep one episode's evidence together.  The directory is deliberately
        # under the isolated candidate root, so parallel GRPO candidates cannot
        # overwrite one another's bridge/checker diagnostics.
        path = self.root / "logs" / str(self.episode_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _spawn(self, role: str, argv: list[str], *, cwd: Path,
               env: Mapping[str, str], log_name: str) -> subprocess.Popen:
        log_path = self._log_dir() / log_name
        handle = log_path.open("ab", buffering=0)
        self._log_handles.append(handle)
        proc = subprocess.Popen(
            # Give child processes a private stream rather than the training
            # process's controlling terminal.
            argv, cwd=cwd, env=dict(env), stdin=subprocess.PIPE,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
        self.processes.append(proc)
        self._process_roles[proc.pid] = role
        self._process_logs[proc.pid] = log_path
        return proc

    def _process_error(self, proc: subprocess.Popen) -> str:
        role = self._process_roles.get(proc.pid, "child")
        log_path = self._process_logs.get(
            proc.pid, self._log_dir() / f"child_{proc.pid}.log")
        try:
            tail = log_path.read_text(errors="replace")[-2000:]
        except OSError:
            tail = ""
        return f"{role} exited with code {proc.returncode}; log={log_path}; tail={tail!r}"

    def _assert_children_alive(self, *, allow_checker_exit=False):
        for proc in self.processes:
            if proc.poll() is None:
                continue
            role = self._process_roles.get(proc.pid, "child")
            if allow_checker_exit and role == "checker":
                continue
            raise RuntimeError(self._process_error(proc))

    def _kill_children(self):
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    # _spawn starts a new session so this also reaps the
                    # javascript bridge child created by env.py.
                    os.killpg(proc.pid, 15)
                except (OSError, ProcessLookupError):
                    proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.stdin is not None:
                proc.stdin.close()
        self.processes.clear()
        self._process_roles.clear()
        self._process_logs.clear()
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._log_handles.clear()

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
        child_env["PYTHONUNBUFFERED"] = "1"
        # Set the JavaScript-to-Python call budget before env_api imports
        # JSPyBridge.  The legacy minecraft_server.py sets it only after that
        # import, which is too late for the already spawned Node process.
        # Keep this opt-in scoped to Qwen; the shared closed-model adapter
        # keeps its existing environment.
        if os.environ.get("MASBENCH_OPEN_MODEL_RUN") == "1":
            child_env.setdefault("REQ_TIMEOUT", "1800000")
        node_selection = (child_env.get("MASBENCH_NODE_BIN") or
                          child_env.get("NODE_BIN") or "node")
        node_bin = shutil.which(node_selection, path=child_env.get("PATH"))
        if node_bin is None:
            raise RuntimeError(
                "Node executable is unavailable; set MASBENCH_NODE_BIN or NODE_BIN")
        node_path = Path(node_bin).resolve()
        child_env["NODE_BIN"] = str(node_path)
        # env.py starts its legacy judgers with ``python``.  Ray workers may
        # inherit a base-shell PATH, so put this worker's interpreter first;
        # otherwise the judger misses packages installed in the veRL env.
        python_dir = str(Path(sys.executable).resolve().parent)
        child_env["PATH"] = os.pathsep.join(
            [python_dir, str(node_path.parent), child_env.get("PATH", "")])
        # The checker and bridges import packages from the candidate checkout.
        # Preserve any caller supplied path while making that checkout first.
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(self.root), child_env.get("PYTHONPATH", "")])
        bridge_stagger = max(0.0, float(os.environ.get("MASBENCH_BRIDGE_STAGGER", "0")))
        for idx, name in enumerate(self.agent_names):
            self._spawn("alice_bridge" if idx == 0 else "bob_bridge", [
                sys.executable, str(env_dir / "minecraft_server.py"),
                "-H", self.host, "-P", str(self.port), "-LP", str(self.agent_port_base + idx),
                "-U", name, "-W", "world", "-D", "False"],
                cwd=self.root, env=child_env, log_name=f"bridge_{name.lower()}.log")
            if bridge_stagger and idx + 1 < len(self.agent_names):
                time.sleep(bridge_stagger)
        task_idx = str(task["task_idx"] if task.get("task_idx") is not None else os.environ["MASBENCH_TASK_IDX"])
        self._spawn("checker", [
            sys.executable, str(env_dir / "build_judger.py"), "--idx", task_idx,
            "--host", self.host, "--port", str(self.port), "--agent_num",
            str(len(self.agent_names)), "--agent_names", ",".join(self.agent_names),
            "--task_name", self.task_name], cwd=self.root, env=child_env,
            log_name="checker.log")
        # Flask bridges need time to bind before the first tool call.  Poll the
        # configured bridge ports and issue a request to their base URL: any
        # HTTP response (including 404) proves the server is serving, without
        # assuming a route that belongs to a particular VillagerAgent version.
        self._wait_bridges_ready(float(os.environ.get("MASBENCH_AGENT_START_WAIT", "30")))
        self._assert_children_alive()
        self.last_results = {}
        return None

    def _wait_bridges_ready(self, timeout: float):
        deadline = time.time() + timeout
        pending = list(enumerate(self.agent_names))
        while pending and time.time() < deadline:
            self._assert_children_alive()
            next_pending = []
            for idx, name in pending:
                port = self.agent_port_base + idx
                try:
                    with socket.create_connection((self.host, port), timeout=0.5):
                        request = urllib.request.Request(
                            f"http://{self.host}:{port}/", method="GET")
                        try:
                            urllib.request.urlopen(request, timeout=1).close()
                        except urllib.error.HTTPError:
                            # A 404/405 is still a valid HTTP readiness signal.
                            pass
                        except (urllib.error.URLError, OSError):
                            next_pending.append((idx, name))
                            continue
                        continue
                except OSError:
                    next_pending.append((idx, name))
            pending = next_pending
            if pending:
                time.sleep(0.25)
        if pending:
            self._assert_children_alive()
            details = ", ".join(
                f"{name}:{self.agent_port_base + idx}" for idx, name in pending)
            raise RuntimeError(f"bridge HTTP readiness timed out ({details}); logs={self._log_dir()}")
        # HTTP readiness only means waitress is serving; the Mineflayer bot
        # may still be logging into Minecraft.  In the open-model candidate
        # wrapper, /post_ping safely reports status=false until bot.entity is
        # available.  Waiting for that marker prevents the first rollout tool
        # from entering the legacy error path while the bot is still spawning.
        if os.environ.get("MASBENCH_OPEN_MODEL_RUN") == "1":
            deadline = time.time() + timeout
            pending = list(enumerate(self.agent_names))
            while pending and time.time() < deadline:
                self._assert_children_alive()
                next_pending = []
                for idx, name in pending:
                    try:
                        request = urllib.request.Request(
                            f"http://{self.host}:{self.agent_port_base + idx}/post_ping",
                            method="GET")
                        with urllib.request.urlopen(request, timeout=1) as response:
                            payload = json.loads(response.read().decode("utf-8"))
                        if not bool(payload.get("status")):
                            next_pending.append((idx, name))
                    except (urllib.error.URLError, OSError, ValueError):
                        next_pending.append((idx, name))
                pending = next_pending
                if pending:
                    time.sleep(0.5)
            if pending:
                self._assert_children_alive()
                details = ", ".join(
                    f"{name}:{self.agent_port_base + idx}" for idx, name in pending)
                raise RuntimeError(f"Mineflayer spawn readiness timed out ({details}); logs={self._log_dir()}")

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
        # The legacy wrapper tries to create a second Mineflayer bot whenever
        # a request arrives before the first bot has spawned. That synchronous
        # JSPyBridge call is the source of the createBot timeout seen in GRPO.
        # Patch only isolated open-model candidates.
        if (os.environ.get("MASBENCH_OPEN_MODEL_RUN") == "1"
                and (source is None or source != self.root)):
            self._patch_legacy_wrapper()
        for directory in ("data", "result", ".cache"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self._ensure_checker_description()
        endpoints = {name: f"http://{self.host}:{self.agent_port_base + i}"
                     for i, name in enumerate(self.agent_names)}
        (self.root / "data" / "url_prefix.json").write_text(json.dumps(endpoints))

    def _ensure_checker_description(self):
        """Avoid an unrelated external LLM call during Qwen checker startup.

        VillagerAgent lazily invokes OpenAI when a blueprint description cache
        misses the current task.  The score itself is computed from blocks and
        views, and the Qwen prompt already carries the blueprint metadata, so
        an isolated open-model candidate can seed a compact local listing.
        """
        if os.environ.get("MASBENCH_OPEN_MODEL_RUN") != "1":
            return
        task_idx = self.task.get("task_idx")
        blueprint_path = self.root / "data" / "building_blue_print.json"
        cache_path = self.root / "data" / "blueprint_description_all.json"
        if task_idx is None or not blueprint_path.is_file():
            return
        try:
            blueprints = json.loads(blueprint_path.read_text(encoding="utf-8"))
            cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
            if not isinstance(cache, dict):
                cache = {}
            key = f"task_{int(task_idx)}"
            if key in cache:
                return
            blueprint = blueprints[int(task_idx)]
            cache[key] = [
                "material: {name} facing: {facing} position: {position}".format(
                    name=block.get("name", "air"), facing=block.get("facing", "A"),
                    position=block.get("position", []))
                for block in blueprint.get("blocks", [])
                if isinstance(block, Mapping)
                and block.get("name") not in {"air", "water", "lava"}
            ]
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError,
                json.JSONDecodeError):
            # Preserve fail-closed checker semantics if the source is malformed.
            return

    def _patch_legacy_wrapper(self):
        """Disable legacy synchronous bot recreation on tool errors."""
        path = self.root / "env" / "minecraft_server.py"
        try:
            source = path.read_text()
        except OSError as exc:
            raise RuntimeError(f"cannot inspect VillagerAgent wrapper: {path}") from exc
        if "MASBENCH_NO_BOT_RESTART" in source:
            return
        marker = "                  global bot\n"
        start = source.find(marker)
        if start < 0:
            return
        end_marker = "                  return jsonify({'message': f\"Exception in task {func.__name__}: {str(e)}\", 'status': False, \"new_events\": []})\n"
        end = source.find(end_marker, start)
        if end < 0:
            return
        end += len(end_marker)
        replacement = (
            "                  # MASBENCH_NO_BOT_RESTART: keep the original bot alive while\n"
            "                  # its asynchronous login finishes; a second createBot call\n"
            "                  # can deadlock JSPyBridge.\n"
            "                  print(f\"VillagerAgent action {func.__name__} failed: {e}\", flush=True)\n"
            + end_marker
        )
        path.write_text(source[:start] + replacement + source[end:])

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
            if self._server_process is not None and self._server_process.poll() is not None:
                log_path = self._log_dir() / "minecraft_server.log"
                try:
                    tail = log_path.read_text(errors="replace")[-2000:]
                except OSError:
                    tail = ""
                raise RuntimeError(
                    f"minecraft_server exited with code {self._server_process.returncode}; "
                    f"log={log_path}; tail={tail!r}")
            try:
                with socket.create_connection((self.host, self.port), timeout=1):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Minecraft server did not become ready")

    def _reset_world_snapshot(self):
        """Restore one identical world snapshot for every GRPO candidate.

        Each candidate copies the authoritative snapshot into its own server
        world before startup, so a GRPO group never shares mutated state.
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
        server_log_path = self._log_dir() / "minecraft_server.log"
        server_log = server_log_path.open("ab", buffering=0)
        self._log_handles.append(server_log)
        self._server_process = subprocess.Popen(
            ["java", f"-Xms{java_xms}", f"-Xmx{java_xmx}", "-jar", "server.jar", "nogui"],
            cwd=self.server_root, stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT, start_new_session=True)
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
                checker = next((p for p in self.processes
                                if self._process_roles.get(p.pid) == "checker"), None)
                if checker is not None and checker.poll() is not None:
                    return {"evaluation_status": "unassessed", "checks": {},
                            "reason": "checker_exited",
                            "detail": self._process_error(checker),
                            "log_dir": str(self._log_dir()),
                            "real_rollout": True}
                time.sleep(1)
        if raw is None:
            return {"evaluation_status": "unassessed", "checks": {},
                    "reason": "checker_missing_or_empty",
                    "log_dir": str(self._log_dir()),
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
            try:
                os.killpg(self._server_process.pid, 15)
            except (OSError, ProcessLookupError):
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
