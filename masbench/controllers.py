"""Controller and model-adapter interfaces shared by all MASbench tasks."""

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path


class Controller(ABC):
    """A controller consumes observations and emits executable action batches."""

    @abstractmethod
    def reset(self, case, env):
        raise NotImplementedError

    @abstractmethod
    def act(self, observations):
        raise NotImplementedError

    def finish(self):
        return None


class ModelAdapter(ABC):
    """Provider-neutral text model interface; implementations may call an API."""

    @abstractmethod
    def generate(self, messages, *, response_schema=None, cache_key=None):
        raise NotImplementedError


class CachedModelAdapter(ModelAdapter):
    """Cache model responses on disk so benchmark reruns do not spend twice."""

    def __init__(self, backend, cache_dir):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages, response_schema, cache_key):
        value = {"messages": messages, "schema": response_schema, "key": cache_key}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def generate(self, messages, *, response_schema=None, cache_key=None):
        key = self._key(messages, response_schema, cache_key)
        path = self.cache_dir / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        result = self.backend.generate(messages, response_schema=response_schema, cache_key=cache_key)
        path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result


class JsonContractAdapter(ModelAdapter):
    """Wrap a text backend and reject malformed/non-JSON responses early."""

    def __init__(self, backend):
        self.backend = backend

    def generate(self, messages, *, response_schema=None, cache_key=None):
        raw = self.backend.generate(messages, response_schema=response_schema, cache_key=cache_key)
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise ValueError("model response must be a dict or JSON string")
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            # Some hosted endpoints append a short explanation or emit a
            # second JSON fragment despite responseMimeType=json. Recover the
            # first complete object so one malformed suffix does not abort a
            # long real-environment rollout.
            start = text.find("{")
            if start < 0:
                return text
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                return text
        if not isinstance(value, dict):
            raise ValueError("model response JSON must be an object")
        return value


class ScriptedController(Controller):
    """Deterministic controller used for environment and metric calibration."""

    def __init__(self, planner):
        self.planner = planner

    def reset(self, case, env):
        self.case, self.env = case, env
        self.planner.reset(case, env)

    def act(self, observations):
        return self.planner.act(observations)

    def finish(self):
        return self.planner.finish()
