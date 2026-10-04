import tempfile
import unittest

from masbench.controllers import CachedModelAdapter, JsonContractAdapter, ModelAdapter


class FakeBackend(ModelAdapter):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, response_schema=None, cache_key=None):
        self.calls += 1
        return "```json\n{\"ok\": true}\n```"


class ControllerApiTests(unittest.TestCase):
    def test_json_contract_and_cache(self):
        backend = FakeBackend()
        adapter = CachedModelAdapter(JsonContractAdapter(backend), tempfile.mkdtemp())
        first = adapter.generate([{"role": "user", "content": "x"}], cache_key="case-1")
        second = adapter.generate([{"role": "user", "content": "x"}], cache_key="case-1")
        self.assertEqual(first, {"ok": True})
        self.assertEqual(second, first)
        self.assertEqual(backend.calls, 1)


if __name__ == "__main__":
    unittest.main()
