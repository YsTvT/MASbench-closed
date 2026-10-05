"""Small dependency-free adapters for common closed-model HTTP APIs.

They implement the repository's provider-neutral ``ModelAdapter`` interface.
Keys are passed explicitly or read from environment variables by the caller;
this module never persists credentials.  Custom providers can implement the
same three-argument ``generate`` method and be injected into the prompt
runner.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..controllers import ModelAdapter


def _post_json(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON, retrying transient provider overloads before failing.

    Free-tier hosted models occasionally return 429/5xx while the endpoint is
    healthy. A single such response must not discard a nearly complete real
    Minecraft episode, so retries are bounded and configurable per run.
    """
    attempts = max(0, int(os.environ.get("MASBENCH_API_RETRIES", "4")))
    backoff = max(0.0, float(os.environ.get("MASBENCH_API_RETRY_BASE", "2")))
    for attempt in range(attempts + 1):
        request = Request(url, data=json.dumps(payload).encode("utf-8"),
                         headers={"content-type": "application/json", **dict(headers)},
                         method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {408, 429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"model API HTTP {exc.code}: {detail[:1000]}") from exc
        except URLError as exc:
            if attempt < attempts:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"model API request failed: {exc.reason}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("model API returned non-JSON response") from exc
    if not isinstance(value, dict):
        raise RuntimeError("model API response must be a JSON object")
    return value


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return ""


class OpenAICompatibleAdapter(ModelAdapter):
    """OpenAI Chat Completions or Responses compatible endpoint."""

    def __init__(self, model: str, api_key: str | None = None, *,
                 base_url: str = "https://api.openai.com/v1/chat/completions",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout: float = 120.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"provider": "openai_compatible", "model": self.model,
                "endpoint": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "n": 1}

    def generate(self, messages, *, response_schema=None, cache_key=None):
        del cache_key
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.base_url.rstrip("/").endswith("/responses"):
            payload = {"model": self.model, "input": list(messages),
                       "temperature": self.temperature, "max_output_tokens": self.max_tokens}
            if response_schema:
                payload["text"] = {"format": {"type": "json_object"}}
            response = _post_json(self.base_url, headers, payload, self.timeout)
            if isinstance(response.get("output_text"), str):
                return response["output_text"]
            output = response.get("output", [])
            chunks = []
            for item in output if isinstance(output, list) else []:
                for part in item.get("content", []) if isinstance(item, Mapping) else []:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            return "".join(chunks)
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "n": 1}
        if os.environ.get("MASBENCH_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}:
            payload["thinking"] = {"type": "disabled"}
        if response_schema:
            payload["response_format"] = {"type": "json_object"}
        response = _post_json(self.base_url, headers, payload, self.timeout)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible response has no choices")
        message = choices[0].get("message", {})
        content = _text_content(message.get("content")) if isinstance(message, Mapping) else ""
        if not content:
            raise RuntimeError("OpenAI-compatible response has empty message content")
        return content


class AnthropicMessagesAdapter(ModelAdapter):
    """Anthropic Messages API adapter with JSON enforced by the prompt."""

    def __init__(self, model: str, api_key: str | None = None, *,
                 base_url: str = "https://api.anthropic.com/v1/messages",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout: float = 120.0, api_version: str = "2023-06-01"):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.api_version = api_version

    @property
    def metadata(self) -> dict[str, Any]:
        return {"provider": "anthropic", "model": self.model,
                "endpoint": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "n": 1}

    def generate(self, messages, *, response_schema=None, cache_key=None):
        del response_schema, cache_key
        system = "\n".join(str(m.get("content", "")) for m in messages
                            if isinstance(m, Mapping) and m.get("role") == "system")
        user_messages = [dict(m) for m in messages
                         if isinstance(m, Mapping) and m.get("role") != "system"]
        payload = {"model": self.model, "messages": user_messages,
                   "max_tokens": self.max_tokens, "temperature": self.temperature}
        if system:
            payload["system"] = system
        response = _post_json(self.base_url,
                              {"x-api-key": self.api_key, "anthropic-version": self.api_version},
                              payload, self.timeout)
        text = _text_content(response.get("content"))
        if not text:
            raise RuntimeError("Anthropic response has empty content")
        return text


class GeminiGenerateContentAdapter(ModelAdapter):
    """Google Gemini ``generateContent`` adapter."""

    def __init__(self, model: str, api_key: str | None = None, *,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout: float = 120.0):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"provider": "gemini", "model": self.model,
                "endpoint": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "n": 1}

    def generate(self, messages, *, response_schema=None, cache_key=None):
        del cache_key
        contents = []
        system_parts = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            text = str(message.get("content", ""))
            if role == "system":
                system_parts.append(text)
            else:
                contents.append({"role": "model" if role == "assistant" else "user",
                                 "parts": [{"text": text}]})
        payload = {"contents": contents,
                   "generationConfig": {"temperature": self.temperature,
                                        "maxOutputTokens": self.max_tokens}}
        if response_schema:
            payload["generationConfig"].update({"responseMimeType": "application/json"})
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        response = _post_json(url, {}, payload, self.timeout)
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response has no candidates")
        content = candidates[0].get("content", {})
        text = _text_content(content.get("parts")) if isinstance(content, Mapping) else ""
        if not text:
            raise RuntimeError("Gemini response has empty content")
        return text
