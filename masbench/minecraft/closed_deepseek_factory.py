"""DeepSeek OpenAI-compatible factory for closed prompt evaluation."""

from __future__ import annotations

import os

from .api_adapters import OpenAICompatibleAdapter
from .villageragent_real import make_adapter as _make_real_adapter


def make_model():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    return OpenAICompatibleAdapter(
        model=os.environ.get("MASBENCH_CLOSED_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"
        ),
        temperature=float(os.environ.get("MASBENCH_CLOSED_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("MASBENCH_CLOSED_MAX_TOKENS", "256")),
    )


def make_real_adapter(task, episode_id):
    return _make_real_adapter(task, episode_id)
