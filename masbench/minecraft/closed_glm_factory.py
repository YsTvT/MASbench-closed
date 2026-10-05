"""Z.ai GLM factory for closed prompt-only Minecraft evaluation.

The API key is read only from ``ZAI_API_KEY`` at runtime. It is never written
to an episode, cache, result file or repository.
"""

from __future__ import annotations

import os

from .api_adapters import OpenAICompatibleAdapter
from .villageragent_real import make_adapter as _make_real_adapter


def make_model():
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is required")
    return OpenAICompatibleAdapter(
        model=os.environ.get("MASBENCH_CLOSED_MODEL", "glm-4.5-flash"),
        api_key=api_key,
        base_url=os.environ.get(
            "ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/chat/completions"
        ),
        temperature=float(os.environ.get("MASBENCH_CLOSED_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("MASBENCH_CLOSED_MAX_TOKENS", "256")),
    )


def make_real_adapter(task, episode_id):
    return _make_real_adapter(task, episode_id)
