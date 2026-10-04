"""Gemini free-tier factory for the closed-model Minecraft prompt pilot."""

from __future__ import annotations

import os

from .api_adapters import GeminiGenerateContentAdapter
from .villageragent_real import make_adapter as _make_real_adapter


def make_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required; export it on the server shell")
    return GeminiGenerateContentAdapter(
        model=os.environ.get("MASBENCH_CLOSED_MODEL", "gemini-3.5-flash-lite"),
        api_key=api_key,
        temperature=float(os.environ.get("MASBENCH_CLOSED_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("MASBENCH_CLOSED_MAX_TOKENS", "256")),
    )


def make_real_adapter(task, episode_id):
    return _make_real_adapter(task, episode_id)
