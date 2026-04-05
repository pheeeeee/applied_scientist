from __future__ import annotations

import json
import time

import openai

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse
from applied_scientist.llm.openai_backend import OpenAIBackend


class OpenAICompatibleBackend(OpenAIBackend):
    """Any OpenAI-compatible API (Ollama, vLLM, Together AI, Groq, etc.).
    Same as OpenAIBackend but with custom base_url."""

    def __init__(self, model: str, api_key: str, base_url: str, max_retries: int = 3):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
