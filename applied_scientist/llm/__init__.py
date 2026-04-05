"""LLM backends (pluggable). Registry provides get_backend(config)."""

from __future__ import annotations

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse


def get_backend(config: dict) -> LLMBackend:
    """Instantiate backend from config dict with keys: backend, model, api_key, (base_url).
    Lazy imports to avoid requiring all SDKs."""
    backend_name = config["backend"]
    kwargs = {k: v for k, v in config.items() if k != "backend"}

    if backend_name == "anthropic":
        from applied_scientist.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(**kwargs)
    elif backend_name == "openai":
        from applied_scientist.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(**kwargs)
    elif backend_name == "gemini":
        from applied_scientist.llm.gemini_backend import GeminiBackend
        return GeminiBackend(**kwargs)
    elif backend_name == "openai_compatible":
        from applied_scientist.llm.openai_compatible import OpenAICompatibleBackend
        return OpenAICompatibleBackend(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name}. "
                         f"Available: anthropic, openai, gemini, openai_compatible")
