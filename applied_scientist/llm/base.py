from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str                          # "system", "user", "assistant", "tool_result"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    content: str                       # text response
    tool_calls: list[dict] | None      # list of {id, name, arguments}
    usage: dict                        # {"input_tokens": int, "output_tokens": int}
    stop_reason: str                   # "end_turn", "tool_use", "max_tokens"


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        """Send messages and return response. Handle retries internally."""

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier string."""

    @property
    @abstractmethod
    def supports_tool_use(self) -> bool:
        """Whether this backend supports tool calling."""
