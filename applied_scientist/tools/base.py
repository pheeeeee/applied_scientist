from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as seen by the LLM."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for the LLM."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute and return string result."""

    def to_schema(self) -> dict:
        """Convert to LLM tool schema. Works for Anthropic, OpenAI, Gemini formats."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
