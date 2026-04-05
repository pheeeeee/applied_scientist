from __future__ import annotations

import json
import time

import anthropic

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse


class AnthropicBackend(LLMBackend):
    """Claude via Anthropic SDK. System prompt extracted as top-level parameter."""

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        # Extract system message (Anthropic uses top-level system param)
        system_prompt = None
        filtered = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "tool_result":
                filtered.append({
                    "role": "user",
                    "content": [{"type": "tool_result",
                                 "tool_use_id": m.tool_call_id,
                                 "content": m.content}],
                })
            elif m.role == "assistant" and m.tool_calls:
                content = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                filtered.append({"role": "assistant", "content": content})
            else:
                filtered.append({"role": m.role, "content": m.content})

        anthropic_tools = None
        if tools:
            anthropic_tools = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["input_schema"]} for t in tools]

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=filtered,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if system_prompt:
                    kwargs["system"] = system_prompt
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools
                response = self.client.messages.create(**kwargs)
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        # Parse response
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls if tool_calls else None,
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens},
            stop_reason=response.stop_reason,
        )

    def get_model_id(self) -> str:
        return self.model

    @property
    def supports_tool_use(self) -> bool:
        return True
