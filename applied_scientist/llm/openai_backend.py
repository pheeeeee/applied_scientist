from __future__ import annotations

import json
import time

import openai

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse


class OpenAIBackend(LLMBackend):
    """GPT via OpenAI SDK."""

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Convert LLMMessage list to OpenAI format."""
        converted = []
        for m in messages:
            if m.role == "tool_result":
                converted.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                })
            elif m.role == "assistant" and m.tool_calls:
                msg = {"role": "assistant", "content": m.content or ""}
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in m.tool_calls
                ]
                converted.append(msg)
            else:
                converted.append({"role": m.role, "content": m.content})
        return converted

    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        openai_messages = self._convert_messages(messages)

        openai_tools = None
        if tools:
            openai_tools = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}}
                for t in tools
            ]

        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if openai_tools:
                    kwargs["tools"] = openai_tools
                response = self.client.chat.completions.create(**kwargs)
                break
            except (openai.RateLimitError, openai.InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {"id": tc.id, "name": tc.function.name,
                 "arguments": json.loads(tc.function.arguments)}
                for tc in choice.message.tool_calls
            ]

        stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage={"input_tokens": response.usage.prompt_tokens,
                   "output_tokens": response.usage.completion_tokens},
            stop_reason=stop_map.get(choice.finish_reason, choice.finish_reason),
        )

    def get_model_id(self) -> str:
        return self.model

    @property
    def supports_tool_use(self) -> bool:
        return True
