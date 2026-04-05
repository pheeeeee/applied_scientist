from __future__ import annotations

import json
import time

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse


class GeminiBackend(LLMBackend):
    """Gemini via Google GenAI SDK."""

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model
        self.max_retries = max_retries
        self._model = None  # Lazy init with system instruction
        self._call_id_to_name: dict[str, str] = {}  # track tool call IDs to function names

    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        genai = self._genai

        # Extract system message
        system_instruction = None
        contents = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            elif m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                parts = []
                if m.content:
                    parts.append({"text": m.content})
                if m.tool_calls:
                    for tc in m.tool_calls:
                        parts.append(genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=tc["name"], args=tc["arguments"])))
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool_result":
                func_name = self._call_id_to_name.get(m.tool_call_id, "unknown")
                contents.append({"role": "user", "parts": [
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=func_name, response={"result": m.content}))
                ]})

        # Create model with system instruction
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_instruction,
        )

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            declarations = []
            for t in tools:
                declarations.append(genai.protos.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["input_schema"],
                ))
            gemini_tools = [genai.protos.Tool(function_declarations=declarations)]

        config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(
                    contents,
                    tools=gemini_tools,
                    generation_config=config,
                )
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        # Parse response
        text_parts = []
        tool_calls = []
        if response.candidates:
            for i, part in enumerate(response.candidates[0].content.parts):
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = f"gemini_{fc.name}_{i}"
                    self._call_id_to_name[call_id] = fc.name
                    tool_calls.append({
                        "id": call_id,
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage["input_tokens"] = getattr(response.usage_metadata, "prompt_token_count", 0)
            usage["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", 0)

        stop_reason = "end_turn" if not tool_calls else "tool_use"

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            stop_reason=stop_reason,
        )

    def get_model_id(self) -> str:
        return self.model_name

    @property
    def supports_tool_use(self) -> bool:
        return True
