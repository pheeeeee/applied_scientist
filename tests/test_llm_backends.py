import pytest
from applied_scientist.llm.base import LLMMessage, LLMResponse


def test_llm_message():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.tool_calls is None


def test_llm_response():
    resp = LLMResponse(
        content="hi", tool_calls=None,
        usage={"input_tokens": 10, "output_tokens": 5},
        stop_reason="end_turn")
    assert resp.content == "hi"
    assert resp.usage["input_tokens"] == 10


def test_tool_call_response():
    resp = LLMResponse(
        content="", tool_calls=[{"id": "1", "name": "read_file", "arguments": {"path": "x"}}],
        usage={"input_tokens": 10, "output_tokens": 5},
        stop_reason="tool_use")
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "read_file"
