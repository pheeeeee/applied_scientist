import pytest
from applied_scientist.core.message_bus import MessageBus, AgentMessage
from applied_scientist.core.cost_tracker import CostTracker
from tests.conftest import MockLLM


def test_base_agent_chat():
    from applied_scientist.agents.base import BaseAgent
    llm = MockLLM(["Hello back!"])
    bus = MessageBus()
    cost = CostTracker()
    cost.register("test", "mock-model")
    agent = BaseAgent("test", llm, [], "You are a test agent.", bus, cost)
    response = agent.chat("Hello")
    assert response == "Hello back!"
    assert len(agent.conversation) == 2


def test_agent_unknown_tool_handling():
    from applied_scientist.agents.base import BaseAgent
    from applied_scientist.llm.base import LLMResponse, LLMMessage

    class ToolCallLLM(MockLLM):
        def __init__(self):
            self.call_count = 0
            self.calls = []
        def complete(self, messages, tools=None, **kwargs):
            self.calls.append(messages)
            self.call_count += 1
            if self.call_count == 1:
                return LLMResponse(
                    content="", tool_calls=[{"id": "1", "name": "nonexistent", "arguments": {}}],
                    usage={"input_tokens": 10, "output_tokens": 5},
                    stop_reason="tool_use")
            return LLMResponse(
                content="Got error, moving on.",
                tool_calls=None,
                usage={"input_tokens": 10, "output_tokens": 5},
                stop_reason="end_turn")

    bus = MessageBus()
    cost = CostTracker()
    cost.register("test", "mock-model")
    agent = BaseAgent("test", ToolCallLLM(), [], "test", bus, cost)
    result = agent.chat("Try a bad tool")
    assert "Got error" in result


def test_agent_reset_conversation():
    from applied_scientist.agents.base import BaseAgent
    llm = MockLLM()
    bus = MessageBus()
    cost = CostTracker()
    cost.register("test", "mock-model")
    agent = BaseAgent("test", llm, [], "test", bus, cost)
    agent.chat("Hello")
    assert len(agent.conversation) > 0
    agent.reset_conversation()
    assert len(agent.conversation) == 0


def test_agent_inbox():
    from applied_scientist.agents.base import BaseAgent
    bus = MessageBus()
    cost = CostTracker()
    cost.register("test", "mock-model")
    agent = BaseAgent("test", MockLLM(), [], "test", bus, cost)
    bus.post(AgentMessage(sender="other", recipient="test", type="hello", payload={"msg": "hi"}))
    msgs = agent._process_inbox()
    assert len(msgs) == 1
    assert msgs[0].type == "hello"
