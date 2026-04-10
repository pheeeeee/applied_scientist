from __future__ import annotations

import time

from applied_scientist.llm.base import LLMBackend, LLMMessage, LLMResponse
from applied_scientist.tools.base import Tool
from applied_scientist.core.message_bus import MessageBus, AgentMessage
from applied_scientist.core.cost_tracker import CostTracker


class BaseAgent:
    """Base agent with LLM conversation loop, tool dispatch, and context management."""

    def __init__(self, role: str, llm: LLMBackend, tools: list[Tool],
                 system_prompt: str, message_bus: MessageBus,
                 cost_tracker: CostTracker):
        self.role = role
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]
        self.system_prompt = system_prompt
        self.bus = message_bus
        self.cost = cost_tracker
        self.conversation: list[LLMMessage] = []
        self._stopped = False

    def chat(self, user_message: str) -> str:
        """Send a message and run the full tool-use loop until the agent
        produces a final text response. Handles multi-turn tool calls."""
        self._maybe_trim()
        self.conversation.append(LLMMessage(role="user", content=user_message))

        while True:
            response = self.llm.complete(
                messages=[LLMMessage(role="system", content=self.system_prompt)]
                        + self.conversation,
                tools=self.tool_schemas if self.tools else None,
            )
            self.cost.record(self.role,
                             response.usage["input_tokens"],
                             response.usage["output_tokens"])

            if response.tool_calls:
                self.conversation.append(LLMMessage(
                    role="assistant", content=response.content,
                    tool_calls=response.tool_calls))
                for call in response.tool_calls:
                    tool_name = call["name"]
                    if tool_name not in self.tools:
                        result = f"ERROR: Unknown tool '{tool_name}'. Available: {', '.join(self.tools.keys())}"
                    else:
                        try:
                            result = self.tools[tool_name].execute(**call["arguments"])
                        except Exception as e:
                            result = f"ERROR: Tool '{tool_name}' failed: {e}"
                    self.conversation.append(LLMMessage(
                        role="tool_result", content=result,
                        tool_call_id=call["id"]))
            else:
                self.conversation.append(LLMMessage(
                    role="assistant", content=response.content))
                return response.content

    def reset_conversation(self):
        """Clear conversation history to manage context window."""
        self.conversation = []

    def _maybe_trim(self):
        """Auto-trim if approaching context limit (80% of max)."""
        estimated_tokens = sum(len(m.content or "") // 4 for m in self.conversation)
        max_context = getattr(self, 'max_context_tokens', 180_000)
        if estimated_tokens > max_context * 0.8:
            summary_prompt = (
                "Summarize the key findings and decisions from this conversation "
                "in a concise paragraph. Focus on facts, not process."
            )
            summary = self.llm.complete([
                LLMMessage(role="system", content="You are a summarizer."),
                LLMMessage(role="user", content=summary_prompt + "\n\n" +
                           "\n".join(m.content for m in self.conversation[-20:]))
            ])
            self.cost.record(self.role,
                             summary.usage["input_tokens"],
                             summary.usage["output_tokens"])
            self.conversation = [
                LLMMessage(role="user",
                           content=f"## Prior context summary\n{summary.content}")
            ]

    def _process_inbox(self) -> list[AgentMessage]:
        """Drain all pending messages from inbox. Non-blocking."""
        return self.bus.drain(self.role)

    def _alert(self, alert_type: str, message: str):
        """Send alert to Orchestrator for display."""
        self.bus.post(AgentMessage(
            sender=self.role, recipient="orchestrator",
            type="alert", payload={"alert_type": alert_type, "message": message}
        ))

    def _wait_for_message(self, msg_type: str | tuple[str, ...] | None = None,
                          match_fn=None,
                          timeout: float = 300.0) -> AgentMessage | None:
        """Block until a matching message arrives.

        Args:
            msg_type: Single type string, tuple of types, or None (match any type).
            match_fn: Optional callable(AgentMessage) -> bool for additional filtering.
            timeout: Max seconds to wait.

        Other messages that arrive while waiting are re-injected into the bus
        so they are not lost. Returns the matched message, or None on timeout.
        """
        deadline = time.time() + timeout
        buffered: list[AgentMessage] = []

        if isinstance(msg_type, str):
            msg_type = (msg_type,)

        while time.time() < deadline and not self._stopped:
            msg = self.bus.poll(self.role, timeout=2.0)
            if msg is None:
                continue
            type_match = msg_type is None or msg.type in msg_type
            fn_match = match_fn is None or match_fn(msg)
            if type_match and fn_match:
                for buffered_msg in buffered:
                    self.bus.post(buffered_msg)
                return msg
            else:
                buffered.append(msg)

        for buffered_msg in buffered:
            self.bus.post(buffered_msg)
        return None
