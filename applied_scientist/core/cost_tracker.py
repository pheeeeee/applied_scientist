from __future__ import annotations

import threading
from dataclasses import dataclass

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


@dataclass
class AgentCost:
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimated dollar cost based on model pricing."""
        in_price, out_price = MODEL_PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000


class CostTracker:
    """Per-agent token and cost tracking."""

    def __init__(self):
        self._agents: dict[str, AgentCost] = {}
        self._lock = threading.Lock()

    def register(self, agent: str, model: str) -> None:
        """Register an agent with its model for cost calculation."""
        with self._lock:
            self._agents[agent] = AgentCost(agent=agent, model=model)

    def record(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for an agent. Thread-safe."""
        with self._lock:
            if agent not in self._agents:
                raise ValueError(f"Agent '{agent}' not registered. Call register() first.")
            self._agents[agent].input_tokens += input_tokens
            self._agents[agent].output_tokens += output_tokens

    def get_agent_cost(self, agent: str) -> AgentCost | None:
        """Return cost for a specific agent."""
        with self._lock:
            return self._agents.get(agent)

    def get_total_cost(self) -> float:
        """Return total estimated cost across all agents."""
        with self._lock:
            return sum(ac.estimated_cost for ac in self._agents.values())

    def get_summary(self) -> str:
        """Return human-readable cost summary table."""
        with self._lock:
            lines = [f"{'Agent':<15} {'Model':<30} {'Input':<12} {'Output':<12} {'Cost':<10}"]
            lines.append("-" * 80)
            for ac in self._agents.values():
                lines.append(f"{ac.agent:<15} {ac.model:<30} "
                             f"{ac.input_tokens:<12,} {ac.output_tokens:<12,} "
                             f"${ac.estimated_cost:.4f}")
            lines.append("-" * 80)
            total = sum(ac.estimated_cost for ac in self._agents.values())
            lines.append(f"{'TOTAL':<57} ${total:.4f}")
            return "\n".join(lines)
