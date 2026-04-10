from __future__ import annotations

import os
import queue as queue_mod
import time

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import AgentMessage, PRIORITY_SPEC_REVIEW
from applied_scientist.core.event_logger import EventLogger


class OrchestratorAgent(BaseAgent):
    """Natural language interface: interprets user input, routes commands."""

    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 priority_queue, knowledge_base, results_tracker, gpu_pool,
                 system_logger: EventLogger, agents_ref: dict,
                 report_fn=None, io=None):
        super().__init__("orchestrator", llm, tools, system_prompt, message_bus, cost_tracker)
        self.queue = priority_queue
        self.kb = knowledge_base
        self.results = results_tracker
        self.pool = gpu_pool
        self.system_logger = system_logger
        self.agents = agents_ref
        self._report_fn = report_fn
        self.io = io  # None = terminal mode (stdin/stdout)

    def _output(self, text: str):
        """Send text to user. Routes through Slack/Telegram when io is set."""
        if self.io:
            self.io.send(text)
        else:
            print(text)

    def run(self):
        """Main loop. Reads user input, interprets with LLM, routes."""
        while not self._stopped:
            try:
                self._display_alerts()

                if self.io:
                    # Remote mode: use timeout so we loop back to display alerts
                    try:
                        user_input = self.io.receive(timeout=2.0)
                    except queue_mod.Empty:
                        continue  # no message — loop back to check alerts
                else:
                    # Terminal mode: blocking input with prompt
                    user_input = input("> ")

            except EOFError:
                continue

            if not user_input.strip():
                continue

            if user_input.strip().startswith("@"):
                self._handle_direct_message(user_input)
                continue

            if self._handle_builtin(user_input):
                continue

            status = self._build_status()
            response = self.chat(
                f"## System Status\n{status}\n\n"
                f"## User Message\n{user_input}\n\n"
                f"Interpret the user's intent. If it's a query, answer from status. "
                f"If it's a command, state what action you're taking."
            )
            self._output(response)
            self._route_command(user_input, response)
            self.reset_conversation()

    def _handle_builtin(self, user_input: str) -> bool:
        """Handle commands that don't need LLM interpretation."""
        cmd = user_input.strip().lower()
        if cmd == "status":
            self._output(self._build_status())
            return True
        elif cmd == "results":
            self._output(self.results.get_summary())
            return True
        elif cmd == "knowledge":
            self._output(self.kb.get_synthesis())
            return True
        elif cmd == "queue":
            items = self.queue.peek(10)
            if items:
                lines = [f"  {score:.1f}  {spec.name}: {spec.description}"
                         for spec, score in items]
                self._output("\n".join(lines))
            else:
                self._output("Queue is empty.")
            return True
        elif cmd == "cost":
            self._output(self.cost.get_summary())
            return True
        elif cmd == "report":
            if self._report_fn:
                path = self._report_fn()
                if path:
                    self._output(f"Report saved to {path}.md / .json")
                else:
                    self._output("Report generation failed. Check system_events.jsonl.")
            else:
                self._output("Report function not available.")
            return True
        elif cmd == "pause":
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="builder",
                type="command", payload={"action": "pause"}
            ))
            self._output("Builder paused. Running experiments will finish.")
            return True
        elif cmd == "resume":
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="builder",
                type="command", payload={"action": "resume"}
            ))
            self._output("Builder resumed.")
            return True
        elif cmd.startswith("gpus "):
            try:
                n = int(cmd.split()[1])
                self.pool.resize(n)
                self._output(f"GPU pool resized to {n} slots.")
            except (ValueError, IndexError):
                self._output("Usage: gpus N")
            return True
        return False

    def _handle_direct_message(self, user_input: str):
        """Handle @agent messages via message bus (thread-safe).
        Routes through the bus instead of calling agent.chat() directly,
        avoiding concurrent access to the agent's conversation list."""
        parts = user_input.strip().split(None, 1)
        target = parts[0][1:].lower()
        message = parts[1] if len(parts) > 1 else ""
        valid_agents = {"explorer", "critic", "builder"}
        if target in valid_agents:
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient=target,
                type="command",
                payload={"action": "user_message", "message": message},
            ))
            self._output(f"Message sent to {target.title()}.")
        else:
            self._output(f"Unknown agent: {target}. Available: explorer, critic, builder")

    def _route_command(self, user_input: str, response: str):
        """Parse LLM response and dispatch commands to agents."""
        lower_input = user_input.lower()
        lower_response = response.lower()

        if any(w in lower_response for w in ["prioritiz", "boost", "focus on"]):
            topic = user_input
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="priority_adjust",
                payload={"topic": topic, "boost": True},
            ))

        elif any(w in lower_response for w in ["deprioritiz", "lower", "skip"]):
            topic = user_input
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="priority_adjust",
                payload={"topic": topic, "boost": False},
            ))

        if any(w in lower_response for w in ["relay", "investigate", "paper", "look into"]):
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="explorer",
                type="investigate_paper",
                payload={"reference": user_input},
            ))

        if any(w in lower_response for w in ["inject", "try this", "add spec"]):
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="inject_spec",
                payload={"spec_data": {
                    "name": "user_injected",
                    "description": user_input,
                    "source_paper": "user suggestion",
                    "architecture": {},
                    "why_it_might_work": user_input,
                    "task_config": {},
                    "training_config": {},
                    "resource_estimate": {},
                }},
                priority=PRIORITY_SPEC_REVIEW,
            ))

    def _build_status(self) -> str:
        """Build a status summary string."""
        best = self.results.get_best(1)
        best_str = (f"{best[0].name} ({best[0].metric_value})"
                    if best and best[0].metric_value is not None
                    else "none yet")
        return (
            f"GPU Pool:\n{self.pool.get_status_summary()}\n\n"
            f"Queue: {self.queue.depth()} pending specs\n\n"
            f"Results: {len(self.results.get_all())} experiments completed\n"
            f"Best: {best_str}\n\n"
            f"Cost: ${self.cost.get_total_cost():.2f}\n\n"
            f"Knowledge synthesis:\n{self.kb.get_synthesis()[:500]}"
        )

    def _display_alerts(self):
        """Display any pending alerts from other agents."""
        for msg in self._process_inbox():
            if msg.type == "alert":
                alert_text = f"[ALERT] {msg.payload.get('message', '')}"
                if self.io:
                    self.io.send(alert_text)
                else:
                    print(f"\n  {alert_text}\n> ", end="", flush=True)
