from __future__ import annotations

import os
import queue as queue_mod
import time

import re
import yaml

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import AgentMessage, PRIORITY_SPEC_REVIEW, PRIORITY_HUMAN_DIRECTIVE
from applied_scientist.core.event_logger import EventLogger
from applied_scientist.core.spec import ExperimentSpec


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

    def _parse_config_string(self, text: str) -> dict:
        """Parse key=value pairs from a string.
        Example: 'lr=0.001 batch_size=32' -> {'lr': 0.001, 'batch_size': 32}
        """
        config = {}
        # Match key=value pairs (value can be number, string, or quoted string)
        pattern = r'(\w+)=(["\']?)([^"\'\s]+)\2'
        for match in re.finditer(pattern, text):
            key, _, value = match.groups()
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                config[key] = value
        return config

    def _generate_spec_name(self, description: str) -> str:
        """Generate a unique spec name from description."""
        # Extract key words, lowercase, join with underscore
        words = re.findall(r'\w+', description.lower())[:4]
        base = '_'.join(words) if words else 'human_exp'
        return f"human_{base}_{int(time.time()) % 10000}"

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

        # Human directive commands - bypass critic, insert directly to queue
        elif user_input.strip().startswith("/try "):
            return self._handle_try_command(user_input)
        elif user_input.strip().startswith("/paper "):
            return self._handle_paper_command(user_input)
        elif user_input.strip().startswith("/config "):
            return self._handle_config_command(user_input)
        elif user_input.strip().startswith("/priority "):
            return self._handle_priority_command(user_input)
        elif user_input.strip().startswith("/load "):
            return self._handle_load_command(user_input)

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

    # -------------------------------------------------------------------------
    # Human directive commands - bypass critic, insert directly to queue
    # -------------------------------------------------------------------------

    def _handle_try_command(self, user_input: str) -> bool:
        """/try <description> - Quick experiment injection with high priority.
        Bypasses critic review and inserts directly to queue.
        Example: /try PPO with lr=0.001 batch_size=32
        """
        # Extract description after /try
        description = user_input.strip()[5:].strip()
        if not description:
            self._output("Usage: /try <description> [key=value ...]")
            return True

        # Parse any config from the description
        training_config = self._parse_config_string(description)

        # Create experiment spec
        spec = ExperimentSpec(
            name=self._generate_spec_name(description),
            description=description,
            source_paper="human directive",
            architecture={},
            why_it_might_work=f"Human requested: {description}",
            task_config={},
            training_config=training_config,
            resource_estimate={},
            confidence="high",  # Force-approved
        )

        # Insert directly to queue with high score (bypasses critic)
        high_score = 100.0
        self.queue.insert(spec, high_score)

        # Log event
        self.system_logger.log("human_directive", {
            "command": "try",
            "spec_name": spec.name,
            "description": description,
            "training_config": training_config,
        })

        self._output(f"Queued: {spec.name} (score: {high_score})")
        return True

    def _handle_priority_command(self, user_input: str) -> bool:
        """/priority <spec_name> - Boost existing spec to run next.
        Example: /priority mappo
        """
        # Extract spec name after /priority
        spec_name = user_input.strip()[10:].strip()
        if not spec_name:
            self._output("Usage: /priority <spec_name>")
            return True

        # Check if spec exists in queue
        if not self.queue.contains(spec_name):
            self._output(f"Spec '{spec_name}' not found in queue.")
            return True

        # Boost to very high score
        boost_score = 100.0
        self.queue.update_score(spec_name, boost_score)

        # Log event
        self.system_logger.log("human_directive", {
            "command": "priority",
            "spec_name": spec_name,
            "new_score": boost_score,
        })

        self._output(f"Boosted '{spec_name}' to score {boost_score}")
        return True

    def _handle_paper_command(self, user_input: str) -> bool:
        """/paper <reference> - Investigate and implement a paper.
        Sends to explorer for analysis. Example: /paper "Attention is All You Need"
        """
        # Extract reference after /paper
        reference = user_input.strip()[7:].strip()
        if not reference:
            self._output("Usage: /paper <reference>")
            return True

        # Send to explorer with high priority
        self.bus.post(AgentMessage(
            sender="orchestrator",
            recipient="explorer",
            type="investigate_paper",
            payload={"reference": reference, "human_requested": True},
            priority=PRIORITY_HUMAN_DIRECTIVE,
        ))

        # Log event
        self.system_logger.log("human_directive", {
            "command": "paper",
            "reference": reference,
        })

        self._output(f"Sent to Explorer: investigate '{reference}'")
        return True

    def _handle_config_command(self, user_input: str) -> bool:
        """/config <spec_name> <changes> - Modify config of queued spec.
        Creates a variant with modified config. Example: /config mappo lr=0.0001
        """
        # Parse: /config <spec_name> <key=value pairs>
        parts = user_input.strip()[8:].strip().split(None, 1)
        if len(parts) < 2:
            self._output("Usage: /config <spec_name> <key=value ...>")
            return True

        spec_name = parts[0]
        config_str = parts[1]

        # Find the spec in queue
        all_specs = self.queue.get_all()
        source_spec = None
        for spec, score in all_specs:
            if spec.name.lower() == spec_name.lower():
                source_spec = spec
                break

        if not source_spec:
            self._output(f"Spec '{spec_name}' not found in queue.")
            return True

        # Parse config changes
        config_changes = self._parse_config_string(config_str)
        if not config_changes:
            self._output("No valid config changes found. Use key=value format.")
            return True

        # Create variant spec with merged config
        merged_config = {**source_spec.training_config, **config_changes}
        variant_name = f"{source_spec.name}_variant_{int(time.time()) % 10000}"

        variant_spec = ExperimentSpec(
            name=variant_name,
            description=f"{source_spec.description} (config variant)",
            source_paper=source_spec.source_paper,
            architecture=source_spec.architecture,
            why_it_might_work=f"Config variant of {source_spec.name}: {config_changes}",
            task_config=source_spec.task_config,
            training_config=merged_config,
            resource_estimate=source_spec.resource_estimate,
            confidence="high",  # Force-approved
            control_variant_of=source_spec.name,
            category=source_spec.category,
            tags=source_spec.tags + ["human_config_variant"],
        )

        # Insert with high priority
        high_score = 100.0
        self.queue.insert(variant_spec, high_score)

        # Log event
        self.system_logger.log("human_directive", {
            "command": "config",
            "source_spec": source_spec.name,
            "variant_name": variant_name,
            "config_changes": config_changes,
        })

        self._output(f"Created variant: {variant_name} with {config_changes}")
        return True

    def _handle_load_command(self, user_input: str) -> bool:
        """/load <file.yaml> - Bulk inject experiments from a YAML file.

        Expected YAML format:
        ```yaml
        - score: 100.0
          spec:
            name: ppo_baseline
            description: "PPO baseline experiment"
            source_paper: "human directive"
            architecture: {}
            why_it_might_work: "Testing PPO"
            task_config: {}
            training_config: {lr: 0.001}
            resource_estimate: {}
        - score: 95.0
          spec:
            name: mappo_variant
            ...
        ```
        """
        # Extract file path after /load
        file_path = user_input.strip()[6:].strip()
        if not file_path:
            self._output("Usage: /load <file.yaml>")
            return True

        # Expand ~ and resolve path
        file_path = os.path.expanduser(file_path)
        if not os.path.isabs(file_path):
            # Relative to current working directory
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            self._output(f"File not found: {file_path}")
            return True

        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self._output(f"YAML parse error: {e}")
            return True
        except Exception as e:
            self._output(f"Error reading file: {e}")
            return True

        if not isinstance(data, list):
            self._output("YAML must be a list of {score, spec} entries.")
            return True

        loaded = 0
        errors = []
        for i, entry in enumerate(data):
            try:
                score = float(entry.get("score", 100.0))
                spec_data = entry.get("spec", {})

                # Fill in defaults for missing fields
                spec = ExperimentSpec(
                    name=spec_data.get("name", f"loaded_spec_{i}"),
                    description=spec_data.get("description", "Loaded from file"),
                    source_paper=spec_data.get("source_paper", "human directive"),
                    architecture=spec_data.get("architecture", {}),
                    why_it_might_work=spec_data.get("why_it_might_work", "Human loaded"),
                    task_config=spec_data.get("task_config", {}),
                    training_config=spec_data.get("training_config", {}),
                    resource_estimate=spec_data.get("resource_estimate", {}),
                    confidence=spec_data.get("confidence", "high"),
                    control_variant_of=spec_data.get("control_variant_of"),
                    category=spec_data.get("category", ""),
                    tags=spec_data.get("tags", []) + ["human_loaded"],
                )

                self.queue.insert(spec, score)
                loaded += 1

            except Exception as e:
                errors.append(f"Entry {i}: {e}")

        # Log event
        self.system_logger.log("human_directive", {
            "command": "load",
            "file": file_path,
            "loaded_count": loaded,
            "error_count": len(errors),
        })

        if errors:
            self._output(f"Loaded {loaded} specs. Errors:\n" + "\n".join(errors[:5]))
        else:
            self._output(f"Loaded {loaded} specs from {os.path.basename(file_path)}")

        return True
