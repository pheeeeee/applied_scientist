from __future__ import annotations

import json
import os
import time

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import (
    AgentMessage, PRIORITY_CODE_REVIEW, PRIORITY_SPEC_REVIEW,
    PRIORITY_INSIGHT_REVIEW, PRIORITY_SUGGESTION, PRIORITY_RERANK,
)
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.event_logger import EventLogger
from applied_scientist.core.utils import normalize_name


class CriticAgent(BaseAgent):
    """Inbox-based review agent: structured decisions, dedup, ranking."""

    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 priority_queue, knowledge_base, results_tracker,
                 review_logger: EventLogger, system_logger: EventLogger, config):
        super().__init__("critic", llm, tools, system_prompt, message_bus, cost_tracker)
        self.queue = priority_queue
        self.kb = knowledge_base
        self.results = results_tracker
        self.review_logger = review_logger
        self.system_logger = system_logger
        self.config = config

    def run(self):
        """Main loop. Processes inbox in priority order."""
        while not self._stopped:
            try:
                messages = self._process_inbox()
                if not messages:
                    time.sleep(2)
                    continue

                for message in messages:
                    if message.type == "spec_review_request":
                        self._handle_spec_review(message)
                    elif message.type == "code_review_request":
                        self._handle_code_review(message)
                    elif message.type == "insight_review_request":
                        self._handle_insight_review(message)
                    elif message.type == "suggestion_triage_request":
                        self._handle_suggestion_triage(message)
                    elif message.type == "rerank_request":
                        self._handle_rerank(message)
                    elif message.type == "priority_adjust":
                        self._handle_priority_adjust(message)
                    elif message.type == "inject_spec":
                        self._handle_inject_spec(message)

                    self.reset_conversation()

            except Exception as e:
                self._alert("error", f"Critic error: {e}")
                self.reset_conversation()
                time.sleep(5)

    def _handle_spec_review(self, message):
        """Review a draft spec. Uses structured submit_review tool."""
        draft_path = message.payload["draft_path"]
        round_num = message.payload["round"]

        spec = ExperimentSpec.from_yaml(draft_path)
        if self.results.is_completed(spec.name):
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected", feedback="Already completed.")
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": f"Experiment '{spec.name}' already completed."}
            ))
            return
        similar = self.queue.contains_similar(spec.name)
        if similar:
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected",
                                   feedback=f"Similar to '{similar}' in queue.")
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": f"Similar experiment '{similar}' already in queue."}
            ))
            return

        with open(draft_path) as f:
            spec_content = f.read()
        force_approve = round_num >= self.config.system.max_spec_review_rounds

        response = self.chat(
            f"## Review this experiment spec (round {round_num}/"
            f"{self.config.system.max_spec_review_rounds})\n\n"
            f"```yaml\n{spec_content}\n```\n\n"
            f"## Current results context\n{self.results.get_summary()}\n\n"
            f"Check: completeness, accuracy, specificity, resource feasibility, "
            f"control variant needed?\n\n"
            f"{'FINAL ROUND: force-approve with your fixes if still issues.' if force_approve else ''}\n\n"
            f"Use the submit_review tool to submit your decision."
        )

        decision = self._parse_last_tool_result("submit_review")

        if decision and decision.get("verdict") == "approved":
            score = decision.get("score", self._compute_score(spec))
            approved_path = draft_path.replace("/drafts/", "/experiments/")
            os.makedirs(os.path.dirname(approved_path), exist_ok=True)
            os.rename(draft_path, approved_path)
            self.queue.insert(spec, score)
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="approved", score=score)
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_approved",
                payload={"name": spec.name, "score": score}
            ))
            if decision.get("control_variant_needed"):
                self.bus.post(AgentMessage(
                    sender="critic", recipient="explorer",
                    type="spec_approved",
                    payload={"name": spec.name, "score": score,
                             "control_request": decision.get("control_description", "")}
                ))
        else:
            feedback = decision.get("feedback", response) if decision else response
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected", feedback=str(feedback)[:500])
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": feedback}
            ))

    def _handle_code_review(self, message):
        """Review structural code change. Uses structured submit_code_review tool."""
        diff = message.payload["diff"]
        spec_name = message.payload["spec_name"]
        self.chat(
            f"## Code review for experiment: {spec_name}\n\n"
            f"```diff\n{diff}\n```\n\n"
            f"Check: implementation matches spec, no obvious bugs, correct API usage.\n"
            f"Use the submit_code_review tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_code_review")
        verdict = decision.get("verdict", "approved") if decision else "approved"
        feedback = decision.get("feedback", "") if decision else ""
        self.review_logger.log(type="code_review", spec_name=spec_name,
                               verdict=verdict, feedback=str(feedback)[:500])
        self.bus.post(AgentMessage(
            sender="critic", recipient="builder",
            type="code_approved" if verdict == "approved" else "code_rejected",
            payload={"spec_name": spec_name, "feedback": feedback}
        ))

    def _handle_insight_review(self, message):
        """Review a draft insight using structured submit_insight_review tool."""
        insight = message.payload["insight"]
        experiment = message.payload.get("experiment", "unknown")
        self.chat(
            f"## Review this experiment insight for: {experiment}\n\n{insight}\n\n"
            f"## Knowledge base context\n{self.kb.get_synthesis()}\n\n"
            f"Check: over-claiming, unsupported causal claims, missed observations, "
            f"contradictions with prior knowledge.\n\n"
            f"Use the submit_insight_review tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_insight_review")
        if decision and decision.get("verdict") == "approved":
            edited = decision.get("edits", insight)
            self.review_logger.log(type="insight_review", experiment=experiment,
                                   verdict="approved")
            self.bus.post(AgentMessage(
                sender="critic", recipient="builder",
                type="insight_approved",
                payload={"insight": edited, "experiment": experiment}
            ))
        else:
            feedback = decision.get("feedback", "") if decision else ""
            self.review_logger.log(type="insight_review", experiment=experiment,
                                   verdict="rejected", feedback=str(feedback)[:500])
            self.bus.post(AgentMessage(
                sender="critic", recipient="builder",
                type="insight_rejected",
                payload={"feedback": feedback, "experiment": experiment}
            ))

    def _handle_suggestion_triage(self, message):
        """Triage a Builder suggestion using structured submit_triage tool."""
        suggestion = message.payload["suggestion"]
        self.chat(
            f"## Triage this improvement suggestion\n\n{suggestion}\n\n"
            f"Classify as:\n"
            f"- minor: hyperparameter tweak, config-only change\n"
            f"- moderate: meaningful modification, needs new code\n"
            f"- major: novel architectural idea, potential research contribution\n\n"
            f"Use the submit_triage tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_triage")
        level = decision.get("level", "minor") if decision else "minor"

        self.review_logger.log(type="suggestion_triage", level=level,
                               suggestion=str(suggestion)[:200])

        if level == "major":
            ideas_path = os.path.join(self.config.paths.results, "ideas_for_system2.md")
            os.makedirs(os.path.dirname(ideas_path), exist_ok=True)
            with open(ideas_path, "a") as f:
                f.write(f"\n## {str(suggestion)[:80]}\n\n")
                f.write(f"**Triage:** {decision.get('justification', '') if decision else ''}\n\n")
                f.write(f"**Suggested action:** {decision.get('action', '') if decision else ''}\n\n")
                f.write("---\n")

        self.bus.post(AgentMessage(
            sender="critic", recipient="builder",
            type="suggestion_triaged",
            payload={"level": level, "suggestion": suggestion,
                     "response": decision if decision else {}}
        ))

    def _handle_rerank(self, message):
        """Rerank the priority queue based on current results context."""
        context = {"results": self.results.get_all(),
                    "completed": self.results.get_completed_names()}
        self.queue.rerank(
            scoring_fn=lambda spec, ctx: self._compute_score(spec),
            context=context,
        )

    def _handle_priority_adjust(self, message):
        """Adjust priority scores for specs matching a topic."""
        topic = message.payload.get("topic", "")
        boost = message.payload.get("boost", True)
        multiplier = 1.5 if boost else 0.5
        for spec, score in self.queue.get_all():
            if (topic.lower() in spec.category.lower() or
                    topic.lower() in spec.description.lower()):
                self.queue.update_score(spec.name, score * multiplier)
        self.system_logger.log(event="priority_adjust", topic=topic, boost=boost)

    def _handle_inject_spec(self, message):
        """Review a user-injected spec."""
        spec_data = message.payload.get("spec_data")
        draft_path = os.path.join(self.config.paths.configs, "drafts",
                                  f"{spec_data['name']}.yaml")
        os.makedirs(os.path.dirname(draft_path), exist_ok=True)
        spec = ExperimentSpec(**spec_data)
        spec.to_yaml(draft_path)
        self._handle_spec_review(AgentMessage(
            sender="orchestrator", recipient="critic",
            type="spec_review_request",
            payload={"draft_path": draft_path, "round": 1},
            priority=PRIORITY_SPEC_REVIEW,
        ))

    def _compute_score(self, spec: ExperimentSpec) -> float:
        """Score a spec for priority ranking. Higher = run sooner."""
        completed = self.results.get_completed_names()
        performance_score = 1.0
        if spec.category:
            similar_count = sum(1 for name in completed
                              if normalize_name(name).startswith(normalize_name(spec.category)))
            category_count = sum(1 for r in self.results.get_all()
                                if r.description and spec.category in r.description)
        else:
            similar_count = 0
            category_count = 0
        diversity_bonus = 2.0 if similar_count == 0 else 0.5 / max(similar_count, 1)
        information_bonus = 3.0 if category_count == 0 else 1.0 / (category_count + 1)
        return performance_score + diversity_bonus + information_bonus

    def _parse_last_tool_result(self, tool_name: str) -> dict | None:
        """Find the last tool result from a specific tool in conversation.
        Verifies the tool_call_id matches a call to the expected tool_name."""
        for msg in reversed(self.conversation):
            if msg.role == "tool_result" and msg.tool_call_id:
                for prev_msg in self.conversation:
                    if prev_msg.tool_calls:
                        for call in prev_msg.tool_calls:
                            if (call["id"] == msg.tool_call_id and
                                    call["name"] == tool_name):
                                try:
                                    return json.loads(msg.content)
                                except (json.JSONDecodeError, TypeError):
                                    continue
        return None
