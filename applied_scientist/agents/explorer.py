from __future__ import annotations

import glob
import os
import time

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import AgentMessage, PRIORITY_SPEC_REVIEW
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.event_logger import EventLogger


class ExplorerAgent(BaseAgent):
    """Continuous literature search, spec writing, scope advancement."""

    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 knowledge_base, results_tracker, priority_queue,
                 explorer_logger: EventLogger, config):
        super().__init__("explorer", llm, tools, system_prompt, message_bus, cost_tracker)
        self.kb = knowledge_base
        self.results = results_tracker
        self.queue = priority_queue
        self.explorer_logger = explorer_logger
        self.config = config
        self.current_round = 1
        self.paused = False
        self._consecutive_empty_rounds = 0
        self._pending_revisions: list[dict] = []
        self._last_draft_check_time: float = 0

    def run(self):
        """Main loop. Runs in its own thread. Never returns unless stopped."""
        while not self._stopped:
            try:
                self._handle_inbox()

                queue_depth = self._get_queue_depth()
                if queue_depth >= self.config.system.queue_throttle_high:
                    self._alert("throttled", f"Queue has {queue_depth} specs. Pausing.")
                    self._wait_until_queue_below(self.config.system.queue_throttle_low)
                    continue

                if self.paused:
                    time.sleep(5)
                    continue

                if self._pending_revisions:
                    self._revise_spec(self._pending_revisions.pop(0))
                    self.reset_conversation()
                    continue

                context = self._build_context()
                self.chat(
                    f"## Current Context\n{context}\n\n"
                    f"## Scope Round: {self.current_round}\n"
                    f"Find a relevant architecture and write an experiment spec. "
                    f"Use search_papers and read_paper tools to find papers, "
                    f"then write the spec YAML to configs/drafts/ using write_file.\n\n"
                    f"IMPORTANT: Before choosing a spec name, check for collisions. "
                    f"Use list_directory on configs/drafts/, configs/experiments/, "
                    f"and results/ to see existing names."
                )

                drafts = self._check_for_new_drafts()
                for draft_path in drafts:
                    spec = ExperimentSpec.from_yaml(draft_path)
                    deduped_name = self._dedup_spec_name(spec.name)
                    if deduped_name != spec.name:
                        spec.name = deduped_name
                        new_path = os.path.join(os.path.dirname(draft_path),
                                                f"{deduped_name}.yaml")
                        spec.to_yaml(new_path)
                        os.remove(draft_path)
                        draft_path = new_path

                    self.bus.post(AgentMessage(
                        sender="explorer", recipient="critic",
                        type="spec_review_request",
                        payload={"draft_path": draft_path, "round": 1},
                        priority=PRIORITY_SPEC_REVIEW,
                    ))

                if not drafts:
                    self._consecutive_empty_rounds += 1
                else:
                    self._consecutive_empty_rounds = 0
                self._maybe_advance_scope()
                self.reset_conversation()

            except Exception as e:
                self._alert("error", f"Explorer error: {e}")
                self.reset_conversation()
                time.sleep(10)

    def _handle_inbox(self):
        """Process all pending messages from other agents."""
        for msg in self._process_inbox():
            if msg.type == "spec_approved":
                pass
            elif msg.type == "spec_rejected":
                self._pending_revisions.append(msg.payload)
            elif msg.type == "suggestion_for_spec":
                suggestion = msg.payload.get("suggestion", "")
                self._pending_revisions.append({
                    "draft_path": None,
                    "feedback": f"Builder suggestion for new spec:\n{suggestion}",
                    "round": 0,
                    "is_suggestion": True,
                })
            elif msg.type == "investigate_paper":
                reference = msg.payload.get("reference", "")
                self.chat(
                    f"## Investigate this reference\n\n{reference}\n\n"
                    f"Use search_papers and read_paper to find and read this paper. "
                    f"If relevant, write an experiment spec to configs/drafts/."
                )
                self._log_search(reference, 0)
                self.reset_conversation()
            elif msg.type == "clarification_request":
                self._handle_clarification(msg)
            elif msg.type == "command":
                action = msg.payload.get("action", "")
                if action == "pause":
                    self.paused = True
                elif action == "resume":
                    self.paused = False

    def _revise_spec(self, payload: dict):
        """Revise a rejected spec or draft from suggestion."""
        draft_path = payload.get("draft_path")
        feedback = payload.get("feedback", "")
        round_num = payload.get("round", 1) + 1
        is_suggestion = payload.get("is_suggestion", False)

        if is_suggestion:
            context = self._build_context()
            self.chat(
                f"## Draft a new spec from this suggestion\n\n{feedback}\n\n"
                f"## Current Context\n{context}\n\n"
                f"Write the spec YAML to configs/drafts/ using write_file."
            )
            return

        if round_num > self.config.system.max_spec_review_rounds:
            return
        self.chat(f"Revise the spec at {draft_path} based on this feedback:\n{feedback}")
        self.bus.post(AgentMessage(
            sender="explorer", recipient="critic",
            type="spec_review_request",
            payload={"draft_path": draft_path, "round": round_num},
            priority=PRIORITY_SPEC_REVIEW,
        ))

    def _build_context(self) -> str:
        return (
            f"## Task\n{self.config.task.get('domain_context', '')}\n\n"
            f"## Knowledge Base Synthesis\n{self.kb.get_synthesis()}\n\n"
            f"## Current Results\n{self.results.get_summary()}\n\n"
            f"## Completed Experiments\n{', '.join(self.results.get_completed_names())}"
        )

    def _maybe_advance_scope(self):
        if self._consecutive_empty_rounds >= 3:
            self.current_round += 1
            self._consecutive_empty_rounds = 0
            self._alert("scope", f"Advancing to scope round {self.current_round}")
            self.explorer_logger.log(event="scope_advance",
                                     **{"from": self.current_round - 1,
                                        "to": self.current_round})

    def _get_queue_depth(self) -> int:
        return self.queue.depth()

    def _wait_until_queue_below(self, threshold: int):
        while not self._stopped:
            if self.queue.depth() < threshold:
                return
            self._handle_inbox()
            time.sleep(30)

    def _check_for_new_drafts(self) -> list[str]:
        drafts_dir = os.path.join(self.config.paths.configs, "drafts")
        if not os.path.isdir(drafts_dir):
            return []
        new_drafts = []
        for path in glob.glob(os.path.join(drafts_dir, "*.yaml")):
            mtime = os.path.getmtime(path)
            if mtime > self._last_draft_check_time:
                new_drafts.append(path)
        self._last_draft_check_time = time.time()
        return new_drafts

    def _handle_clarification(self, msg: AgentMessage):
        question = msg.payload.get("question", "")
        spec_name = msg.payload.get("spec_name", "")
        source_paper = msg.payload.get("source_paper", "")
        context = self._build_context()
        answer = self.chat(
            f"## Clarification request from Builder\n\n"
            f"**Experiment:** {spec_name}\n"
            f"**Source paper:** {source_paper}\n"
            f"**Question:** {question}\n\n"
            f"## Context\n{context}\n\n"
            f"Answer the Builder's question using your knowledge of the paper and domain."
        )
        self.bus.post(AgentMessage(
            sender="explorer", recipient="builder",
            type="clarification_response",
            payload={"spec_name": spec_name, "answer": answer},
        ))
        self.reset_conversation()

    def _dedup_spec_name(self, base_name: str) -> str:
        existing_names: set[str] = set()
        drafts_dir = os.path.join(self.config.paths.configs, "drafts")
        if os.path.isdir(drafts_dir):
            for f in os.listdir(drafts_dir):
                if f.endswith(".yaml"):
                    existing_names.add(f[:-5])
        experiments_dir = os.path.join(self.config.paths.configs, "experiments")
        if os.path.isdir(experiments_dir):
            for f in os.listdir(experiments_dir):
                if f.endswith(".yaml"):
                    existing_names.add(f[:-5])
        existing_names.update(self.results.get_completed_names())

        if base_name not in existing_names:
            return base_name
        version = 2
        while f"{base_name}_v{version}" in existing_names:
            version += 1
        return f"{base_name}_v{version}"

    def _log_search(self, query: str, results_count: int):
        self.explorer_logger.log(event="search", query=query,
                                 results_count=results_count,
                                 round=self.current_round)

    def _log_paper_read(self, title: str, authors: str, url: str, relevant: bool,
                        summary: str | None = None, key_insights: str | None = None,
                        limitations: str | None = None, relevance: str | None = None,
                        spec_drafted: str | None = None, reason: str | None = None):
        self.explorer_logger.log(
            event="paper_read", title=title, authors=authors, url=url,
            relevant=relevant, summary=summary, key_insights=key_insights,
            limitations=limitations, relevance=relevance,
            spec_drafted=spec_drafted, reason=reason,
            round=self.current_round,
        )
