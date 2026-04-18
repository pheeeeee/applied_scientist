from __future__ import annotations

import json
import os
import queue
import uuid
import threading
from dataclasses import dataclass, field
from typing import Callable

from applied_scientist.core.utils import atomic_write

# Priority constants — higher = processed first
PRIORITY_HUMAN_DIRECTIVE = 150  # Human commands bypass review, run next
PRIORITY_CODE_REVIEW = 100
PRIORITY_PLAN_REVIEW = 90
PRIORITY_SPEC_REVIEW = 80
PRIORITY_INSIGHT_REVIEW = 60
PRIORITY_SUGGESTION = 40
PRIORITY_RERANK = 20
PRIORITY_DEFAULT = 0


@dataclass
class AgentMessage:
    sender: str          # "explorer", "builder", "critic", "orchestrator", "human", "watchdog"
    recipient: str       # target agent name or "all"
    type: str            # message type (see implementation plan for complete list)
    payload: dict        # message-specific data
    priority: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class MessageBus:
    """Thread-safe async inter-agent communication.
    Agents post messages and continue without waiting. Recipients poll their inbox."""

    def __init__(self, state_path: str | None = None):
        """Create bus with per-agent priority queues.
        If state_path given, persist pending messages for crash recovery."""
        self._state_path = state_path
        self._lock = threading.Lock()
        self._queues: dict[str, queue.PriorityQueue] = {}
        self._observers: list[Callable[[AgentMessage], None]] = []
        if state_path and os.path.exists(state_path):
            self._load_state(state_path)

    def add_observer(self, callback: Callable[[AgentMessage], None]) -> None:
        """Register a callback invoked on every post().
        Callback receives an AgentMessage. Must be fast/non-blocking.
        Called OUTSIDE the bus lock — safe to call bus.post() from callback."""
        with self._lock:
            self._observers.append(callback)

    def post(self, message: AgentMessage) -> None:
        """Post message to recipient's queue. Thread-safe. Non-blocking."""
        with self._lock:
            if message.recipient not in self._queues:
                self._queues[message.recipient] = queue.PriorityQueue()
            # Negate priority so higher priority = dequeued first
            self._queues[message.recipient].put((-message.priority, message.id, message))
            if self._state_path:
                self._persist()
            observers = list(self._observers)  # snapshot under lock

        # Invoke observers OUTSIDE the lock to prevent deadlock
        for obs in observers:
            try:
                obs(message)
            except Exception:
                pass  # Never let observer errors break message delivery

    def poll(self, agent: str, timeout: float = 0.1) -> AgentMessage | None:
        """Non-blocking poll. Returns highest-priority message for agent,
        or None if inbox is empty (after brief timeout)."""
        with self._lock:
            if agent not in self._queues:
                return None
            q = self._queues[agent]
        try:
            _, _, message = q.get(timeout=timeout)
            if self._state_path:
                with self._lock:
                    self._persist()
            return message
        except queue.Empty:
            return None

    def drain(self, agent: str) -> list[AgentMessage]:
        """Return all pending messages for agent, sorted by priority. Non-blocking."""
        messages = []
        with self._lock:
            if agent not in self._queues:
                return []
            q = self._queues[agent]
            while not q.empty():
                try:
                    _, _, msg = q.get_nowait()
                    messages.append(msg)
                except queue.Empty:
                    break
            if self._state_path:
                self._persist()
        return messages

    def _persist(self) -> None:
        """Persist all pending messages to state_path (atomic write)."""
        data = {}
        for agent, q in self._queues.items():
            items = list(q.queue)
            data[agent] = [{"sender": m.sender, "recipient": m.recipient,
                            "type": m.type, "payload": m.payload,
                            "priority": m.priority, "id": m.id}
                           for _, _, m in items]
        atomic_write(self._state_path, json.dumps(data, indent=2))

    def _load_state(self, state_path: str) -> None:
        """Load pending messages from state file."""
        with open(state_path) as f:
            data = json.load(f)
        for agent, messages in data.items():
            self._queues[agent] = queue.PriorityQueue()
            for m in messages:
                msg = AgentMessage(**m)
                self._queues[agent].put((-msg.priority, msg.id, msg))

    @classmethod
    def from_state(cls, state_path: str) -> MessageBus:
        """Recover bus from persisted state file."""
        return cls(state_path=state_path)
