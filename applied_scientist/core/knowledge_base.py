from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

from applied_scientist.core.utils import atomic_write


class KnowledgeBase:
    """Manages knowledge.md with synthesis pinned at top."""

    def __init__(self, path: str):
        """Load existing knowledge.md or create empty."""
        self._path = path
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path) as f:
                self._content = f.read()
        else:
            self._content = "# Knowledge Base\n\n## Synthesis\n\n## Insights\n"

    def add_insight(self, title: str, body: str) -> None:
        """Append a Critic-approved insight to the individual section. Atomic write."""
        with self._lock:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            entry = f"\n### {title} ({timestamp})\n\n{body}\n"
            if "## Insights" in self._content:
                self._content = self._content + entry
            else:
                self._content += f"\n## Insights\n{entry}"
            atomic_write(self._path, self._content)

    def add_synthesis(self, category: str, body: str) -> None:
        """Add or update a synthesis entry (pinned at top). Atomic write."""
        with self._lock:
            marker_start = f"### {category}\n"
            marker_end = "\n### "
            if marker_start in self._content:
                start = self._content.index(marker_start)
                rest = self._content[start + len(marker_start):]
                end_idx = rest.find(marker_end)
                if end_idx == -1 or rest.find("## Insights") < end_idx:
                    end_idx = rest.find("## Insights")
                if end_idx == -1:
                    end_idx = len(rest)
                self._content = (self._content[:start] + marker_start +
                                 body + "\n" + rest[end_idx:])
            else:
                insert_pos = self._content.find("## Insights")
                if insert_pos == -1:
                    self._content += f"\n### {category}\n{body}\n"
                else:
                    self._content = (self._content[:insert_pos] +
                                     f"### {category}\n{body}\n\n" +
                                     self._content[insert_pos:])
            atomic_write(self._path, self._content)

    def get_synthesis(self) -> str:
        """Return all synthesis sections as text."""
        with self._lock:
            idx = self._content.find("## Insights")
            if idx == -1:
                return self._content
            return self._content[:idx].strip()

    def get_full(self) -> str:
        """Return the entire knowledge base."""
        with self._lock:
            return self._content

    def get_recent(self, n: int = 5) -> str:
        """Return the n most recent insights."""
        with self._lock:
            insights_idx = self._content.find("## Insights")
            if insights_idx == -1:
                return ""
            insights_text = self._content[insights_idx:]
            insight_parts = insights_text.split("### ")[1:]
            recent = insight_parts[-n:] if len(insight_parts) > n else insight_parts
            return "\n\n### ".join([""] + recent).strip()
