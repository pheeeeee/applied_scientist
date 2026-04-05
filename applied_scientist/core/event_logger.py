from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class EventLogger:
    """Append-only JSONL logger. Thread-safe via atomic append."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, **fields) -> None:
        """Append one JSON line with auto-timestamp."""
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        line = json.dumps(entry, default=str) + "\n"
        with open(self.path, "a") as f:
            f.write(line)

    def read_all(self) -> list[dict]:
        """Read all entries (for debugging/display)."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if raw_line:
                    entries.append(json.loads(raw_line))
        return entries
