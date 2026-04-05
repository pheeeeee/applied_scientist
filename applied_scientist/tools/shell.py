from __future__ import annotations

import re
import subprocess

from applied_scientist.tools.base import Tool

BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"chmod\s+777",
    r"\.\./\.\./",  # path traversal
]


class ShellTool(Tool):
    name = "run_command"
    description = "Run a shell command in the workspace. Output truncated to 10K chars."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, command: str, timeout: int = 120) -> str:
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"ERROR: Command blocked by security policy: {command}"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=self.workspace,
            )
            output = result.stdout + result.stderr
            return output[:10000]
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"
