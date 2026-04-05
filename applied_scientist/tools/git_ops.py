from __future__ import annotations

import subprocess

from applied_scientist.tools.base import Tool


class GitCommit(Tool):
    name = "git_commit"
    description = "Stage and commit files in the workspace git repo."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "Files to stage (default: all changed)"},
        },
        "required": ["message"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, message: str, files: list[str] = None) -> str:
        if files:
            for f in files:
                subprocess.run(["git", "add", f], cwd=self.workspace, capture_output=True)
        else:
            subprocess.run(["git", "add", "-A"], cwd=self.workspace, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message], cwd=self.workspace,
            capture_output=True, text=True)
        if result.returncode != 0:
            return f"ERROR: {result.stderr}"
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.workspace,
            capture_output=True, text=True)
        return hash_result.stdout.strip()


class GitLog(Tool):
    name = "git_log"
    description = "Show recent git commit log."
    parameters = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of commits to show", "default": 10},
        },
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, n: int = 10) -> str:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            cwd=self.workspace, capture_output=True, text=True)
        return result.stdout


class GitDiff(Tool):
    name = "git_diff"
    description = "Show diff of current changes or against a ref."
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Git ref to diff against"},
        },
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, ref: str = None) -> str:
        cmd = ["git", "diff"]
        if ref:
            cmd.append(ref)
        result = subprocess.run(cmd, cwd=self.workspace, capture_output=True, text=True)
        return result.stdout[:10000]


class GitReset(Tool):
    name = "git_reset"
    description = "Reset workspace to a git ref."
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Git ref to reset to"},
        },
        "required": ["ref"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, ref: str) -> str:
        result = subprocess.run(
            ["git", "reset", "--hard", ref],
            cwd=self.workspace, capture_output=True, text=True)
        if result.returncode != 0:
            return f"ERROR: {result.stderr}"
        return f"Reset to {ref}"
