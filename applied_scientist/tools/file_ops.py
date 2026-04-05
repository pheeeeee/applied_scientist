from __future__ import annotations

import os
import glob as glob_module

from applied_scientist.tools.base import Tool


class ReadFile(Tool):
    name = "read_file"
    description = "Read a file's contents. Returns content with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "offset": {"type": "integer", "description": "Line to start from (0-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.exists(full_path):
            return f"ERROR: File not found: {path}"
        with open(full_path) as f:
            lines = f.readlines()
        selected = lines[offset:offset + limit]
        return "".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(selected))


class WriteFile(Tool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, content: str) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"


class EditFile(Tool):
    name = "edit_file"
    description = "Replace an exact string in a file. Fails if old_string is not found."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "old_string": {"type": "string", "description": "Exact text to find"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.exists(full_path):
            return f"ERROR: File not found: {path}"
        with open(full_path) as f:
            content = f.read()
        if old_string not in content:
            return f"ERROR: old_string not found in {path}"
        content = content.replace(old_string, new_string, 1)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Edited {path}: replaced 1 occurrence."


class ListDirectory(Tool):
    name = "list_directory"
    description = "List files in a directory, optionally filtered by glob pattern."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace"},
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py')"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, pattern: str = "*") -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.isdir(full_path):
            return f"ERROR: Not a directory: {path}"
        matches = glob_module.glob(os.path.join(full_path, pattern))
        rel_paths = [os.path.relpath(m, self.workspace) for m in sorted(matches)]
        return "\n".join(rel_paths) if rel_paths else "(empty directory)"
