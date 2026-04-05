"""Agent tools (pluggable). Registry provides get_tools(agent_role, workspace_dir)."""

from __future__ import annotations

from applied_scientist.tools.base import Tool
from applied_scientist.tools.file_ops import ReadFile, WriteFile, EditFile, ListDirectory
from applied_scientist.tools.web_search import SearchWeb
from applied_scientist.tools.paper_reader import SearchPapers, ReadPaper
from applied_scientist.tools.shell import ShellTool
from applied_scientist.tools.git_ops import GitCommit, GitLog, GitDiff, GitReset
from applied_scientist.tools.critic_tools import (
    SubmitSpecReview, SubmitCodeReview, SubmitInsightReview, SubmitSuggestionTriage,
)

AGENT_TOOLS = {
    "explorer": ["read_file", "write_file", "list_directory",
                 "search_web", "search_papers", "read_paper"],
    "critic":   ["read_file", "write_file", "list_directory",
                 "submit_review", "submit_code_review",
                 "submit_insight_review", "submit_triage"],
    "builder":  ["read_file", "write_file", "edit_file", "list_directory",
                 "run_command", "git_commit", "git_log", "git_diff", "git_reset"],
    "orchestrator": ["read_file", "list_directory"],
}

_TOOL_CLASSES = {
    "read_file": ReadFile,
    "write_file": WriteFile,
    "edit_file": EditFile,
    "list_directory": ListDirectory,
    "search_web": SearchWeb,
    "search_papers": SearchPapers,
    "read_paper": ReadPaper,
    "run_command": ShellTool,
    "git_commit": GitCommit,
    "git_log": GitLog,
    "git_diff": GitDiff,
    "git_reset": GitReset,
    "submit_review": SubmitSpecReview,
    "submit_code_review": SubmitCodeReview,
    "submit_insight_review": SubmitInsightReview,
    "submit_triage": SubmitSuggestionTriage,
}


def get_tools(agent_role: str, workspace_dir: str) -> list[Tool]:
    """Return instantiated tools for the given agent role."""
    tool_names = AGENT_TOOLS.get(agent_role, [])
    tools = []
    for name in tool_names:
        cls = _TOOL_CLASSES[name]
        try:
            tools.append(cls(workspace_dir=workspace_dir))
        except TypeError:
            tools.append(cls())
    return tools
