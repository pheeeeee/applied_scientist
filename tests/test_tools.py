import os
import json
import pytest
from applied_scientist.tools.file_ops import ReadFile, WriteFile, EditFile, ListDirectory
from applied_scientist.tools.shell import ShellTool
from applied_scientist.tools.git_ops import GitCommit, GitLog
from applied_scientist.tools.critic_tools import (
    SubmitSpecReview, SubmitCodeReview, SubmitInsightReview, SubmitSuggestionTriage,
)


def test_write_and_read(tmp_path):
    ws = str(tmp_path)
    WriteFile(ws).execute(path="test.txt", content="hello world")
    result = ReadFile(ws).execute(path="test.txt")
    assert "hello world" in result


def test_edit_file(tmp_path):
    ws = str(tmp_path)
    WriteFile(ws).execute(path="test.txt", content="foo bar baz")
    EditFile(ws).execute(path="test.txt", old_string="bar", new_string="qux")
    result = ReadFile(ws).execute(path="test.txt")
    assert "qux" in result
    assert "bar" not in result


def test_list_directory(tmp_path):
    ws = str(tmp_path)
    WriteFile(ws).execute(path="a.py", content="x")
    WriteFile(ws).execute(path="b.py", content="y")
    result = ListDirectory(ws).execute(path=".", pattern="*.py")
    assert "a.py" in result
    assert "b.py" in result


def test_path_traversal_blocked(tmp_path):
    ws = str(tmp_path / "workspace")
    os.makedirs(ws)
    result = ReadFile(ws).execute(path="../../../etc/passwd")
    assert "ERROR" in result


def test_shell_tool(tmp_path):
    ws = str(tmp_path)
    result = ShellTool(ws).execute(command="echo hello")
    assert "hello" in result


def test_shell_blocked_commands(tmp_path):
    ws = str(tmp_path)
    result = ShellTool(ws).execute(command="sudo rm -rf /")
    assert "ERROR" in result


def test_critic_tools():
    review = SubmitSpecReview()
    result = json.loads(review.execute(verdict="approved", feedback="LGTM", score=4.5))
    assert result["verdict"] == "approved"

    code_review = SubmitCodeReview()
    result = json.loads(code_review.execute(verdict="rejected", feedback="Bug found"))
    assert result["verdict"] == "rejected"

    insight = SubmitInsightReview()
    result = json.loads(insight.execute(verdict="approved"))
    assert result["verdict"] == "approved"

    triage = SubmitSuggestionTriage()
    result = json.loads(triage.execute(level="major", justification="Novel idea"))
    assert result["level"] == "major"
