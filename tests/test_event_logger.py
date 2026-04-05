import json
import pytest
from applied_scientist.core.event_logger import EventLogger


def test_log_and_read(tmp_path):
    logger = EventLogger(str(tmp_path / "events.jsonl"))
    logger.log(event="test", value=42)
    entries = logger.read_all()
    assert len(entries) == 1
    assert entries[0]["event"] == "test"
    assert entries[0]["value"] == 42
    assert "timestamp" in entries[0]


def test_append(tmp_path):
    path = str(tmp_path / "events.jsonl")
    logger = EventLogger(path)
    logger.log(event="first")
    logger.log(event="second")
    entries = logger.read_all()
    assert len(entries) == 2


def test_empty_read(tmp_path):
    logger = EventLogger(str(tmp_path / "nonexistent.jsonl"))
    assert logger.read_all() == []
