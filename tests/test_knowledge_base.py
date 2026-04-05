import os
import pytest
from applied_scientist.core.knowledge_base import KnowledgeBase


def test_add_insight(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.md"))
    kb.add_insight("Experiment 1", "MAPPO outperformed baseline.")
    content = kb.get_full()
    assert "Experiment 1" in content
    assert "MAPPO outperformed baseline." in content


def test_add_synthesis(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.md"))
    kb.add_synthesis("Algorithms", "PPO variants work best.")
    synthesis = kb.get_synthesis()
    assert "Algorithms" in synthesis
    assert "PPO variants work best." in synthesis


def test_synthesis_pinned_above_insights(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.md"))
    kb.add_insight("Exp1", "Body 1")
    kb.add_synthesis("Summary", "Overall findings")
    content = kb.get_full()
    synthesis_pos = content.find("Summary")
    insights_pos = content.find("Exp1")
    assert synthesis_pos < insights_pos


def test_persistence(tmp_path):
    path = str(tmp_path / "kb.md")
    kb = KnowledgeBase(path)
    kb.add_insight("Test", "Persisted insight")
    kb2 = KnowledgeBase(path)
    assert "Persisted insight" in kb2.get_full()


def test_get_recent(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.md"))
    for i in range(10):
        kb.add_insight(f"Exp{i}", f"Body {i}")
    recent = kb.get_recent(3)
    assert "Exp9" in recent
    assert "Exp7" in recent
