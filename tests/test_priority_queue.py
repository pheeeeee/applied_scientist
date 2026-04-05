import os
import pytest
from applied_scientist.core.priority_queue import PriorityQueue
from applied_scientist.core.spec import ExperimentSpec


def make_spec(name, category=""):
    return ExperimentSpec(
        name=name, description=f"Test {name}", source_paper="test",
        architecture={}, why_it_might_work="test",
        task_config={}, training_config={}, resource_estimate={},
        category=category,
    )


def test_insert_and_pop(tmp_path):
    q = PriorityQueue(str(tmp_path / "queue.yaml"))
    q.insert(make_spec("a"), 1.0)
    q.insert(make_spec("b"), 3.0)
    q.insert(make_spec("c"), 2.0)

    item = q.pop()
    assert item is not None
    assert item[0].name == "b"
    assert item[1] == 3.0


def test_peek(tmp_path):
    q = PriorityQueue(str(tmp_path / "queue.yaml"))
    q.insert(make_spec("a"), 1.0)
    q.insert(make_spec("b"), 3.0)
    items = q.peek(2)
    assert len(items) == 2
    assert items[0][0].name == "b"
    assert q.depth() == 2  # peek doesn't remove


def test_persistence(tmp_path):
    path = str(tmp_path / "queue.yaml")
    q = PriorityQueue(path)
    q.insert(make_spec("a"), 5.0)
    q.insert(make_spec("b"), 2.0)

    q2 = PriorityQueue(path)
    assert q2.depth() == 2
    item = q2.pop()
    assert item[0].name == "a"


def test_contains_and_similar(tmp_path):
    q = PriorityQueue(str(tmp_path / "queue.yaml"))
    q.insert(make_spec("mappo"), 1.0)
    assert q.contains("mappo")
    assert not q.contains("qmix")
    assert q.contains_similar("MAPPO") == "mappo"
    assert q.contains_similar("qmix") is None


def test_update_score(tmp_path):
    q = PriorityQueue(str(tmp_path / "queue.yaml"))
    q.insert(make_spec("a"), 1.0)
    q.insert(make_spec("b"), 2.0)
    q.update_score("a", 10.0)
    item = q.pop()
    assert item[0].name == "a"


def test_empty_pop(tmp_path):
    q = PriorityQueue(str(tmp_path / "queue.yaml"))
    assert q.pop() is None
