import pytest
from applied_scientist.core.message_bus import MessageBus, AgentMessage


def test_post_and_drain(tmp_path):
    bus = MessageBus(str(tmp_path / "bus.json"))
    bus.post(AgentMessage(sender="a", recipient="b", type="test", payload={"x": 1}))
    msgs = bus.drain("b")
    assert len(msgs) == 1
    assert msgs[0].payload == {"x": 1}


def test_priority_ordering(tmp_path):
    bus = MessageBus()
    bus.post(AgentMessage(sender="a", recipient="b", type="low", payload={}, priority=10))
    bus.post(AgentMessage(sender="a", recipient="b", type="high", payload={}, priority=100))
    msgs = bus.drain("b")
    assert msgs[0].type == "high"
    assert msgs[1].type == "low"


def test_empty_drain():
    bus = MessageBus()
    assert bus.drain("nobody") == []


def test_poll(tmp_path):
    bus = MessageBus()
    bus.post(AgentMessage(sender="a", recipient="b", type="test", payload={}))
    msg = bus.poll("b", timeout=1.0)
    assert msg is not None
    assert msg.type == "test"
    assert bus.poll("b", timeout=0.1) is None


def test_persistence(tmp_path):
    path = str(tmp_path / "bus.json")
    bus = MessageBus(path)
    bus.post(AgentMessage(sender="a", recipient="b", type="test", payload={"k": "v"}))
    bus2 = MessageBus(path)
    msgs = bus2.drain("b")
    assert len(msgs) == 1
    assert msgs[0].payload == {"k": "v"}


def test_separate_inboxes():
    bus = MessageBus()
    bus.post(AgentMessage(sender="a", recipient="explorer", type="t1", payload={}))
    bus.post(AgentMessage(sender="a", recipient="critic", type="t2", payload={}))
    assert len(bus.drain("explorer")) == 1
    assert len(bus.drain("critic")) == 1
    assert len(bus.drain("builder")) == 0
