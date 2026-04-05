from applied_scientist.core.cost_tracker import CostTracker


def test_register_and_record():
    ct = CostTracker()
    ct.register("explorer", "claude-haiku-4-5-20251001")
    ct.record("explorer", 1000, 500)
    cost = ct.get_agent_cost("explorer")
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500
    assert cost.estimated_cost > 0


def test_total_cost():
    ct = CostTracker()
    ct.register("explorer", "claude-haiku-4-5-20251001")
    ct.register("builder", "claude-sonnet-4-6")
    ct.record("explorer", 1_000_000, 0)
    ct.record("builder", 1_000_000, 0)
    total = ct.get_total_cost()
    assert total > 0


def test_unregistered_agent():
    ct = CostTracker()
    import pytest
    with pytest.raises(ValueError):
        ct.record("unknown", 100, 50)


def test_summary():
    ct = CostTracker()
    ct.register("explorer", "claude-haiku-4-5-20251001")
    ct.record("explorer", 100, 50)
    summary = ct.get_summary()
    assert "explorer" in summary
    assert "TOTAL" in summary
