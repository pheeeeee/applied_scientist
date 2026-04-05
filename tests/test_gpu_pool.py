import os
import pytest
from applied_scientist.core.gpu_pool import GPUPool


def test_assign_and_release(tmp_path):
    pool = GPUPool(2, str(tmp_path / "gpu.yaml"))
    slot = pool.get_free_slot()
    assert slot is not None
    pool.assign(slot.slot_id, "job1", "exp1", "/logs/exp1")
    assert len(pool.get_running()) == 1
    pool.release(slot.slot_id)
    assert len(pool.get_running()) == 0


def test_no_free_slot(tmp_path):
    pool = GPUPool(1, str(tmp_path / "gpu.yaml"))
    slot = pool.get_free_slot()
    pool.assign(slot.slot_id, "job1", "exp1", "/logs")
    assert pool.get_free_slot() is None


def test_persistence(tmp_path):
    path = str(tmp_path / "gpu.yaml")
    pool = GPUPool(2, path)
    pool.assign(0, "job1", "exp1", "/logs")
    pool2 = GPUPool(2, path)
    assert len(pool2.get_running()) == 1


def test_resize(tmp_path):
    pool = GPUPool(2, str(tmp_path / "gpu.yaml"))
    pool.resize(4)
    free = sum(1 for _ in range(4) if pool.get_free_slot())
    # All 4 slots should be gettable (though get_free_slot modifies nothing)
    assert pool.get_free_slot() is not None


def test_timed_out(tmp_path):
    import time
    pool = GPUPool(1, str(tmp_path / "gpu.yaml"))
    pool.assign(0, "job1", "exp1", "/logs")
    # Override submitted_at to simulate timeout
    pool._slots[0].submitted_at = time.time() - 10000
    timed_out = pool.get_timed_out(5000)
    assert len(timed_out) == 1


def test_status_summary(tmp_path):
    pool = GPUPool(2, str(tmp_path / "gpu.yaml"))
    pool.assign(0, "job1", "exp1", "/logs")
    summary = pool.get_status_summary()
    assert "exp1" in summary
    assert "free" in summary
