from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass

import yaml

from applied_scientist.core.utils import atomic_write


@dataclass
class GPUSlot:
    slot_id: int
    job_id: str | None = None
    experiment_name: str | None = None
    submitted_at: float | None = None
    status: str = "free"               # free | running | pending_review | completing
    log_dir: str | None = None


class GPUPool:
    """Manages GPU slot assignments with state persistence for crash recovery."""

    def __init__(self, n_gpus: int, state_path: str):
        """Create pool with n_gpus slots. Load from state_path if exists."""
        self._state_path = state_path
        self._lock = threading.Lock()
        if os.path.exists(state_path):
            self._slots = self._load_state(state_path, n_gpus)
        else:
            self._slots = [GPUSlot(slot_id=i) for i in range(n_gpus)]

    def get_free_slot(self) -> GPUSlot | None:
        """Return a free slot, or None if all occupied."""
        with self._lock:
            for slot in self._slots:
                if slot.status == "free":
                    return slot
            return None

    def assign(self, slot_id: int, job_id: str, experiment_name: str,
               log_dir: str) -> None:
        """Mark slot as running with given job. Persist state."""
        with self._lock:
            slot = self._slots[slot_id]
            slot.job_id = job_id
            slot.experiment_name = experiment_name
            slot.submitted_at = time.time()
            slot.status = "running"
            slot.log_dir = log_dir
            self._persist()

    def release(self, slot_id: int) -> None:
        """Mark slot as free. Persist state."""
        with self._lock:
            slot = self._slots[slot_id]
            slot.job_id = None
            slot.experiment_name = None
            slot.submitted_at = None
            slot.status = "free"
            slot.log_dir = None
            self._persist()

    def get_running(self) -> list[GPUSlot]:
        """Return all running slots."""
        with self._lock:
            return [s for s in self._slots if s.status == "running"]

    def get_timed_out(self, timeout: float) -> list[GPUSlot]:
        """Return slots that have exceeded the timeout."""
        with self._lock:
            now = time.time()
            return [s for s in self._slots
                    if s.status == "running" and s.submitted_at
                    and (now - s.submitted_at) > timeout]

    def resize(self, n_gpus: int) -> None:
        """Change pool size. Only affects free slots."""
        with self._lock:
            current = len(self._slots)
            if n_gpus > current:
                for i in range(current, n_gpus):
                    self._slots.append(GPUSlot(slot_id=i))
            elif n_gpus < current:
                while len(self._slots) > n_gpus:
                    if self._slots[-1].status == "free":
                        self._slots.pop()
                    else:
                        break
            self._persist()

    def get_status_summary(self) -> str:
        """Human-readable pool status."""
        with self._lock:
            lines = []
            for s in self._slots:
                if s.status == "free":
                    lines.append(f"  Slot {s.slot_id}: free")
                else:
                    elapsed = time.time() - s.submitted_at if s.submitted_at else 0
                    lines.append(f"  Slot {s.slot_id}: {s.status} — "
                                 f"{s.experiment_name} ({elapsed:.0f}s)")
            return "\n".join(lines)

    def _persist(self) -> None:
        """Write slot state to state_path (atomic write). Called after every mutation."""
        data = []
        for s in self._slots:
            data.append({
                "slot_id": s.slot_id, "job_id": s.job_id,
                "experiment_name": s.experiment_name,
                "submitted_at": s.submitted_at, "status": s.status,
                "log_dir": s.log_dir,
            })
        atomic_write(self._state_path, yaml.dump(data, default_flow_style=False))

    @classmethod
    def from_state(cls, state_path: str, n_gpus: int) -> GPUPool:
        """Recover pool from persisted state file."""
        pool = cls.__new__(cls)
        pool._state_path = state_path
        pool._lock = threading.Lock()
        pool._slots = pool._load_state(state_path, n_gpus)
        return pool

    def _load_state(self, state_path: str, n_gpus: int) -> list[GPUSlot]:
        """Load slot state from YAML file."""
        with open(state_path) as f:
            data = yaml.safe_load(f) or []
        slots = []
        for entry in data:
            slots.append(GPUSlot(**entry))
        while len(slots) < n_gpus:
            slots.append(GPUSlot(slot_id=len(slots)))
        return slots[:n_gpus]
