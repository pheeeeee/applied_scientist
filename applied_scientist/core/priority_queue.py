from __future__ import annotations

import os
import threading
from typing import Callable

import yaml

from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.utils import atomic_write, normalize_name


class PriorityQueue:
    """Thread-safe ranked queue of experiment specs with deduplication.
    Persisted to YAML via atomic write on every mutation."""

    def __init__(self, queue_path: str):
        """Load from YAML or create empty."""
        self._path = queue_path
        self._lock = threading.Lock()
        self._items: list[tuple[float, ExperimentSpec]] = []
        if os.path.exists(queue_path):
            with open(queue_path) as f:
                data = yaml.safe_load(f) or []
            for entry in data:
                spec = ExperimentSpec(**entry["spec"])
                self._items.append((entry["score"], spec))
            self._items.sort(key=lambda x: -x[0])

    def insert(self, spec: ExperimentSpec, score: float) -> None:
        """Insert spec with score. Thread-safe. Atomic write to disk."""
        with self._lock:
            self._items.append((score, spec))
            self._items.sort(key=lambda x: -x[0])
            self._save_locked()

    def pop(self) -> tuple[ExperimentSpec, float] | None:
        """Remove and return highest-scored spec. None if empty."""
        with self._lock:
            if not self._items:
                return None
            score, spec = self._items.pop(0)
            self._save_locked()
            return (spec, score)

    def peek(self, n: int = 5) -> list[tuple[ExperimentSpec, float]]:
        """View top-n without removing."""
        with self._lock:
            return [(spec, score) for score, spec in self._items[:n]]

    def update_score(self, name: str, new_score: float) -> None:
        """Update score for a spec. Used during re-ranking."""
        with self._lock:
            for i, (score, spec) in enumerate(self._items):
                if normalize_name(spec.name) == normalize_name(name):
                    self._items[i] = (new_score, spec)
                    break
            self._items.sort(key=lambda x: -x[0])
            self._save_locked()

    def rerank(self, scoring_fn: Callable, context: dict) -> None:
        """Re-rank all specs using a scoring function and current context.
        scoring_fn(spec, context) -> float"""
        with self._lock:
            self._items = [(scoring_fn(spec, context), spec)
                           for _, spec in self._items]
            self._items.sort(key=lambda x: -x[0])
            self._save_locked()

    def depth(self) -> int:
        """Number of pending specs."""
        with self._lock:
            return len(self._items)

    def save(self) -> None:
        """Persist to YAML (atomic write). Public API for explicit save."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Internal save — must be called while holding self._lock."""
        data = []
        for score, spec in self._items:
            spec_dict = {
                "name": spec.name, "description": spec.description,
                "source_paper": spec.source_paper, "architecture": spec.architecture,
                "why_it_might_work": spec.why_it_might_work,
                "task_config": spec.task_config, "training_config": spec.training_config,
                "resource_estimate": spec.resource_estimate, "confidence": spec.confidence,
                "control_variant_of": spec.control_variant_of, "category": spec.category,
                "tags": spec.tags, "tunable_hyperparameters": spec.tunable_hyperparameters,
            }
            data.append({"score": score, "spec": spec_dict})
        atomic_write(self._path, yaml.dump(data, default_flow_style=False, sort_keys=False))

    def get_all(self) -> list[tuple[ExperimentSpec, float]]:
        """Return all specs with scores, sorted by rank."""
        with self._lock:
            return [(spec, score) for score, spec in self._items]

    def contains(self, name: str) -> bool:
        """Check if a spec with this name is in the queue."""
        with self._lock:
            return any(normalize_name(spec.name) == normalize_name(name)
                       for _, spec in self._items)

    def contains_similar(self, name: str) -> str | None:
        """Check if a semantically similar name exists (normalized comparison).
        Returns the matching name or None."""
        with self._lock:
            target = normalize_name(name)
            for _, spec in self._items:
                if normalize_name(spec.name) == target:
                    return spec.name
            return None
