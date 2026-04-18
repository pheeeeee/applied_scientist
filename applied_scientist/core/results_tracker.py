from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, field

from applied_scientist.core.utils import atomic_write


@dataclass
class ExperimentResult:
    name: str
    commit: str
    algorithm: str
    model: str
    metric_value: float | None         # primary metric (None for crashes/validation_fail)
    metric_std: float | None           # std across seeds (baselines only)
    seeds: int
    training_seconds: float
    peak_memory_mb: float
    status: str                        # baseline | keep | crash | early_stop | timeout | validation_fail
    description: str
    extra_metrics: dict = field(default_factory=dict)


class ResultsTracker:
    """Manages results.tsv with thread-safe append and atomic persistence."""

    HEADER = ["name", "commit", "algorithm", "model", "metric_value", "metric_std",
              "seeds", "training_seconds", "peak_memory_mb", "status", "description",
              "extra_metrics"]

    def __init__(self, path: str, metric_name: str, metric_direction: str):
        """Load existing results.tsv or create with header."""
        self._path = path
        self._metric_name = metric_name
        self._direction = metric_direction  # "higher" or "lower"
        self._lock = threading.Lock()
        self._results: list[ExperimentResult] = []

        def _to_int(v, default=0):
            if v is None or v == "":
                return default
            return int(float(v))

        def _to_float(v, default=0.0):
            if v is None or v == "":
                return default
            return float(v)

        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    self._results.append(ExperimentResult(
                        name=row.get("name", "") or "",
                        commit=row.get("commit", "") or "",
                        algorithm=row.get("algorithm", "") or "",
                        model=row.get("model", "") or "",
                        metric_value=_to_float(row.get("metric_value"), default=None) if row.get("metric_value") not in (None, "") else None,
                        metric_std=_to_float(row.get("metric_std"), default=None) if row.get("metric_std") not in (None, "") else None,
                        seeds=_to_int(row.get("seeds")),
                        training_seconds=_to_float(row.get("training_seconds")),
                        peak_memory_mb=_to_float(row.get("peak_memory_mb")),
                        status=row.get("status", "") or "",
                        description=row.get("description", "") or "",
                        extra_metrics=json.loads(row.get("extra_metrics", "{}"))
                        if row.get("extra_metrics") else {},
                    ))

    def add(self, result: ExperimentResult) -> None:
        """Append result. Thread-safe. Atomic write."""
        with self._lock:
            self._results.append(result)
            self._persist()

    def get_all(self) -> list[ExperimentResult]:
        """Return all results."""
        with self._lock:
            return list(self._results)

    def get_best(self, n: int = 5) -> list[ExperimentResult]:
        """Return top-n by primary metric (respecting direction)."""
        with self._lock:
            valid = [r for r in self._results if r.metric_value is not None]
            reverse = self._direction == "higher"
            valid.sort(key=lambda r: r.metric_value, reverse=reverse)
            return valid[:n]

    def get_baselines(self) -> list[ExperimentResult]:
        """Return all baseline results."""
        with self._lock:
            return [r for r in self._results if r.status == "baseline"]

    def get_by_name(self, name: str) -> ExperimentResult | None:
        """Return result for a specific experiment name."""
        with self._lock:
            for r in self._results:
                if r.name == name:
                    return r
            return None

    def is_completed(self, name: str) -> bool:
        """Check if experiment already has results (any status)."""
        with self._lock:
            return any(r.name == name for r in self._results)

    def get_summary(self) -> str:
        """Return human-readable summary table."""
        with self._lock:
            if not self._results:
                return "No results yet."
            lines = [f"{'Name':<30} {'Metric':<12} {'Status':<15} {'Time':<10}"]
            lines.append("-" * 70)
            for r in self._results:
                val = f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A"
                time_str = f"{r.training_seconds:.0f}s"
                lines.append(f"{r.name:<30} {val:<12} {r.status:<15} {time_str:<10}")
            return "\n".join(lines)

    def get_completed_names(self) -> set[str]:
        """Return set of completed experiment names."""
        with self._lock:
            return {r.name for r in self._results}

    def _persist(self) -> None:
        """Write results.tsv atomically."""
        lines = ["\t".join(self.HEADER)]
        for r in self._results:
            # Sanitize description — replace newlines/tabs to prevent TSV corruption
            safe_desc = r.description.replace("\n", " ").replace("\t", " ").replace("\r", "")
            row = [r.name, r.commit, r.algorithm, r.model,
                   str(r.metric_value) if r.metric_value is not None else "",
                   str(r.metric_std) if r.metric_std is not None else "",
                   str(r.seeds), str(r.training_seconds), str(r.peak_memory_mb),
                   r.status, safe_desc, json.dumps(r.extra_metrics)]
            lines.append("\t".join(row))
        atomic_write(self._path, "\n".join(lines) + "\n")
