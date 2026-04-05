from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from applied_scientist.core.utils import atomic_write


@dataclass
class ExperimentSpec:
    name: str                                      # unique identifier (e.g., "mappo")
    description: str                               # one-line summary
    source_paper: str                              # citation
    architecture: dict                             # free-form architecture description
    why_it_might_work: str                         # rationale for this task
    task_config: dict                              # task-specific configuration
    training_config: dict                          # hyperparameters
    resource_estimate: dict                        # memory, time estimates
    confidence: str = "high"                       # "high" or "low" (force-approved)
    control_variant_of: str | None = None          # name of parent spec if this is a control
    category: str = ""                             # for synthesis grouping
    tags: list[str] = field(default_factory=list)
    tunable_hyperparameters: dict | None = None    # per-spec search space from Explorer

    def to_yaml(self, path: str | None = None) -> str:
        """Serialize to YAML. If path given, write atomically. Returns YAML string."""
        data = {
            "name": self.name,
            "description": self.description,
            "source_paper": self.source_paper,
            "architecture": self.architecture,
            "why_it_might_work": self.why_it_might_work,
            "task_config": self.task_config,
            "training_config": self.training_config,
            "resource_estimate": self.resource_estimate,
            "confidence": self.confidence,
            "control_variant_of": self.control_variant_of,
            "category": self.category,
            "tags": self.tags,
            "tunable_hyperparameters": self.tunable_hyperparameters,
        }
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            atomic_write(path, content)
        return content

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentSpec:
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
