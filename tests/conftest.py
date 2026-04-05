import pytest
import os
import tempfile

from applied_scientist.llm.base import LLMBackend, LLMResponse, LLMMessage
from applied_scientist.task.base import TaskAdapter
from applied_scientist.compute.base import JobRunner, JobStatus
from applied_scientist.config import (
    Config, SystemConfig, TuningConfig, LLMAgentConfig,
    ComputeConfig, PathsConfig,
)


class MockLLM(LLMBackend):
    """Returns canned responses. Records calls for assertions."""

    def __init__(self, responses=None):
        self.responses = iter(responses or ["mock response"])
        self.calls = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        content = next(self.responses, "mock response")
        return LLMResponse(
            content=content, tool_calls=None,
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn")

    def get_model_id(self):
        return "mock-model"

    @property
    def supports_tool_use(self):
        return True


class MockTaskAdapter(TaskAdapter):
    @property
    def name(self):
        return "MockTask"

    @property
    def metric(self):
        return ("accuracy", "higher")

    @property
    def domain_context(self):
        return "A test task."

    @property
    def code_map(self):
        return "## Files\n- model.py: edit this\n- train.py: DO NOT EDIT"

    @property
    def progress_file(self):
        return "progress.json"

    def train(self, *args, **kwargs):
        return {"accuracy": 0.85}

    def evaluate(self, *args, **kwargs):
        return {"accuracy": 0.85}

    def get_random_baseline(self):
        return {"accuracy": 0.10}

    def get_tunable_defaults(self):
        return {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}}


class MockJobRunner(JobRunner):
    def __init__(self):
        self.submitted = []
        self.cancelled = []

    def submit(self, command, job_name, resources=None):
        self.submitted.append({"command": command, "job_name": job_name})
        return f"mock_job_{len(self.submitted)}"

    def status(self, job_id):
        return JobStatus(job_id, "completed", 0, 10.0)

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def get_log(self, job_id, tail=50):
        return '{"accuracy": 0.85, "training_seconds": 10, "peak_memory_mb": 100}'


@pytest.fixture
def tmp_workspace(tmp_path):
    dirs = ["results", "configs", "configs/drafts", "configs/experiments",
            "results/checkpoints", "results/logs", ".state"]
    for d in dirs:
        os.makedirs(tmp_path / d, exist_ok=True)
    return tmp_path


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_task():
    return MockTaskAdapter()


@pytest.fixture
def mock_runner():
    return MockJobRunner()


@pytest.fixture
def mock_config(tmp_workspace):
    return Config(
        system=SystemConfig(),
        tuning=TuningConfig(),
        llm={
            role: LLMAgentConfig(backend="mock", model="mock-model")
            for role in ["explorer", "critic", "builder", "orchestrator"]
        },
        compute=ComputeConfig(backend="local"),
        task={"module": "tasks/mock"},
        paths=PathsConfig(
            workspace=str(tmp_workspace),
            results=str(tmp_workspace / "results"),
            configs=str(tmp_workspace / "configs"),
            checkpoints=str(tmp_workspace / "results" / "checkpoints"),
            logs=str(tmp_workspace / "results" / "logs"),
        ),
    )
