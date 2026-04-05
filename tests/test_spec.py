import os
import pytest
from applied_scientist.core.spec import ExperimentSpec


@pytest.fixture
def sample_spec():
    return ExperimentSpec(
        name="test_exp",
        description="A test experiment",
        source_paper="Test et al. 2024",
        architecture={"type": "fc", "layers": 2},
        why_it_might_work="Testing purposes",
        task_config={"env": "test"},
        training_config={"lr": 0.001},
        resource_estimate={"memory": "4GB"},
    )


def test_spec_creation(sample_spec):
    assert sample_spec.name == "test_exp"
    assert sample_spec.confidence == "high"
    assert sample_spec.category == ""
    assert sample_spec.tags == []
    assert sample_spec.tunable_hyperparameters is None


def test_spec_to_yaml(sample_spec, tmp_path):
    yaml_str = sample_spec.to_yaml()
    assert "test_exp" in yaml_str
    assert "Test et al. 2024" in yaml_str

    path = str(tmp_path / "spec.yaml")
    sample_spec.to_yaml(path)
    assert os.path.exists(path)


def test_spec_roundtrip(sample_spec, tmp_path):
    path = str(tmp_path / "spec.yaml")
    sample_spec.to_yaml(path)
    loaded = ExperimentSpec.from_yaml(path)
    assert loaded.name == sample_spec.name
    assert loaded.architecture == sample_spec.architecture
    assert loaded.training_config == sample_spec.training_config


def test_spec_tunable_hyperparameters(tmp_path):
    spec = ExperimentSpec(
        name="tunable", description="test", source_paper="test",
        architecture={}, why_it_might_work="test",
        task_config={}, training_config={}, resource_estimate={},
        tunable_hyperparameters={"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}},
    )
    path = str(tmp_path / "tunable.yaml")
    spec.to_yaml(path)
    loaded = ExperimentSpec.from_yaml(path)
    assert loaded.tunable_hyperparameters == spec.tunable_hyperparameters
