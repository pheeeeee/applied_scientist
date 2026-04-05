import os
import pytest
from applied_scientist.core.results_tracker import ResultsTracker, ExperimentResult


def make_result(name, metric=0.5, status="keep"):
    return ExperimentResult(
        name=name, commit="abc123", algorithm=name, model="mock",
        metric_value=metric, metric_std=None, seeds=1,
        training_seconds=100, peak_memory_mb=4000,
        status=status, description=f"Test {name}",
    )


def test_add_and_get(tmp_path):
    rt = ResultsTracker(str(tmp_path / "results.tsv"), "accuracy", "higher")
    rt.add(make_result("exp1", 0.8))
    assert len(rt.get_all()) == 1
    assert rt.is_completed("exp1")
    assert not rt.is_completed("exp2")


def test_get_best(tmp_path):
    rt = ResultsTracker(str(tmp_path / "results.tsv"), "accuracy", "higher")
    rt.add(make_result("a", 0.5))
    rt.add(make_result("b", 0.9))
    rt.add(make_result("c", 0.7))
    best = rt.get_best(2)
    assert best[0].name == "b"
    assert best[1].name == "c"


def test_get_baselines(tmp_path):
    rt = ResultsTracker(str(tmp_path / "results.tsv"), "accuracy", "higher")
    rt.add(make_result("baseline", 0.5, status="baseline"))
    rt.add(make_result("exp1", 0.7))
    baselines = rt.get_baselines()
    assert len(baselines) == 1
    assert baselines[0].name == "baseline"


def test_persistence(tmp_path):
    path = str(tmp_path / "results.tsv")
    rt = ResultsTracker(path, "accuracy", "higher")
    rt.add(make_result("exp1", 0.85))
    rt2 = ResultsTracker(path, "accuracy", "higher")
    assert len(rt2.get_all()) == 1
    assert rt2.get_all()[0].metric_value == 0.85


def test_none_metric(tmp_path):
    rt = ResultsTracker(str(tmp_path / "results.tsv"), "accuracy", "higher")
    rt.add(make_result("crash", None, status="crash"))
    assert rt.get_all()[0].metric_value is None
    assert len(rt.get_best(5)) == 0  # None values excluded from best


def test_summary(tmp_path):
    rt = ResultsTracker(str(tmp_path / "results.tsv"), "accuracy", "higher")
    rt.add(make_result("exp1", 0.85))
    summary = rt.get_summary()
    assert "exp1" in summary
    assert "0.8500" in summary
