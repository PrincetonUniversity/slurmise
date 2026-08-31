"""Tests for the model-accuracy ResourceFit predictions."""

import numpy as np
import pytest

from slurmise.fit.poly_fit import PolynomialFit
from slurmise.fit.resource_fit import MPE_THRESHOLD
from slurmise.job_data import JobData


@pytest.fixture
def monkey_patch_basepath(tmp_path, monkeypatch):
    """
    Monkey patch the BASEMODELPATH to the tmp_path, don't want to write to the actual
    path during testing (probably is user's home directory or something)
    """
    monkeypatch.setattr("slurmise.fit.resource_fit.BASEMODELPATH", tmp_path)
    yield
    monkeypatch.undo()


def test_accurate_model_prediction_is_used(monkey_patch_basepath):
    """Accurate PolynomialFit on synthetic job with one numeric parameter which
    has linear scaling.
    """

    jobs = [
        JobData(
            job_name="synthetic",
            slurm_id=str(n),
            numerics={"n": n},
            runtime=2 * n + 5,
            memory=100 * n + 50,
        )
        for n in range(1, 41, 2)  # Odd values for numeric n
    ]

    fit = PolynomialFit(query=JobData(job_name="synthetic"), degree=2)
    fit.fit(jobs, random_state=np.random.RandomState(42))

    assert fit.model_metrics["runtime"]["mpe"] < MPE_THRESHOLD
    assert fit.model_metrics["memory"]["mpe"] < MPE_THRESHOLD

    # Predict on n=8 which is not in the training
    query = JobData(job_name="synthetic", numerics={"n": 8}, runtime=60, memory=1000)
    job, warnings = fit.predict(query)

    assert warnings == []
    assert job.runtime == pytest.approx(2 * 8 + 5)
    assert job.memory == pytest.approx(100 * 8 + 50)


def test_inaccurate_model_prediction_is_used_with_warning(monkey_patch_basepath):
    """Test a synthetic job that has an odd/even numeric parameter
    dependence that leads to poor PolynomialFit training accuracy.

    Exceeding the error threshold only attaches a warning; the prediction is still used.
    """

    jobs = [
        JobData(
            job_name="synthetic",
            slurm_id=str(n),
            numerics={"n": n},
            runtime=10 if n % 2 else 200,
            memory=100 if n % 2 else 5000,
        )
        for n in range(1, 21)
    ]

    fit = PolynomialFit(query=JobData(job_name="synthetic"), degree=2)
    fit.fit(jobs, random_state=np.random.RandomState(42))

    assert fit.model_metrics["runtime"]["mpe"] > MPE_THRESHOLD
    assert fit.model_metrics["memory"]["mpe"] > MPE_THRESHOLD

    query = JobData(job_name="synthetic", numerics={"n": 7}, runtime=200, memory=5000)
    job, warnings = fit.predict(query)

    warning_text = "\n".join(warnings)
    assert "Runtime prediction" in warning_text
    assert "Memory prediction" in warning_text
    # The poor fit is warned about, but its prediction is still what gets returned.
    # rather than the defaults
    assert job.runtime != 200
    assert job.memory != 5000
    assert job.runtime > 0
    assert job.memory > 0
