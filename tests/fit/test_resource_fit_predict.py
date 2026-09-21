"""Tests for the model-accuracy ResourceFit predictions."""

import math

import numpy as np
import pytest

from slurmise.fit.poly_fit import PolynomialFit
from slurmise.fit.resource_fit import MPE_THRESHOLD
from slurmise.job_data import JobData
from slurmise.resource_corrector import ResourceCorrector


def _corrector(resource="runtime", default=60, **kwargs) -> ResourceCorrector:
    return ResourceCorrector(resource=resource, default=default, **kwargs)


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
    rt_corrector = _corrector("runtime", default=60)
    mem_corrector = _corrector("memory", default=1000)
    job, warnings = fit.predict(query, rt_corrector, mem_corrector)

    assert warnings == []
    assert job.runtime == pytest.approx(2 * 8 + 5)
    assert job.memory == pytest.approx(100 * 8 + 50)


def test_inaccurate_model_prediction_is_used_with_warning(monkey_patch_basepath):
    """Test a synthetic job that has an odd/even numeric parameter
    dependence that leads to poor PolynomialFit training accuracy.

    Exceeding the error threshold attaches a warning; with the default
    on_high_uncertainty_return='prediction' the prediction is still used.
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
    rt_corrector = _corrector("runtime", default=200, on_high_uncertainty_return="prediction")
    mem_corrector = _corrector("memory", default=5000, on_high_uncertainty_return="prediction")
    job, warnings = fit.predict(query, rt_corrector, mem_corrector)

    warning_text = "\n".join(warnings)
    assert "uncertainty" in warning_text
    # The poor fit is warned about, but its prediction is still what gets returned.
    assert job.runtime != 200
    assert job.memory != 5000
    assert job.runtime > 0
    assert job.memory > 0


def test_inaccurate_model_returns_default_when_configured(monkey_patch_basepath):
    """on_high_uncertainty_return='default' returns the default value on a poor fit."""

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

    query = JobData(job_name="synthetic", numerics={"n": 7}, runtime=99, memory=888)
    rt_corrector = _corrector("runtime", default=99, on_high_uncertainty_return="default")
    mem_corrector = _corrector("memory", default=888, on_high_uncertainty_return="default")
    job, warnings = fit.predict(query, rt_corrector, mem_corrector)

    assert job.runtime == pytest.approx(99)
    assert job.memory == pytest.approx(888)
    assert any("default" in w for w in warnings)


def test_too_few_records_is_not_predicted_with_warning(monkey_patch_basepath):
    """Under ten records there is not enough to fit, so predict declines and says so.

    The record is handed back as it came in, with a warning naming the reason.
    Choosing the defaults is left to the caller.
    """
    jobs = [
        JobData(
            job_name="synthetic",
            slurm_id=str(n),
            numerics={"n": n},
            runtime=2 * n + 5,
            memory=100 * n + 50,
        )
        for n in range(1, 6)
    ]

    fit = PolynomialFit(query=JobData(job_name="synthetic"), degree=2)
    fit.fit(jobs, random_state=np.random.RandomState(42))

    query = JobData(job_name="synthetic", numerics={"n": 3}, runtime=60, memory=1000)
    rt_corrector = _corrector("runtime", default=60)
    mem_corrector = _corrector("memory", default=1000)
    job, warnings = fit.predict(query, rt_corrector, mem_corrector)

    assert len(warnings) == 1
    assert "Not enough fitting data points" in warnings[0]
    # The record comes back untouched; applying defaults is the caller's job.
    assert job.runtime == 60
    assert job.memory == 1000


def test_prediction_clamped_to_maximum(monkey_patch_basepath):
    """A model prediction above the configured maximum is clamped, not rejected."""

    jobs = [
        JobData(
            job_name="synthetic",
            slurm_id=str(n),
            numerics={"n": n},
            runtime=10 * n,
            memory=1000 * n,
        )
        for n in range(1, 41, 2)
    ]

    fit = PolynomialFit(query=JobData(job_name="synthetic"), degree=2)
    fit.fit(jobs, random_state=np.random.RandomState(42))

    # Ask to predict for a huge n so the model extrapolates far above the cap
    query = JobData(job_name="synthetic", numerics={"n": 9999}, runtime=60, memory=1000)
    rt_corrector = _corrector("runtime", default=60, maximum=500)
    mem_corrector = _corrector("memory", default=1000, maximum=50000)
    job, warnings = fit.predict(query, rt_corrector, mem_corrector)

    assert job.runtime == pytest.approx(500)
    assert job.memory == pytest.approx(50000)
    assert any("maximum" in w for w in warnings)


def test_retry_scaling_applied(monkey_patch_basepath):
    """attempt>0 scales the raw prediction by attempt**retry_exponent before clamping."""

    jobs = [
        JobData(
            job_name="synthetic",
            slurm_id=str(n),
            numerics={"n": n},
            runtime=10 * n,
            memory=1000 * n,
        )
        for n in range(1, 41, 2)
    ]

    fit = PolynomialFit(query=JobData(job_name="synthetic"), degree=2)
    fit.fit(jobs, random_state=np.random.RandomState(42))

    query_base = JobData(job_name="synthetic", numerics={"n": 5}, runtime=60, memory=1000)
    query_retry = JobData(job_name="synthetic", numerics={"n": 5}, runtime=60, memory=1000)

    rt_corrector = _corrector("runtime", default=60, maximum=math.inf, retry_exponent=1.0)
    mem_corrector = _corrector("memory", default=1000, maximum=math.inf, retry_exponent=1.0)

    base_job, _ = fit.predict(query_base, rt_corrector, mem_corrector, attempt=0)
    retry_job, _ = fit.predict(query_retry, rt_corrector, mem_corrector, attempt=2)

    assert retry_job.runtime == pytest.approx(base_job.runtime * 2)
    assert retry_job.memory == pytest.approx(base_job.memory * 2)
