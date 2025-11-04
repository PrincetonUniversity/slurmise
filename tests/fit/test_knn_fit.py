from pathlib import Path

import numpy as np
import pytest

from slurmise.fit.kneighbors_fit import KNNFit
from slurmise.fit.resource_fit import ResourceFit
from slurmise.job_data import JobData
from slurmise.job_database import JobDatabase


@pytest.fixture(autouse=True)
def monkey_patch_basepath(tmp_path, monkeypatch):
    """
    Monkey patch the BASEMODEL_PATH to the tmp_path, don't want to write to the actual
    path during testing (probably is user's home directory or something)
    """
    monkeypatch.setattr("slurmise.fit.resource_fit.BASEMODELPATH", tmp_path)
    yield
    monkeypatch.undo()


@pytest.mark.parametrize("specify_path", [True, False])
def test_model_path_creation(tmp_path, specify_path):
    """Test that both default and user-specified model paths are created correctly"""
    query = JobData(job_name="nupack")

    fit = ResourceFit(query=query, path=tmp_path if specify_path else None)

    # Check if the auto-generated path contains the class anem and the hash of the query
    if not specify_path:
        assert str(Path(fit.__class__.__name__) / Path(fit._get_model_info_hash(query))) in str(fit.path)
    else:
        assert tmp_path == fit.path


@pytest.fixture()
def nupack_data():
    query = JobData(job_name="nupack")

    with JobDatabase.get_database("tests/nupack2.h5") as db:
        # Get the job data
        jobs = db.query(job_data=query)

    # Drop jobs with 0 runtime or memory
    jobs = [job for job in jobs if job.runtime > 0 and job.memory > 0]

    # Only take jobs where sequences is len = 10
    # jobs = [job for job in jobs if job.numerical['sequences'].shape[0] == 10]

    return query, jobs


@pytest.mark.parametrize(
    ("model", "kwargs", "expected_metrics"),
    [
        (
            KNNFit,
            {"nneighbors": 5},
            {
                "runtime": {"mpe": 29.219869, "mse": 33.474729},
                "memory": {"mpe": 12.834444, "mse": 551520.403101},
            },
        )
    ],
)
def test_knn_fit_and_predict(nupack_data, model, kwargs, expected_metrics):
    """Test the fit classes on the nupack data"""

    query, jobs = nupack_data

    knn_fit = model(query=query, **kwargs)

    random_state = np.random.RandomState(42)
    knn_fit.fit(jobs, random_state=random_state)

    # Predict the runtime and memory of a job
    job = jobs[0]

    predicted_job, _ = knn_fit.predict(job)

    assert knn_fit.last_fit_dsize == int(len(jobs) * 0.8)

    # Save the model
    knn_fit.save()

    # Load the model
    knn_fit_loaded = KNNFit.load(query=query)
    assert knn_fit_loaded.last_fit_dsize == int(len(jobs) * 0.8)
    predicted_job2, _ = knn_fit_loaded.predict(job)
    assert predicted_job.runtime == predicted_job2.runtime
    assert predicted_job.memory == predicted_job2.memory

    for key in knn_fit.model_metrics.keys():
        assert key in expected_metrics
        for metric in knn_fit.model_metrics[key].keys():
            assert metric in expected_metrics[key]
            np.testing.assert_allclose(
                knn_fit.model_metrics[key][metric],
                expected_metrics[key][metric],
                rtol=1e-6,
            )
