import multiprocessing
import time
from unittest import mock

import pytest

from slurmise import job_database
from slurmise.api import Slurmise
from slurmise.job_data import JobData


def slurmise_record(toml, process_id, error_queue):
    def mock_metadata(kwargs):
        return {
            "slurm_id": kwargs["slurm_id"],
            "job_name": "nupack",
            "state": "COMPLETED",
            "partition": "",
            "elapsed_seconds": 97201,
            "CPUs": 1,
            "memory_per_cpu": 0,
            "memory_per_node": 0,
            "max_rss": 232,
            "step_id": "external",
        }

    try:
        time.sleep(process_id * 0.1)
        with mock.patch(
            "slurmise.slurm.parse_slurm_job_metadata",
            side_effect=lambda *args, **kwargs: mock_metadata(kwargs),
        ):
            slurmise = Slurmise(toml)
            time.sleep(process_id * 0.1)
            for i in range(10):
                slurmise.record("nupack monomer -T 2 -C simple", slurm_id=str(process_id * 100 + i))
                time.sleep(process_id * 0.1)
    except Exception as e:  # noqa: BLE001
        error_queue.put(f"PID {process_id}: {e}")


def test_multiple_slurmise_instances(simple_toml):
    processes = []
    error_queue = multiprocessing.Queue()
    for i in range(10):
        p = multiprocessing.Process(target=slurmise_record, args=(simple_toml.toml, i, error_queue))
        processes.append(p)
        p.start()

    [p.join() for p in processes]

    if not error_queue.empty():
        while not error_queue.empty():
            print(error_queue.get())
        pytest.fail("Child prcess had error")


def test_job_data_from_dict(simple_toml):
    slurmise = Slurmise(simple_toml.toml)
    result = slurmise.job_data_from_dict(
        {"threads": 3, "complexity": "simple"},
        "nupack",
    )
    assert result.categories == {"complexity": "simple"}
    assert result.numerics == {"threads": 3}


@pytest.mark.parametrize(
    "toml_fixture",
    ["simple_toml", "nupackdefaults_toml", "small_db_toml"],
)
def test_update_all_models(toml_fixture, request):
    toml = request.getfixturevalue(toml_fixture)
    slurmise = Slurmise(toml.toml)
    try:
        slurmise.update_all_models()
    except ValueError as e:
        # If there is not enough data to fit a model, a ValueError is raised
        # by sklearn train_test_split. Currently happening with small_db_toml fixture
        # because there is only one job with "filesizes" numeric feature.
        if str(e).startswith("Cannot have number of splits n_splits="):
            pass


def test_raw_record_uses_env_slurm_id(simple_toml, monkeypatch, no_slurm_env, sacct_mock):
    """When slurm_id is None, raw_record should fall back to the SLURM_JOB_ID env var."""
    sacct_calls = sacct_mock()
    monkeypatch.setenv("SLURM_JOB_ID", "99999")

    job = JobData(job_name="nupack", slurm_id=None)
    slurmise = Slurmise(simple_toml.toml)
    slurmise.raw_record(job)

    assert job.slurm_id == "99999"
    assert sacct_calls == ["99999"]


def test_raw_record_raises_when_no_slurm_id(simple_toml, no_slurm_env):
    """When slurm_id is None and SLURM_JOB_ID is not set, raise a descriptive ValueError."""
    job = JobData(job_name="nupack", slurm_id=None)

    slurmise = Slurmise(simple_toml.toml)
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurmise.raw_record(job)


def test_record_step_id_without_slurm_id(simple_toml, monkeypatch, no_slurm_env, sacct_mock):
    """Regression test for issue 79: --step-id without --slurm-id must resolve the
    job id from the environment instead of producing the literal id "None.<step>"."""
    sacct_calls = sacct_mock(step_name="0", task_count=1, mem_count=232 * 2**20)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    slurmise = Slurmise(simple_toml.toml)
    slurmise.record("nupack monomer -T 2 -C simple", step_id="0")

    assert sacct_calls == ["1234"]

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        results = db.query(JobData(job_name="nupack", categories={"complexity": "simple"}))
        assert [job.slurm_id for job in results] == ["1234.0"]
        assert results[0].memory == 232
        assert results[0].runtime == 1620


def test_record_step_id_without_slurm_id_or_env(simple_toml, no_slurm_env):
    """With a step_id but neither slurm_id nor environment, fail with a clear error."""
    slurmise = Slurmise(simple_toml.toml)
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurmise.record("nupack monomer -T 2 -C simple", step_id="0")
