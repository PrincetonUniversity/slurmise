import multiprocessing
import time
from unittest import mock

import pytest

from slurmise.api import Slurmise
from slurmise.job_data import JobData
from slurmise.job_database import JobDatabase


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
    assert result.categorical == {"complexity": "simple"}
    assert result.numerical == {"threads": 3}


def test_update_all_models_empty_database(simple_toml):
    slurmise = Slurmise(simple_toml.toml)
    slurmise.update_all_models()


# TODO avoid rewriting duplicated logic to fixtures in tests/test_job_database.py
def test_update_all_models(simple_toml, tmp_path):
    hdf5_path = tmp_path / "test_db.h5"
    slurmise = Slurmise(simple_toml.toml)
    slurmise.configuration.db_filename = str(hdf5_path)

    with JobDatabase.get_database(hdf5_path) as db:
        for job_id in range(50):
            db.record(
                JobData(
                    job_name="nupack",
                    slurm_id=str(job_id),
                    categorical={"complexity": "simple"},
                    numerical={"threads": 2},
                    memory=200,
                    runtime=3600,
                )
            )

    slurmise.update_all_models()
