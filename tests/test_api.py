from slurmise.api import Slurmise

import multiprocessing
import time
import pytest
from unittest import mock


def slurmise_record(toml, process_id, error_queue):
    def mock_metadata(kwargs):
        result = {
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
        return result

    try:
        time.sleep(process_id * 0.1)
        with mock.patch(
            "slurmise.slurm.parse_slurm_job_metadata",
            side_effect=lambda *args, **kwargs: mock_metadata(kwargs),
        ):
            slurmise = Slurmise(toml)
            time.sleep(process_id * 0.1)
            for i in range(10):
                slurmise.record(
                    "nupack monomer -T 2 -C simple", slurm_id=process_id * 100 + i
                )
                time.sleep(process_id * 0.1)
    except Exception as e:
        error_queue.put(f"PID {process_id}: {e}")


def test_multiple_slurmise_instances(simple_toml):
    processes = []
    error_queue = multiprocessing.Queue()
    for i in range(10):
        p = multiprocessing.Process(
            target=slurmise_record, args=(simple_toml.toml, i, error_queue)
        )
        processes.append(p)
        p.start()

    [p.join() for p in processes]

    if not error_queue.empty():
        while not error_queue.empty():
            print(error_queue.get())
        pytest.fail("Child prcess had error")
