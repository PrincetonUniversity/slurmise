from slurmise.slurm import parse_slurm_job_metadata


def generate_job_metadata(**kargs):
    return {
        "jobs": [
            {
                "job_id": kargs.get("job_id", 58976578),
                "task_id": "extern",
                "name": "finetune_vicuna_7b",
                "state": {"current": ["RUNNING"]},
                "partition": "pli-c",
                "required": {
                    "CPUs": 96,
                    "memory_per_cpu": {"set": False, "infinite": False, "number": 0},
                    "memory_per_node": {"set": True, "infinite": False, "number": 729088},
                },
                "steps": [
                    {
                        "time": {"elapsed": kargs.get("elapsed", 97201)},
                        "tasks": {"count": kargs.get("task_count", 3)},
                        "step": {"id": f"{kargs.get('job_id', 58976578)}.extern", "name": "extern"},
                        "tres": {
                            "requested": {
                                "max": [
                                    {
                                        "type": "mem",
                                        "name": "",
                                        "id": 2,
                                        "count": kargs.get("mem_count", 24786677760),
                                        "task": 0,
                                        "node": "tiger-g04c6n7",
                                    }
                                ]
                            }
                        },
                    }
                ],
            }
        ]
    }

def test_parse_slurm_job_metadata(monkeypatch):
    def mock_get_slurm_job_sacct(slurm_id):  # noqa: ARG001
        return generate_job_metadata()

    monkeypatch.setattr(
        "slurmise.slurm.get_slurm_job_sacct",
        mock_get_slurm_job_sacct,
    )

    expected_metadata = {
        "CPUs": 96,
        "elapsed_seconds": 97201,
        "job_name": "finetune_vicuna_7b",
        "max_rss": 70917,
        "memory_per_cpu": {
            "infinite": False,
            "number": 0,
            "set": False,
        },
        "memory_per_node": {
            "infinite": False,
            "number": 729088,
            "set": True,
        },
        "partition": "pli-c",
        "slurm_id": 58976578,
        "state": "RUNNING",
        "step_id": "extern",
    }

    assert parse_slurm_job_metadata("58976578") == expected_metadata
    assert parse_slurm_job_metadata("58976578", step_name="extern") == expected_metadata


def test_parse_slurm_job_metadata2(monkeypatch):
    def mock_get_slurm_job_sacct(slurm_id):  # noqa: ARG001
        return generate_job_metadata(task_count=5)

    monkeypatch.setattr(
        "slurmise.slurm.get_slurm_job_sacct",
        mock_get_slurm_job_sacct,
    )

    expected_metadata = {
        "CPUs": 96,
        "elapsed_seconds": 97201,
        "job_name": "finetune_vicuna_7b",
        "max_rss": 118195,
        "memory_per_cpu": {
            "infinite": False,
            "number": 0,
            "set": False,
        },
        "memory_per_node": {
            "infinite": False,
            "number": 729088,
            "set": True,
        },
        "partition": "pli-c",
        "slurm_id": 58976578,
        "state": "RUNNING",
        "step_id": "extern",
    }

    assert parse_slurm_job_metadata("58976578") == expected_metadata
    assert parse_slurm_job_metadata("58976578", step_name="extern") == expected_metadata
