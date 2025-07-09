from pathlib import Path

import json

from slurmise.slurm import parse_slurm_job_metadata


def test_parse_slurm_job_metadata(monkeypatch):
    def mock_get_slurm_job_sacct(slurm_id):
        with open(Path(__file__).parent / "sacct_output.json") as f:
            return json.load(f)

    monkeypatch.setattr(
        "slurmise.slurm.get_slurm_job_sacct",
        mock_get_slurm_job_sacct,
    )

    expected_metadata = {
        "CPUs": 96,
        "elapsed_seconds": 97201,
        "job_name": "finetune_vicuna_7b",
        "max_rss": 0,
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
