import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from slurmise.job_data import JobData
from slurmise.job_database import JobDatabase


class TomlReturn(NamedTuple):
    toml: str
    db: str


def generate_job_metadata(**kwargs):
    """Build a sacct-style JSON dict like `sacct -j <id> --json` returns."""
    job_id = kwargs.get("job_id", 58976578)
    step_name = kwargs.get("step_name", "extern")
    return {
        "jobs": [
            {
                "job_id": job_id,
                "task_id": "extern",
                "name": kwargs.get("job_name", "finetune_vicuna_7b"),
                "state": {"current": ["RUNNING"]},
                "partition": "mypartition",
                "required": {
                    "CPUs": 96,
                    "memory_per_cpu": {"set": False, "infinite": False, "number": 0},
                    "memory_per_node": {"set": True, "infinite": False, "number": 729088},
                },
                "steps": [
                    {
                        "time": {"elapsed": kwargs.get("elapsed", 97201)},
                        "tasks": {"count": kwargs.get("task_count", 3)},
                        "step": {"id": f"{job_id}.{step_name}", "name": step_name},
                        "tres": {
                            "requested": {
                                "max": [
                                    {
                                        "type": "mem",
                                        "name": "",
                                        "id": 2,
                                        "count": kwargs.get("mem_count", 24786677760),
                                        "task": 0,
                                    }
                                ]
                            }
                        },
                    }
                ],
            }
        ]
    }


@pytest.fixture
def sacct_mock(monkeypatch):
    """Patch slurmise.slurm.get_slurm_job_sacct with generated sacct JSON.

    Returns a factory: calling it with generate_job_metadata kwargs installs the
    patch and returns the list of slurm_ids get_slurm_job_sacct is called with.
    """

    def _patch(**kwargs):
        calls = []

        def mock_get_slurm_job_sacct(slurm_id):
            calls.append(slurm_id)
            return generate_job_metadata(**kwargs)

        monkeypatch.setattr("slurmise.slurm.get_slurm_job_sacct", mock_get_slurm_job_sacct)
        return calls

    return _patch


@pytest.fixture
def no_slurm_env(monkeypatch):
    """Remove all SLURM job id variables from the environment."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)


@pytest.fixture
def simple_toml(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d / "slurmise_dir"}"

    [slurmise.job.nupack]
    job_spec = "monomer -T {{threads:numeric}} -C {{complexity:category}}"
    """
    )
    return TomlReturn(p, d / "slurmise_dir" / "slurmise.h5")


@pytest.fixture
def nupack_toml(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d / "slurmise_dir"}"
    db_filename = "nupack2.h5"

    [slurmise.job.nupack]
    job_spec = "monomer -c {{cpus:numeric}} -S {{sequences:numeric}}"
    """
    )

    db_path = d / "slurmise_dir" / "nupack2.h5"
    Path.mkdir(db_path.parent, exist_ok=True, parents=True)
    shutil.copyfile(
        "./tests/nupack2.h5",
        db_path,
    )

    return TomlReturn(p, db_path)


@pytest.fixture
def nupackdefaults_toml(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d / "slurmise_dir"}"
    db_filename = "nupack2.h5"

    [slurmise.job.nupack]
    job_spec = "monomer -c {{cpus:numeric}} -S {{sequences:numeric}}"
    default_mem = 3000
    default_time = 80
    """
    )

    db_path = d / "slurmise_dir" / "nupack2.h5"
    Path.mkdir(db_path.parent, exist_ok=True, parents=True)
    shutil.copyfile(
        "./tests/nupack2.h5",
        db_path,
    )

    return TomlReturn(p, db_path)


@pytest.fixture
def empty_h5py_file(tmp_path):
    d = tmp_path
    return d / "slurmise.h5"


@pytest.fixture
def small_db(empty_h5py_file):
    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                runtime=6,
                memory=128,
                numerics={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
                categories={"option1": "value1", "option2": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                numerics={"filesizes": [123, 512, 128]},
                categories={"option1": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="3",
            )
        )
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="4",
                runtime=7,
                memory=100,
                categories={"option2": "value2", "option1": "value1"},
            )
        )
        yield db


@pytest.fixture
def small_db_toml(tmp_path, small_db):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d / "slurmise_dir"}"
    db_filename = "{small_db.db_file}"

    [slurmise.job.test_job]
    job_spec = "test_job_spec --option1 {{option1:category}} --option2 {{option2:category}} --filesizes {{filesizes:numeric}}"
    """
    )
    return TomlReturn(p, small_db.db_file)
