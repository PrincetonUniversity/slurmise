import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from slurmise.api import Slurmise
from slurmise.job_database import JobDatabase
from slurmise.job_data import JobData


class TomlReturn(NamedTuple):
    toml: str
    db: str


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
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
                categorical={"option1": "value1", "option2": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                numerical={"filesizes": [123, 512, 128]},
                categorical={"option1": "value2"},
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
                categorical={"option2": "value2", "option1": "value1"},
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