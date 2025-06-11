import pytest
from pathlib import Path
import shutil
from collections import namedtuple


TomlReturn = namedtuple('TomlReturn', ['toml', 'db'])


@pytest.fixture
def simple_toml(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d/'slurmise_dir'}"

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
    base_dir = "{d/'slurmise_dir'}"
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
    base_dir = "{d/'slurmise_dir'}"
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
