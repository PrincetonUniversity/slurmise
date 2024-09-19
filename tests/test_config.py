import pytest
from io import StringIO
from slurmise.config import SlurmiseConfiguration
from pprint import pprint

@pytest.fixture
def basic_toml(tmpdir):
    d = tmpdir.mkdir("slurmise_dir")
    f = d.join("basic.toml")

    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_prefix = "nupack"
    job_spec = "monomer -T {threads:numeric} -C {complexity:category}"
    default_mem = 1000
    default_time = 60
    """

    f.write(toml_str)
    return f

def test_init_SlurmiseConfiguration(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    assert config.slurmise_base_dir == "slurmise_dir"
    assert len(config.jobs) == 1
    assert config.jobs['nupack']['job_prefix'] == "nupack"


def test_parse_job_cmd(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1 -C simple")

    assert job_data.job_name == "nupack"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}

def test_parse_job_cmd_invalid(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command dimer -T 1 -C simple."):
        job_data = config.parse_job_cmd("nupack", "1234", "dimer -T 1 -C simple")

def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        job_data = config.parse_job_cmd("oldpack", "1234", "monomer -T 1 -C simple")

def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command monomer -T 1A -C simple."):
        job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1A -C simple")