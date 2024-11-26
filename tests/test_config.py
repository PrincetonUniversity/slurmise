import pytest
from slurmise.config import SlurmiseConfiguration


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

    [slurmise.job.with_ignore]
    job_prefix = "nothing"
    job_spec = "-T {threads:numeric} -C {complexity:category} -i {ignore}"
    default_mem = 1000
    default_time = 60

    # builtins will include file_size and file_lines
    # specify custom options here
    [slurmise.file_parsers.get_epochs]
    return_type = "numerical"
    awk_script = "'/^epochs:/ {print $2}'"

    [slurmise.file_parsers.fasta_lengths]
    return_type = "numerical"
    awk_file = "/a/path/to/file"
    """

    f.write(toml_str)
    return f

def test_init_SlurmiseConfiguration(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    assert config.slurmise_base_dir == "slurmise_dir"
    assert len(config.jobs) == 2
    assert config.jobs['nupack']['job_prefix'] == "nupack"


def test_parse_job_cmd(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1 -C simple")

    assert job_data.job_name == "nupack"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}


def test_parse_job_cmd_with_ignore(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("with_ignore", "1234", "-T 1 -C simple -i can't see me")

    assert job_data.job_name == "with_ignore"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}

def test_parse_job_cmd_invalid(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command dimer -T 1 -C simple."):
        config.parse_job_cmd("nupack", "1234", "dimer -T 1 -C simple")

def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        config.parse_job_cmd("oldpack", "1234", "monomer -T 1 -C simple")

def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command monomer -T 1A -C simple."):
        config.parse_job_cmd("nupack", "1234", "monomer -T 1A -C simple")
