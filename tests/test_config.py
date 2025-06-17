import pytest
from slurmise.config import SlurmiseConfiguration
from slurmise.job_parse import file_parsers


@pytest.fixture
def basic_toml(tmpdir):
    d = tmpdir.mkdir("slurmise_dir")
    f = d.join("basic.toml")

    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"
    default_mem = 2000
    default_time = 70

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads:numeric} -C {complexity:category}"
    default_mem = 3000
    default_time = 80

    [slurmise.job.with_ignore]
    job_prefix = "nothing"
    job_spec = "-T {threads:numeric} -C {complexity:category} -i {ignore}"

    # builtins will include file_size and file_lines
    # specify custom options here
    [slurmise.file_parsers.get_epochs]
    return_type = "numerical"
    awk_script = "'/^epochs:/ {print $2}'"

    [slurmise.file_parsers.fasta_lengths]
    return_type = "numerical"
    awk_script = "/a/path/to/file"
    script_is_file = true

    # categorical default return type
    [slurmise.file_parsers.script_string]
    awk_script = "/^>/"
    script_is_file = false

    # this is ignored in parsing as the argument doesn't match an awk parser
    [slurmise.file_parsers.unknown_type]
    no_awk_script = "/^>/"
    script_is_file = false
    """

    f.write(toml_str)
    return f


@pytest.fixture
def basic_toml_no_default(tmpdir):
    d = tmpdir.mkdir("slurmise_dir")
    f = d.join("basic.toml")

    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads:numeric} -C {complexity:category}"
    """

    f.write(toml_str)
    return f

def test_init_SlurmiseConfiguration(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    assert config.slurmise_base_dir == "slurmise_dir"
    assert len(config.jobs) == 2
    assert config.jobs['with_ignore']['job_prefix'] == "nothing"


def test_parse_job_cmd(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("monomer -T 1 -C simple", "nupack", "1234")

    assert job_data.job_name == "nupack"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}


def test_parse_job_cmd_with_ignore(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("-T 1 -C simple -i can't see me", "with_ignore", "1234")

    assert job_data.job_name == "with_ignore"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}

def test_parse_job_cmd_invalid(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec for nupack does not match command:") as ve:
        config.parse_job_cmd("dimer -T 1 -C simple", "nupack", "1234")
    print(f'\n{ve.value}')

def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        config.parse_job_cmd("monomer -T 1 -C simple", "oldpack", "1234")

def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec for nupack does not match command:") as ve:
        config.parse_job_cmd("monomer -T 1A -C simple", "nupack", "1234")
    print(f'\n{ve.value}')

def test_awk_parsers(basic_toml):
    config = SlurmiseConfiguration(basic_toml)

    assert config.file_parsers == {
        'file_size': file_parsers.FileSizeParser(),
        'file_lines': file_parsers.FileLinesParser(),
        'file_basename': file_parsers.FileBasename(),
        'file_md5': file_parsers.FileMD5(),
        'get_epochs': file_parsers.AwkParser('get_epochs', 'numerical', "'/^epochs:/ {print $2}'", False),
        'fasta_lengths': file_parsers.AwkParser('fasta_lengths', 'numerical', "/a/path/to/file", True),
        'script_string': file_parsers.AwkParser('script_string', 'categorical', "/^>/", False),
    }

def test_parse_job_cmd_inference(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Unable to match job name to 'sort infile'"):
        config.parse_job_cmd('sort infile')

    match_prefix = config.parse_job_cmd('nothing -T 3 -C high -i something')
    assert match_prefix.job_name == 'with_ignore'

    match_name = config.parse_job_cmd('nupack monomer -T 3 -C high')
    assert match_name.job_name == 'nupack'

def test_default_resources_slurmise_base(basic_toml):
    '''Test the default can be set at the slurmise level for all jobs without additional defaults.'''
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd('nothing -T 3 -C high -i something')
    config.add_defaults(job_data)

    assert job_data.memory == 2000
    assert job_data.runtime == 70

    # nupack job has defaults overwritten
    job_data = config.parse_job_cmd('nupack monomer -T 3 -C high')
    config.add_defaults(job_data)

    assert job_data.memory == 3000
    assert job_data.runtime == 80


def test_default_resources_no_setting(basic_toml_no_default):
    '''Test the default can be set at the slurmise level for all jobs without additional defaults.'''
    config = SlurmiseConfiguration(basic_toml_no_default)
    job_data = config.parse_job_cmd('nupack monomer -T 3 -C high')
    config.add_defaults(job_data)

    assert job_data.memory == 1000
    assert job_data.runtime == 60
