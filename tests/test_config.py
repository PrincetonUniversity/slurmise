import pytest
from io import StringIO
from slurmise.config import SlurmiseConfiguration, JobSpec
from slurmise import config
from slurmise.job_data import JobData
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
        job_data = config.parse_job_cmd("nupack", "1234", "dimer -T 1 -C simple")

def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        job_data = config.parse_job_cmd("oldpack", "1234", "monomer -T 1 -C simple")

def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command monomer -T 1A -C simple."):
        job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1A -C simple")


# Tests for JobSpec
def test_job_spec_named_ignore():
    spec = JobSpec('cmd -T {threads:numeric} -i {named:ignore}')
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd='cmd -T 10 -i asdf',
            ))
    assert jd == JobData(
        job_name='test',
        cmd='cmd -T 10 -i asdf',
        numerical={'threads': 10},
    )


def test_job_spec_unknown_kind():
    with pytest.raises(ValueError, match="Token kind double is unknown"):
        JobSpec('cmd -T {threads:double}')

def test_job_spec_token_with_no_name():
    with pytest.raises(ValueError, match="Token {numeric} has no name."):
        JobSpec('cmd -T {numeric}')

def test_job_spec_with_builtin_parsers(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file} --input2 {input2:file}"
        file_parsers.input1 = "file_lines"
        file_parsers.input2 = "file_size"
    '''

    file_parsers = {
        'file_lines': config.FileLinesParser(),
        'file_size': config.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {lines:file} --input2 {filesize:file}",
        file_parsers={'lines': 'file_lines', 'filesize': 'file_size'},
        available_parsers=file_parsers,
    )

    input_file = tmp_path / 'input.txt'
    input_file.write_text(
        '''here is
        some lines
        of text'''
    )

    command = f"--input1 {input_file} --input2 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.numerical == {'lines_file_lines': 3, 'filesize_file_size': 42}

def test_job_spec_with_multiple_builtin_parsers(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "file_lines,file_size"
    '''

    file_parsers = {
        'file_lines': config.FileLinesParser(),
        'file_size': config.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'file_lines,file_size'},
        available_parsers=file_parsers,
    )

    input_file = tmp_path / 'input.txt'
    input_file.write_text(
        '''here is
        some lines
        of text'''
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.numerical == {'input1_file_lines': 3, 'input1_file_size': 42}

def test_job_spec_with_awk_parsers(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "epochs,network"

        [slurmise.file_parsers.epochs]
        return_type = "numerical"
        awk_script = "/^epochs:/ {print $2}"

        [slurmise.file_parsers.network]
        return_type = "categorical"
        awk_script = "/^network type:/ {print $3}"
    '''

    file_parsers = {
        'epochs': config.AwkCommandParser('epochs', 'numerical', '/^epochs:/ {print $2 ; exit}'),
        'network': config.AwkCommandParser('network', 'categorical', '/^network type:/ {print $3 ; exit}'),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'epochs,network'},
        available_parsers=file_parsers,
    )

    input_file = tmp_path / 'input.txt'
    input_file.write_text(
'''epochs: 12
network type: conv_NN
network type: IGNORED!
some more text'''
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.categorical == {'input1_network': 'conv_NN'}
    assert jd.numerical == {'input1_epochs': 12}
