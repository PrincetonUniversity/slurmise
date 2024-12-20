import pytest
import gzip
from slurmise.job_parse.job_specification import JobSpec
from slurmise.job_parse import file_parsers
from slurmise.job_data import JobData


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

    available_parsers = {
        'file_lines': file_parsers.FileLinesParser(),
        'file_size': file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {lines:file} --input2 {filesize:file}",
        file_parsers={'lines': 'file_lines', 'filesize': 'file_size'},
        available_parsers=available_parsers,
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

def test_job_spec_with_builtin_parsers_gzipped(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:gzip_file} --input2 {input2:gzip_file}"
        file_parsers.input1 = "file_lines"
        file_parsers.input2 = "file_size"
    '''

    available_parsers = {
        'file_lines': file_parsers.FileLinesParser(),
        'file_size': file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {lines:gzip_file} --input2 {filesize:gzip_file}",
        file_parsers={'lines': 'file_lines', 'filesize': 'file_size'},
        available_parsers=available_parsers,
    )

    input_file = tmp_path / 'input.txt.gz'
    with gzip.open(input_file, 'wt') as infile:
        for _ in range(100):
            infile.write(
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
    assert jd.numerical == {'lines_file_lines': 201, 'filesize_file_size': 99}

def test_job_spec_with_builtin_parsers_file_list(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file_list}"
        file_parsers.input1 = "file_lines,file_size"
    '''

    available_parsers = {
        'file_lines': file_parsers.FileLinesParser(),
        'file_size': file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {lines:file_list}",
        file_parsers={'lines': 'file_lines,file_size'},
        available_parsers=available_parsers,
    )

    file_list = tmp_path / 'listing.txt'
    with file_list.open('w') as fl:
        for i in range(5):
            input_file = tmp_path / f'input_{i}.txt'
            fl.write(f'{input_file}\n')
            input_file.write_text(
                '''here is
                some lines
                of text''' * (5*(i+1))
            )

    command = f"--input1 {file_list}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.numerical == {
        'lines_file_lines': [10*i + 1 for i in range(1, 6)],
        'lines_file_size': [290*i for i in range(1, 6)],
    }

def test_job_spec_with_multiple_builtin_parsers(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "file_lines,file_size"
    '''

    available_parsers = {
        'file_lines': file_parsers.FileLinesParser(),
        'file_size': file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'file_lines,file_size'},
        available_parsers=available_parsers,
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
        job_spec = "--input1 {input1:gzip_file}"
        job_spec = "--input1 {input1:file_list}"
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "epochs,network"

        [slurmise.file_parsers.epochs]
        return_type = "numerical"
        awk_script = "/^epochs:/ {print $2}"

        [slurmise.file_parsers.network]
        return_type = "categorical"
        awk_script = "/^network type:/ {print $3}"
    '''

    available_parsers = {
        'epochs': file_parsers.AwkParser('epochs', 'numerical', '/^epochs:/ {print $2 ; exit}'),
        'network': file_parsers.AwkParser('network', 'categorical', '/^network type:/ {print $3 ; exit}'),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'epochs,network'},
        available_parsers=available_parsers,
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
    assert jd.numerical == {'input1_epochs': [12]}

def test_job_spec_with_awk_parsers_multiple_numerics(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "layers"

        [slurmise.file_parsers.layers]
        return_type = "numerical"
        awk_script = "/^layers:/ {print $2}"
    '''

    available_parsers = {
        'layers': file_parsers.AwkParser('layers', 'numerical', '/^layers:/ {$1="" ; print $0}'),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'layers'},
        available_parsers=available_parsers,
    )

    input_file = tmp_path / 'input.txt'
    input_file.write_text(
'''layers: 12
layers: 14
layers: 16
layers: 18 24 36
some more text'''
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.categorical == {}
    assert jd.numerical == {'input1_layers': [12, 14, 16, 18, 24, 36]}


def test_job_spec_with_awk_file(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:file}"
        file_parsers.input1 = "fasta_inline,fasta_script"

        [slurmise.file_parsers.fasta_inline]
        return_type = "numerical"
        awk_script = "/^layers:/ {print $2}"

        [slurmise.file_parsers.fasta_script]
        return_type = "numerical"
        awk_script = "/path/to/awk/file.awk"
        script_is_file = True
    '''

    awk_script = ''' /^>/ {if (seq) print seq; seq=0} 
/^>/ {next} 
{seq = seq + length($0)} 
END {if (seq) print seq}
'''
    awk_file = tmp_path / 'parse_fasta.awk'
    awk_file.write_text(awk_script)

    available_parsers = {
        'fasta_inline': file_parsers.AwkParser(
            'fasta_inline', 'numerical', awk_script),
        'fasta_script': file_parsers.AwkParser(
            'fasta_script', 'numerical', awk_file, script_is_file=True),
    }

    spec = JobSpec(
        "--input1 {input1:file}",
        file_parsers={'input1': 'fasta_inline,fasta_script'},
        available_parsers=available_parsers,
    )

    input_file = tmp_path / 'input.txt'
    input_file.write_text(
'''>sequence 1
1234567890
1234567890
1234567890
1234567890
>sequence 2
1234567890
1234567890
12345
>sequence 3
1234567890
1234567890
1234567890
1234567890
123
>sequence 4
1
'''
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.categorical == {}
    assert jd.numerical == {
        'input1_fasta_inline': [40, 25, 43, 1],
        'input1_fasta_script': [40, 25, 43, 1],
    }


def test_job_spec_with_awk_gzip_file(tmp_path):
    '''
        [slurmise.job.builtin_files]
        job_spec = "--input1 {input1:gzip_file}"
        file_parsers.input1 = "fasta_inline,fasta_script"

        [slurmise.file_parsers.fasta_inline]
        return_type = "numerical"
        awk_script = "/^layers:/ {print $2}"

        [slurmise.file_parsers.fasta_script]
        return_type = "numerical"
        awk_script = "/path/to/awk/file.awk"
        script_is_file = True
    '''

    awk_script = ''' /^>/ {if (seq) print seq; seq=0} 
/^>/ {next} 
{seq = seq + length($0)} 
END {if (seq) print seq}
'''
    awk_file = tmp_path / 'parse_fasta.awk'
    awk_file.write_text(awk_script)

    available_parsers = {
        'fasta_inline': file_parsers.AwkParser(
            'fasta_inline', 'numerical', awk_script),
        'fasta_script': file_parsers.AwkParser(
            'fasta_script', 'numerical', awk_file, script_is_file=True),
    }

    spec = JobSpec(
        "--input1 {input1:gzip_file}",
        file_parsers={'input1': 'fasta_inline,fasta_script'},
        available_parsers=available_parsers,
    )

    input_file = tmp_path / 'input.txt.gz'
    with gzip.open(input_file, 'wt') as infile:
        infile.write(
'''>sequence 1
1234567890
1234567890
1234567890
1234567890
>sequence 2
1234567890
1234567890
12345
>sequence 3
1234567890
1234567890
1234567890
1234567890
123
>sequence 4
1
'''
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd=command,
            ))
    assert jd.job_name == 'test'
    assert jd.categorical == {}
    assert jd.numerical == {
        'input1_fasta_inline': [40, 25, 43, 1],
        'input1_fasta_script': [40, 25, 43, 1],
    }
