import gzip
import shutil

import pytest

from slurmise.job_data import JobData
from slurmise.job_parse import file_parsers
from slurmise.job_parse.job_specification import JobSpec


# Tests for JobSpec
def test_job_spec_named_ignore():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd -T {threads} -i {ignore}")
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd="cmd -T 10 -i asdf",
        )
    )
    assert jd == JobData(
        job_name="test",
        cmd="cmd -T 10 -i asdf",
        numerics={"threads": 10},
    )


def test_job_spec_unknown_kind():
    with pytest.raises(ValueError, match="Unknown variable type double for variable threads"):
        JobSpec({"threads": {"type": "double"}})


def test_job_spec_category_returning_parser_is_not_a_numeric():
    """file_basename returns a category, so a file variable alone leaves nothing to fit on."""
    with pytest.raises(ValueError, match="at least one numeric variable"):
        JobSpec(
            {"input1": {"type": "file", "file_parsers": "file_basename"}},
            available_parsers={"file_basename": file_parsers.FileBasename()},
        )


def test_job_spec_numeric_returning_parser_satisfies_the_requirement():
    """file_size returns a numeric, so no numeric variable has to be declared alongside it."""
    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": "file_size"}},
        available_parsers={"file_size": file_parsers.FileSizeParser()},
    )

    assert spec.token_kinds == {"input1": "file"}


def test_basic_job_spec():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd -T {threads}")

    jd = spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -T 3"))
    assert jd.numerics == {"threads": 3}


def test_basic_job_spec_from_dict():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd -T {threads}")

    jd = spec.parse_job_from_dict({"threads": 3}, JobData(job_name="test"))
    assert jd.numerics == {"threads": 3}


def test_basic_job_spec_from_dict_extra_in_dict():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd -T {threads}")

    with pytest.raises(ValueError, match="Dict contained extra variable: 'extra'"):
        spec.parse_job_from_dict(
            {"threads": 3, "extra": "something"},
            JobData(job_name="test"),
        )


def test_basic_job_spec_from_dict_missing_in_dict():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd -T {threads}")

    with pytest.raises(ValueError, match="Dict missing variable: 'threads'"):
        spec.parse_job_from_dict(
            {},
            JobData(job_name="test"),
        )


def test_job_spec_failure_swap():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -S simple -T 5"))
    print(f"\n{ve.value}")


def test_job_spec_failure_typos():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cnd -t 5 -S simple"))
    print(f"\n{ve.value}")


def test_basic_job_spec_extra_cmd_prefix():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="extra cmd -T 3 -S beep"))
    print(f"\n{ve.value}")


def test_basic_job_spec_extra_spec_prefix():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("extra cmd -T {threads} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -T 3 -S beep"))
    print(f"\n{ve.value}")


def test_basic_job_spec_extra_spec_suffix():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another} extra")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -T 3 -S cat"))
    print(f"\n{ve.value}")


def test_basic_job_spec_extra_spec_internal():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} extra -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -T 3 -S cat"))
    print(f"\n{ve.value}")


def test_basic_job_spec_extra_cmd_internal():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd -T 3 extra -S cat"))
    print(f"\n{ve.value}")


def test_basic_job_spec_with_ignore():
    spec = JobSpec(
        {
            "threads": {"type": "numeric"},
            "another": {"type": "category"},
            "named": {"type": "ignore"},
        }
    )
    spec.add_job_spec("cmd {named} -T {threads} {ignore} -S {another}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:") as ve:
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmd ignore me please -T 3 and this too -s cat"))
    print(f"\n{ve.value}")


def test_try_exact_passes():
    spec = JobSpec(
        {
            "threads": {"type": "numeric"},
            "another": {"type": "category"},
        }
    )
    spec.add_job_spec("cmd -T {threads} -S {another}")
    result = spec.align_and_indicate_differences("cmd -T 3 -S cat", try_exact_match=True)
    print(f"\n{result}")
    assert result.startswith("Able to parse")


def test_try_exact_fails():
    spec = JobSpec(
        {
            "threads": {"type": "numeric"},
            "another": {"type": "category"},
        }
    )
    spec.add_job_spec("cmd -T {threads} -S {another}")
    result = spec.align_and_indicate_differences("FAILURE -T 3 -S cat", try_exact_match=True)
    print(f"\n{result}")
    assert result.startswith("Failed to parse")


def test_align_fallback_when_too_different():
    """When the command is too far from the spec for fuzzy matching, fall back to
    a plain character-level diff rather than raising an error."""
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    result = spec.align_and_indicate_differences("completely unrelated input xyz")
    assert isinstance(result, str)
    assert result  # non-empty


def test_align_fallback_long_spec_wrong_cmd():
    """Long specs with a completely wrong command must complete within the built-in 2s timeout."""
    import time

    spec = JobSpec(
        {
            "cpus": {"type": "numeric"},
            "query": {"type": "category"},
            "footprint": {"type": "category"},
            "maps": {"type": "category"},
            "bands": {"type": "category"},
            "iters": {"type": "numeric"},
        }
    )
    spec.add_job_spec(
        "--cpu_bind=cores --export=ALL --ntasks-per-node={cpus}"
        " --cpus-per-task=8 so-site-pipeline make-ml-map {query}"
        " {footprint} {ignore} --comps={maps} -C {ignore}"
        " --bands={bands} --maxiter={iters} -v --tiled=1 --site act"
    )

    start = time.monotonic()
    result = spec.align_and_indicate_differences("wrong cmd")
    elapsed = time.monotonic() - start

    assert elapsed < 3.0, f"align_and_indicate_differences took {elapsed:.1f}s (expected < 3s)"
    assert result


def test_align_fuzzy_timeout_long_cmd():
    """A long command that would make fuzzy matching hang must still return within ~2s."""
    import time

    spec = JobSpec(
        {
            "cpus": {"type": "numeric"},
            "query": {"type": "category"},
            "footprint": {"type": "category"},
            "maps": {"type": "category"},
            "bands": {"type": "category"},
            "iters": {"type": "numeric"},
        }
    )
    spec.add_job_spec(
        "--cpu_bind=cores --export=ALL --ntasks-per-node={cpus}"
        " --cpus-per-task=8 so-site-pipeline make-ml-map {query}"
        " {footprint} {ignore} --comps={maps} -C {ignore}"
        " --bands={bands} --maxiter={iters} -v --tiled=1 --site act"
    )
    # A near-match long command where fuzzy matching would previously hang
    cmd = (
        "--cpu_bind=cores --export=ALL --ntasks-per-node=1"
        " --cpus-per-task=8 so-site-pipeline make-ml-map timestamp_start"
        " somefile.fits output --executable so-site-pipeline"
        " --comps=context.yaml -C context.yaml"
        " --bands=aband"
        " --maxiter=10 -v --tiled=1 --site act --extra-flag-that-breaks-it"
    )

    start = time.monotonic()
    result = spec.align_and_indicate_differences(cmd, try_exact_match=True)
    elapsed = time.monotonic() - start

    assert elapsed < 3.0, f"align_and_indicate_differences took {elapsed:.1f}s (expected < 3s)"
    assert result
    print(f"\n[elapsed: {elapsed:.2f}s]\n{result}")


def test_align_no_anchor_corruption():
    """Stripping anchors before fuzzy matching must not corrupt group capture values."""
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})
    spec.add_job_spec("cmd -T {threads} -S {another}")

    # Exact match: groups should be filled correctly
    result = spec.align_and_indicate_differences("cmd -T 3 -S cat", try_exact_match=True)
    assert "Able to parse" in result
    # The captured value should appear in the annotated output
    assert "3" in result
    assert "cat" in result


def test_long_job_spec():
    spec = JobSpec(
        {
            "cpus": {"type": "numeric"},
            "query": {"type": "category"},
            "footprint": {"type": "category"},
            "maps": {"type": "category"},
            "bands": {"type": "category"},
            "iters": {"type": "numeric"},
        }
    )
    spec.add_job_spec(
        "--cpu_bind=cores --export=ALL --ntasks-per-node={cpus}"
        " --cpus-per-task=8 so-site-pipeline make-ml-map {query}"
        " {footprint} {ignore} --comps={maps} -C {ignore}"
        " --bands={bands} --maxiter={iters} -v --tiled=1 --site act"
    )
    # The {ignore} tokens absorb the extra --executable flag; the command parses exactly.
    cmd = (
        "--cpu_bind=cores --export=ALL --ntasks-per-node=1"
        " --cpus-per-task=8 so-site-pipeline make-ml-map timestamp_start"
        " somefile.fits output --executable so-site-pipeline"
        " --comps=context.yaml -C context.yaml"
        " --bands=aband"
        " --maxiter=10 -v --tiled=1 --site act"
    )

    result = spec.align_and_indicate_differences(cmd, try_exact_match=True)
    print(f"\n{result}")
    assert result.startswith("Able to parse")

    jd = spec.parse_job_cmd(JobData(job_name="test", cmd=cmd))
    assert jd.numerics == {"cpus": 1.0, "iters": 10.0}
    assert jd.categories == {
        "query": "timestamp_start",
        "footprint": "somefile.fits",
        "maps": "context.yaml",
        "bands": "aband",
    }


def test_job_spec_with_no_file_parser(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file"}
    """
    available_parsers = {
        "file_basename": file_parsers.FileBasename(),
    }

    with pytest.raises(ValueError, match="File 'input1' has no assigned file parser"):
        JobSpec(
            {"input1": {"type": "file"}},
            available_parsers=available_parsers,
        )


def test_job_spec_with_parser_not_available(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers="file_bassname"}
    """
    available_parsers = {
        "file_basename": file_parsers.FileBasename(),
    }

    with pytest.raises(ValueError, match=("The parser 'file_bassname' is not available for file 'input1'")):
        JobSpec(
            {"input1": {"type": "file", "file_parsers": "file_bassname"}},
            available_parsers=available_parsers,
        )


def test_job_spec_with_builtin_parsers_basename(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers="file_basename"}
    """
    available_parsers = {
        "file_basename": file_parsers.FileBasename(),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": "file_basename"}, "input2": {"type": "numeric"}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1} --input2 {input2}")

    input_file = tmp_path / "input.txt"

    command = f"--input1 {input_file} --input2 3"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"input2": 3.0}
    assert jd.categories == {"input1_file_basename": "input.txt"}

    jd = spec.parse_job_from_dict(
        {"input1": input_file, "input2": 3},
        JobData(
            job_name="test",
        ),
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"input2": 3.0}
    assert jd.categories == {"input1_file_basename": "input.txt"}


def test_job_spec_with_builtin_parsers_md5hash(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers="file_md5"}
    """
    available_parsers = {
        "file_md5": file_parsers.FileMD5(),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": "file_md5"}, "input2": {"type": "numeric"}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1} --input2 {input2}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """here is
        some lines
        of text"""
    )
    test_file = tmp_path / "test.txt"
    test_file.write_text(
        """here is
        some lines
        of text"""
    )

    command = f"--input1 {input_file} --input2 3"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"input2": 3.0}

    jd_test = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=f"--input1 {test_file} --input2 3",
        )
    )
    # test that md5 digest reflects file content
    assert jd.categories == jd_test.categories


def test_job_spec_with_builtin_parsers(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1} --input2 {input2}"
    [slurmise.job.builtin_files.variables]
    lines = {type = "file", file_parsers="file_lines"}
    filesize = {type = "file", file_parsers="file_size"}
    """

    available_parsers = {
        "file_lines": file_parsers.FileLinesParser(),
        "file_size": file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        {
            "lines": {"type": "file", "file_parsers": "file_lines"},
            "filesize": {"type": "file", "file_parsers": "file_size"},
        },
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {lines} --input2 {filesize}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """here is
        some lines
        of text"""
    )

    command = f"--input1 {input_file} --input2 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"lines_file_lines": 3, "filesize_file_size": 42}


def test_job_spec_with_builtin_parsers_gzipped(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1} --input2 {input2}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "gzip_file", file_parsers="file_lines"}
    input2 = {type = "gzip_file", file_parsers="file_size"}
    """

    available_parsers = {
        "file_lines": file_parsers.FileLinesParser(),
        "file_size": file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        {
            "lines": {"type": "gzip_file", "file_parsers": "file_lines"},
            "filesize": {"type": "gzip_file", "file_parsers": "file_size"},
        },
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {lines} --input2 {filesize}")

    input_file = tmp_path / "input.txt.gz"
    with gzip.open(input_file, "wt") as infile:
        for _ in range(100):
            infile.write(
                """here is
                some lines
                of text"""
            )

    command = f"--input1 {input_file} --input2 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"lines_file_lines": 201, "filesize_file_size": 99}


def test_job_spec_with_builtin_parsers_file_list(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {lines}"
    [slurmise.job.builtin_files.variables]
    lines = {type = "file_list", file_parsers=["file_lines", "file_size"]}
    """

    available_parsers = {
        "file_lines": file_parsers.FileLinesParser(),
        "file_size": file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        {"lines": {"type": "file_list", "file_parsers": ["file_lines", "file_size"]}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {lines}")

    file_list = tmp_path / "listing.txt"
    with file_list.open("w") as fl:
        for i in range(5):
            input_file = tmp_path / f"input_{i}.txt"
            fl.write(f"{input_file}\n")
            input_file.write_text(
                """here is
                some lines
                of text"""
                * (5 * (i + 1))
            )

    command = f"--input1 {file_list}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {
        "lines_file_lines": [10 * i + 1 for i in range(1, 6)],
        "lines_file_size": [290 * i for i in range(1, 6)],
    }


def test_job_spec_with_multiple_builtin_parsers(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers=["file_lines", "file_size"]}
    """

    available_parsers = {
        "file_lines": file_parsers.FileLinesParser(),
        "file_size": file_parsers.FileSizeParser(),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": ["file_lines", "file_size"]}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """here is
        some lines
        of text"""
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.numerics == {"input1_file_lines": 3, "input1_file_size": 42}


def test_job_spec_with_awk_parsers(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers=["epochs", "network"]}

    [slurmise.file_parsers.epochs]
    type = "numeric"
    awk_script = "/^epochs:/ {print $2}"

    [slurmise.file_parsers.network]
    type = "category"
    awk_script = "/^network type:/ {print $3}"
    """

    available_parsers = {
        "epochs": file_parsers.AwkParser("epochs", "numeric", "/^epochs:/ {print $2 ; exit}"),
        "network": file_parsers.AwkParser("network", "category", "/^network type:/ {print $3 ; exit}"),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": ["epochs", "network"]}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """epochs: 12
network type: conv_NN
network type: IGNORED!
some more text"""
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.categories == {"input1_network": "conv_NN"}
    assert jd.numerics == {"input1_epochs": [12]}


def test_job_spec_with_awk_parsers_multiple_numerics(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers="layers"}

    [slurmise.file_parsers.layers]
    type = "numeric"
    awk_script = "/^layers:/ {print $2}"
    """

    available_parsers = {
        "layers": file_parsers.AwkParser("layers", "numeric", '/^layers:/ {$1="" ; print $0}'),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": "layers"}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """layers: 12
layers: 14
layers: 16
layers: 18 24 36
some more text"""
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.categories == {}
    assert jd.numerics == {"input1_layers": [12, 14, 16, 18, 24, 36]}


@pytest.mark.skipif(
    not shutil.which("awk"),
    reason="AWK is not available on this system",
)
def test_job_spec_with_awk_file(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    [slurmise.job.builtin_files.variables]
    input1 = {type = "file", file_parsers=["fasta_inline", "fasta_script"]}

    [slurmise.file_parsers.fasta_inline]
    type = "numeric"
    awk_script = "/^layers:/ {print $2}"

    [slurmise.file_parsers.fasta_script]
    type = "numeric"
    awk_script = "/path/to/awk/file.awk"
    script_is_file = True
    """

    awk_script = """ /^>/ {if (seq) print seq; seq=0}
/^>/ {next}
{seq = seq + length($0)}

END {if (seq) print seq}
"""
    awk_file = tmp_path / "parse_fasta.awk"
    awk_file.write_text(awk_script)

    available_parsers = {
        "fasta_inline": file_parsers.AwkParser("fasta_inline", "numeric", awk_script),
        "fasta_script": file_parsers.AwkParser("fasta_script", "numeric", awk_file, script_is_file=True),
    }

    spec = JobSpec(
        {"input1": {"type": "file", "file_parsers": ["fasta_inline", "fasta_script"]}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1}")

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        """>sequence 1
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
"""
    )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.categories == {}
    assert jd.numerics == {
        "input1_fasta_inline": [40, 25, 43, 1],
        "input1_fasta_script": [40, 25, 43, 1],
    }


@pytest.mark.skipif(
    not shutil.which("awk"),
    reason="AWK is not available on this system",
)
def test_job_spec_with_awk_gzip_file(tmp_path):
    """
    [slurmise.job.builtin_files]
    job_spec = "--input1 {input1}"
    input1 = {type = "gzip_file", file_parsers=["fasta_inline", "fasta_script"]}

    [slurmise.file_parsers.fasta_inline]
    type = "numeric"
    awk_script = "/^layers:/ {print $2}"

    [slurmise.file_parsers.fasta_script]
    type = "numeric"
    awk_script = "/path/to/awk/file.awk"
    script_is_file = True
    """

    awk_script = """ /^>/ {if (seq) print seq; seq=0}
/^>/ {next}
{seq = seq + length($0)}
END {if (seq) print seq}
"""
    awk_file = tmp_path / "parse_fasta.awk"
    awk_file.write_text(awk_script)

    available_parsers = {
        "fasta_inline": file_parsers.AwkParser("fasta_inline", "numeric", awk_script),
        "fasta_script": file_parsers.AwkParser("fasta_script", "numeric", awk_file, script_is_file=True),
    }

    spec = JobSpec(
        {"input1": {"type": "gzip_file", "file_parsers": ["fasta_inline", "fasta_script"]}},
        available_parsers=available_parsers,
    )
    spec.add_job_spec("--input1 {input1}")

    input_file = tmp_path / "input.txt.gz"
    with gzip.open(input_file, "wt") as infile:
        infile.write(
            """>sequence 1
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
"""
        )

    command = f"--input1 {input_file}"
    jd = spec.parse_job_cmd(
        JobData(
            job_name="test",
            cmd=command,
        )
    )
    assert jd.job_name == "test"
    assert jd.categories == {}
    assert jd.numerics == {
        "input1_fasta_inline": [40, 25, 43, 1],
        "input1_fasta_script": [40, 25, 43, 1],
    }


def test_job_spec_dot_in_literal_matches_exactly():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd.super -T {threads}")

    jd = spec.parse_job_cmd(JobData(job_name="test", cmd="cmd.super -T 3"))
    assert jd.numerics == {"threads": 3}


def test_job_spec_dot_in_literal_does_not_over_match():
    spec = JobSpec({"threads": {"type": "numeric"}})
    spec.add_job_spec("cmd.super -T {threads}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:"):
        spec.parse_job_cmd(JobData(job_name="test", cmd="cmdXsuper -T 3"))


def test_job_spec_pipe_in_literal():
    spec = JobSpec({"output": {"type": "category"}, "threads": {"type": "numeric"}})
    spec.add_job_spec("find -name '*.out' | grep {output} -T {threads}")

    jd = spec.parse_job_cmd(JobData(job_name="test", cmd="find -name '*.out' | grep help -T 2"))
    assert jd.categories == {"output": "help"}
    assert jd.numerics == {"threads": 2.0}


def test_job_spec_pipe_in_literal_does_not_over_match():
    spec = JobSpec({"output": {"type": "category"}, "threads": {"type": "numeric"}})
    spec.add_job_spec("find -name '*.out' | grep {output} -T {threads}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:"):
        spec.parse_job_cmd(JobData(job_name="test", cmd="find -name '*.out' X grep help -T 2"))


def test_job_spec_literal_brace_in_awk():
    spec = JobSpec({"input": {"type": "category"}, "threads": {"type": "numeric"}})
    spec.add_job_spec("awk '{{print $1}}' {input} -T {threads}")

    jd = spec.parse_job_cmd(JobData(job_name="test", cmd="awk '{print $1}' data.txt -T 2"))
    assert jd.categories == {"input": "data.txt"}
    assert jd.numerics == {"threads": 2.0}


def test_job_spec_literal_brace_does_not_match_missing_brace():
    spec = JobSpec({"input": {"type": "category"}, "threads": {"type": "numeric"}})
    spec.add_job_spec("awk '{{print $1}}' {input} -T {threads}")

    with pytest.raises(ValueError, match="Job spec for test does not match command:"):
        spec.parse_job_cmd(JobData(job_name="test", cmd="awk 'print $1' data.txt -T 2"))


def test_job_spec_only_escaped_braces_raises():
    spec = JobSpec({"threads": {"type": "numeric"}})
    with pytest.raises(ValueError, match="Job specification contains no variables"):
        spec.add_job_spec("awk '{{print $1}}'")


def test_job_spec_stores_model():
    spec = JobSpec({"threads": {"type": "numeric"}}, model={"model": "knn"})
    assert spec.model == {"model": "knn"}


def test_job_spec_model_default_is_none():
    spec = JobSpec({"threads": {"type": "numeric"}})
    assert spec.model is None


def test_variable_sources():
    spec = JobSpec({"threads": {"type": "numeric"}, "another": {"type": "category"}})

    with pytest.raises(ValueError, match="Variables do not match source specification"):
        spec.get_sources()

    spec = JobSpec(
        {
            "threads": {"type": "numeric", "source": "threads"},
            "another": {"type": "category", "source": "params", "key": "another"},
        }
    )

    result = spec.get_sources()
    assert result == {"threads": "threads", "another": ("params", "another")}
