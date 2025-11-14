import pytest
import subprocess
import json

from slurmise.api import Slurmise
from slurmise.job_database import JobDatabase

def has_snakemake():
    try:
        import snakemake
        return True
    except ImportError:
        return False

def make_snakefile(base_path, append="", slurmise_toml=None):
    snakefile = base_path / "Snakefile"
    if slurmise_toml is not None:
        append = f"""
from slurmise.api import Slurmise
from slurmise.extras.snake_patching import patch_snakemake_workflow
import slurmise.extras.snake_parsers as sp

slurmise = Slurmise("{slurmise_toml}")
        """ + append
    snakefile.write_text(f"""
workdir: "{base_path}"

rule shell_rule:
    output:
        "shell.txt"
    resources:
        runtime=10,
        mem_mb=20,
    shell:
        "echo runtime {{resources.runtime}} > {{output}}\\n"
        "echo memory {{resources.mem_mb}} >> {{output}}\\n"
        "echo threads {{threads}} >> {{output}}"

rule run_rule:
    output:
        "run.txt"
    resources:
        runtime=10,
        mem_mb=20,
    run:
        with open(output[0], 'w') as outfile:
            outfile.write(f"runtime {{resources.runtime}}\\n")
            outfile.write(f"memory {{resources.mem_mb}}\\n")
            outfile.write(f"threads {{threads}}\\n")

{append}
    """)

    return snakefile

SNAKE_RULES = [
    'shell',
    'run',
]

def make_slurmise_toml(base_path, append=""):
    toml = base_path / "slurmise.toml"
    toml.write_text(f"""
[slurmise]
base_dir="{base_path}/slurmise"

[slurmise.job.shell_rule]
default_mem = 1000
default_time = 30
variables.threads = "numeric"

[slurmise.job.run_rule]
default_mem = 1000
default_time = 30
variables.threads = "numeric"
{append}
""")

    return toml


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
@pytest.mark.parametrize("snake_rule", SNAKE_RULES)
def test_snakemake_shell_no_slurmise(snake_rule, tmp_path):
    """Test if this snakefile will be valid for the version of snakemake."""
    snakefile = make_snakefile(tmp_path)
    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        f"{snake_rule}.txt",
    ])
    assert result.returncode == 0
    outfile = snakefile.parent / f"{snake_rule}.txt"
    shell_output = outfile.read_text().split("\n")
    assert shell_output[0] == "runtime 10"
    assert shell_output[1] == "memory 20"
    assert shell_output[2] == "threads 1"


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
@pytest.mark.parametrize("snake_rule", SNAKE_RULES)
def test_snakemake_shell_slurmise_updates_defaults_no_record(snake_rule, tmp_path):
    toml = make_slurmise_toml(tmp_path)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append=f"""
patch_snakemake_workflow(
        slurmise,
        workflow,
        {{
            "{snake_rule}_rule": {{
                "threads": sp.threads(),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            }},
        }},
        record_benchmarks=False,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        f"{snake_rule}.txt",
    ])
    assert result.returncode == 0
    outfile = snakefile.parent / f"{snake_rule}.txt"
    shell_output = outfile.read_text().split("\n")
    assert shell_output[0] == "runtime 30"
    assert shell_output[1] == "memory 1000"
    assert shell_output[2] == "threads 1"

    # database exists
    assert (toml.parent / "slurmise/slurmise.h5").exists() is True
    # should be no benchmark files
    assert (toml.parent / "slurmise/benchmarks").exists() is False
    # should be no recorded rules
    slurmise = Slurmise(toml)
    with JobDatabase.get_database(slurmise.configuration.db_filename) as database:
        assert list(database.iterate_database()) == []


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
@pytest.mark.parametrize("snake_rule", SNAKE_RULES)
def test_snakemake_shell_slurmise_updates_defaults_with_record(snake_rule, tmp_path):
    toml = make_slurmise_toml(tmp_path)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append=f"""
patch_snakemake_workflow(
        slurmise,
        workflow,
        {{
            "{snake_rule}_rule": {{
                "threads": sp.threads(),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            }},
        }},
        keep_benchmarks=True,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        f"{snake_rule}.txt",
    ])
    assert result.returncode == 0
    outfile = snakefile.parent / f"{snake_rule}.txt"
    shell_output = outfile.read_text().split("\n")
    assert shell_output[0] == "runtime 30"
    assert shell_output[1] == "memory 1000"
    assert shell_output[2] == "threads 1"

    # database exists
    assert (toml.parent / "slurmise/slurmise.h5").exists() is True
    # should be no benchmark files
    assert (toml.parent / "slurmise/benchmarks").exists() is True
    bm_file = toml.parent / f"slurmise/benchmarks/{snake_rule}_rule/{snake_rule}_rule.jsonl"
    benchmark_data = json.loads(bm_file.read_text())

    # should have one record matching the file
    slurmise = Slurmise(toml)
    with JobDatabase.get_database(slurmise.configuration.db_filename) as database:
        db = list(database.iterate_database())
        assert len(db) == 1  # one type of job
        query_jd, jobs = db[0]
        assert len(jobs) == 1
        job = jobs[0]

        assert job.job_name == f'{snake_rule}_rule'
        assert job.categorical == {}
        assert job.numerical == {'threads': 1}

        memory = 0 if benchmark_data['max_rss'] == 'NA' else float(benchmark_data['max_rss'] )
        assert job.memory == memory
        runtime = 0 if benchmark_data['s'] == 'NA' else float(benchmark_data['s']) / 60
        assert job.runtime == runtime
