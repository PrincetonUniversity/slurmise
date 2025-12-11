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

rule script_rule:
    output:
        "script.txt"
    resources:
        runtime=10,
        mem_mb=20,
    script:
        "script.py"

{append}
    """)

    scriptfile = base_path / "script.py"
    scriptfile.write_text("""
with open(snakemake.output[0], 'w') as outfile:
    outfile.write(f"runtime {snakemake.resources.runtime}\\n")
    outfile.write(f"memory {snakemake.resources.mem_mb}\\n")
    outfile.write(f"threads {snakemake.threads}\\n")
    """)

    return snakefile

SNAKE_RULES = [
    'shell',
    'run',
    'script',
]

def make_slurmise_toml(base_path, append=""):
    toml = base_path / "slurmise.toml"

    toml_text = f"""
[slurmise]
base_dir="{base_path}/slurmise"
"""

    for rule in SNAKE_RULES:
        toml_text += f"""

[slurmise.job.{rule}_rule]
default_mem = 1000
default_time = 30
variables.threads = "numeric"
"""
    toml_text += append

    toml.write_text(toml_text)

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
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 10"
    assert output[1] == "memory 20"
    assert output[2] == "threads 1"


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
def test_snakemake_slurmise_error_benchmark(tmp_path):
    toml = make_slurmise_toml(tmp_path, append="""
[slurmise.job.bench_rule]
default_mem = 1000
default_time = 30
variables.param = "category"
    """)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append="""
rule bench_rule:
    output:
        "bench_{param}.txt"
    benchmark:
        "test/{param}.josnl"
    shell:
        "echo runtime {resources.runtime} > {output}\\n"
        "echo memory {resources.mem_mb} >> {output}\\n"
        "echo threads {threads} >> {output}\\n"

patch_snakemake_workflow(
        slurmise,
        workflow,
        {
            "bench_rule": {
                "param": sp.wildcards("param"),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            },
        },
        keep_benchmarks=True,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        "bench_1.txt",
    ], capture_output=True, text=True)
    assert result.returncode == 1

    assert ('Slurmise needs to set benchmark locations, '
            'remove benchmark for rule bench_rule.') in result.stderr


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
def test_snakemake_slurmise_no_error_benchmark(tmp_path):
    toml = make_slurmise_toml(tmp_path, append="""
[slurmise.job.bench_rule]
default_mem = 1000
default_time = 30
variables.param = "category"
    """)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append="""
rule bench_rule:
    output:
        "bench_{param}.txt"
    benchmark:
        "test/{param}.jsonl"
    shell:
        "echo runtime {resources.runtime} > {output}\\n"
        "echo memory {resources.mem_mb} >> {output}\\n"
        "echo threads {threads} >> {output}\\n"

patch_snakemake_workflow(
        slurmise,
        workflow,
        {
            "bench_rule": {
                "param": sp.wildcards("param"),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            },
        },
        record_benchmarks=False,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        "bench_1.txt",
    ])
    assert result.returncode == 0

    assert (toml.parent / 'test/1.jsonl').exists()


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
@pytest.mark.parametrize("snake_rule", SNAKE_RULES)
def test_snakemake_slurmise_updates_defaults_no_record(snake_rule, tmp_path):
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
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 1"

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
def test_snakemake_slurmise_updates_defaults_with_record(snake_rule, tmp_path):
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
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 1"

    # database exists
    assert (toml.parent / "slurmise/slurmise.h5").exists() is True
    # should be one benchmark file
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


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
def test_snakemake_slurmise_record_params(tmp_path):
    toml = make_slurmise_toml(tmp_path, append="""
[slurmise.job.param_rule]
default_mem = 1000
default_time = 30
variables.param = "category"
    """)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append="""
rule param_rule:
    output:
        "param_{param}.txt"
    params:
        test_param=lambda wildcards: f"it is {wildcards.param}"
    shell:
        "echo runtime {resources.runtime} > {output}\\n"
        "echo memory {resources.mem_mb} >> {output}\\n"
        "echo threads {threads} >> {output}\\n"
        "echo params {params.test_param} >> {output}"

patch_snakemake_workflow(
        slurmise,
        workflow,
        {
            "param_rule": {
                "param": sp.params("test_param"),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            },
        },
        keep_benchmarks=True,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "1",
        "--snakefile",
        snakefile,
        "param_1.txt",
        "param_asdf.txt",
    ])
    assert result.returncode == 0

    outfile = snakefile.parent / "param_1.txt"
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 1"
    assert output[3] == "params it is 1"

    outfile = snakefile.parent / "param_asdf.txt"
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 1"
    assert output[3] == "params it is asdf"

    # database exists
    assert (toml.parent / "slurmise/slurmise.h5").exists() is True
    # should be two benchmark files, store based on param
    benchmark_dir = toml.parent / "slurmise/benchmarks"
    assert benchmark_dir.exists() is True
    assert len(list(benchmark_dir.rglob('*.jsonl'))) == 2

    benchmark_data = {}
    bm_file = benchmark_dir / "param_rule/param:1.jsonl"
    benchmark_data['it is 1'] = json.loads(bm_file.read_text())

    bm_file = benchmark_dir / "param_rule/param:asdf.jsonl"
    benchmark_data['it is asdf'] = json.loads(bm_file.read_text())

    # should have one record matching the file
    slurmise = Slurmise(toml)
    with JobDatabase.get_database(slurmise.configuration.db_filename) as database:
        db = list(database.iterate_database())
        assert len(db) == 2  # one for each category

        for query_jd, jobs in db:
            assert len(jobs) == 1
            job = jobs[0]
            assert job.job_name == 'param_rule'
            bm_dat = benchmark_data[job.categorical['param']]  # don't know order
            assert job.numerical == {}

            memory = 0 if bm_dat['max_rss'] == 'NA' else float(bm_dat['max_rss'] )
            assert job.memory == memory
            runtime = 0 if bm_dat['s'] == 'NA' else float(bm_dat['s']) / 60
            assert job.runtime == runtime


@pytest.mark.skipif(not has_snakemake(), reason="Requires snakemake")
def test_snakemake_slurmise_record_threads(tmp_path):
    # note: the thread_wc has the requested number of threads
    # the thread wc has the actual usage
    # this can differ if the number of available cores is lower than requested
    # the requested value is used for prediction but we record the actual value
    toml = make_slurmise_toml(tmp_path, append="""
[slurmise.job.thread_rule]
default_mem = 1000
default_time = 30
variables.thread = "numeric"
variables.thread_wc = "numeric"
    """)
    snakefile = make_snakefile(tmp_path, slurmise_toml=toml, append="""
rule thread_rule:
    output:
        "thread_{thrd}.txt"
    threads: lambda wildcards: int(wildcards.thrd)
    shell:
        "echo runtime {resources.runtime} > {output}\\n"
        "echo memory {resources.mem_mb} >> {output}\\n"
        "echo threads {threads} >> {output}"

patch_snakemake_workflow(
        slurmise,
        workflow,
        {
            "thread_rule": {
                "thread": sp.threads(),
                "thread_wc": sp.wildcards('thrd'),
                "SLURMISE_runtime_scale": 1,
                "SLURMISE_memory_scale": 1,
            },
        },
        keep_benchmarks=True,
        )
""")

    result = subprocess.run([
        "snakemake",
        "--cores",
        "2",  # to get enough threads
        "--snakefile",
        snakefile,
        "thread_1.txt",
        "thread_2.txt",
        "thread_3.txt",
    ])
    assert result.returncode == 0

    outfile = snakefile.parent / "thread_1.txt"
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 1"

    outfile = snakefile.parent / "thread_2.txt"
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 2"

    outfile = snakefile.parent / "thread_3.txt"
    output = outfile.read_text().split("\n")
    assert output[0] == "runtime 30"
    assert output[1] == "memory 1000"
    assert output[2] == "threads 2"  # only had 2 cores

    # database exists
    assert (toml.parent / "slurmise/slurmise.h5").exists() is True
    # should be two benchmark files, store based on param
    benchmark_dir = toml.parent / "slurmise/benchmarks/thread_rule"
    assert benchmark_dir.exists() is True
    assert len(list(benchmark_dir.rglob('*.jsonl'))) == 3

    benchmark_data = {
        thrd: json.loads((benchmark_dir / f"thrd:{thrd}.jsonl").read_text())
        for thrd in range(1, 4)
    }

    # should have one record matching the file
    slurmise = Slurmise(toml)
    with JobDatabase.get_database(slurmise.configuration.db_filename) as database:
        db = list(database.iterate_database())
        assert len(db) == 1  # one type of job
        query_jd, jobs = db[0]
        assert len(jobs) == 3

        for job in jobs:
            assert job.job_name == 'thread_rule'
            assert job.categorical == {}
            bm_dat = benchmark_data[job.numerical['thread_wc']]  # don't know order

            memory = 0 if bm_dat['max_rss'] == 'NA' else float(bm_dat['max_rss'] )
            assert job.memory == memory
            runtime = 0 if bm_dat['s'] == 'NA' else float(bm_dat['s']) / 60
            assert job.runtime == runtime

            # we only ran with 2 cores.  The 3 core job should have recorded 2 threads here
            assert job.numerical['thread'] == min(job.numerical['thread_wc'], 2)


# TODO: rules with
# pipes
# benchmarks  (should error)
