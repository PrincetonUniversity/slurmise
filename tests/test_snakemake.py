import pytest

from slurmise.job_data import JobData


@pytest.fixture
def snakemake_benchmark(tmp_path):

    d = tmp_path
    p = d / "snakemake_rule_benchmark.tsv"

    p.write_text("""s	h:m:s	max_rss	max_vms	max_uss	max_pss	io_in	io_out	mean_load	cpu_time	jobid	rule_name	wildcards	params	threads	cpu_usage	resources	input_size_mb
    2050.5583	0:34:10	46.58	63.41	35.42	36.94	0.10	0.00	98.52	2020.59	0	PRIME	{'branch': 'test', 'properties': 'Random-5', 'gene_name': 'ZWINT'}	{'hyphy': '~/local/bin/hyphy'}	1	202025.13	{'_cores': 1, '_nodes': 1, 'mem_mb': 2048, 'mem_mib': 1954, 'disk_mb': 1000, 'disk_mib': 954, 'tmpdir': '/tmp', 'cpus_per_tasks': '{rule.threads}', 'job_name': '{name}.{jobid}', 'runtime': 120}	{'ZWINT_cds.fas': 0.04236316680908203, 'ZWINT.24primates.nh.emf': 0.0009508132934570312}""")

    return p

@pytest.mark.xfail(reason="Implementation in progress")
def test_from_snakemake_benchmark_file(snakemake_benchmark):
    job_data = JobData.from_snakemake_benchmark_file(snakemake_benchmark)
    assert job_data.job_name == "PRIME"
    assert job_data.slurm_id == "0"
    assert job_data.categorical == {'branch': 'test', 'properties': 'Random-5', 'gene_name': 'ZWINT'}
    assert job_data.numerical == {'hyphy': '~/local/bin/hyphy'}
    assert job_data.memory == 2048
    assert job_data.runtime == 120
