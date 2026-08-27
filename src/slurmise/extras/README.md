# Extra integrations

In addition to a CLI and python API, slurmise provides additional integrations
for some workflow managers.


## Snakemake

Slurmise can control the estimation, recording and model updating during the
execution of a snakemake workflow.  To enable, import the snake_patching module
(which registers the patch) and call `slurmise.patch` with the workflow object.
Here is an example snakefile:
```python
# Snakefile
from pathlib import Path
from slurmise.api import Slurmise
import slurmise.extras.snake_patching  # registers the patch

# get the absolute path to the slurmise.toml.  This assumes it is
# in the same directory as the Snakefile
slurmise = Slurmise(Path(workflow.basedir) / 'slurmise.toml')

rule all:
    ...

rule monitored:
    input:
        input_{sample}.txt
    output:
        output_{sample}.txt
    params:
        execution_type="fast"
    threads: 3
    shell:
        "my_command --runtype {params.execution_type} {input} {output}"

# this sets up slurmise to integrate with the workflow
slurmise.patch(workflow=workflow)
```

The corresponding slurmise toml would be
```toml
# slurmise.toml

[slurmise.job.monitored]
default_mem = 1000
default_time = 60
[slurmise.job.monitored.variables]
# can specify a key, otherwise first file
infile = {type = "file", source = "input", file_parsers = "file_size" }
runtype = {type = "category", source = "params", key = "execution_type"}
sample = {type = "category", source = "wildcards", key = "sample"}
threads = {type = "numeric", source = "threads"}
```
The patching function updates the following aspects of the workflow:
 - **onstart**: The onstart function from the workflow will run and then slurmise
 will update all models from it database.
 - **onsuccess**: The onsuccess function from the workflow will run and then slurmise
 will read all benchmark files which were generated from the current run.
 By default, the benchmark files will be deleted after they are recorded.
 - Extended benchmark recording will be enabled.
 - Benchmark files will be set for each rule to be updated.
 - A `slurmise_data` parameter will be added to the rule containing all the
 information parsed by slurmise.
 - Resources for `runtime` and `mem_mb` will be populated by slurmise using the
 `runtime` and `memory` results respectively.

The patching behavior can be further customized in the `slurmise.extras.snakemake`
section of the toml file.
```toml
[slurmise.extras.snakemake]
# location of benchmark files, default is slurmise/benchmarks
benchmark_dir = "nondefault/benchmarks"
# record completed jobs, default true
record_benchmarks = true
# keep benchmark files of jobs after recording their outputs, default false
keep_benchmarks = true
```

By default, `slurmise.patch` monitors all rules defined in the slurmise toml.
To monitor only a subset, pass a list of rule names:
```python
slurmise.patch(workflow=workflow, rules=["monitored"])
```
