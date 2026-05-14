# Extra integrations

In addition to a CLI and python API, slurmise provides additional integrations
for some workflow managers.


## Snakemake

Slurmise can control the estimation, recording and model updating during the
execution of a snakemake workflow.  To enable, you must pass in a slurmise API
instance and the workflow object.  Here is an example snakefile:
```python
# Snakefile
from slurmise.api import Slurmise
from slurmise.extras.snake_patching import patch_snakemake_workflow
import slurmise.extras.snake_parsers as sp

# get the aboslute path to the slurmise.toml.  This assumes it is
# in the same directory as the Snakefile
slurmise_toml = Path(workflow.basedir) / 'slurmise.toml'
slurmise = Slurmise(slurmise_toml)

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
patch_snakemake_workflow(
        slurmise,  # the slurmise instance from above
        workflow,  # the snakemake workflow object
        {  # a dict of rules to monitor and estimate
            'monitored': {  # must match snakemake rule name and slurmisejob name
                # mapping of slurmise variables to rule attributes
                'infile': sp.input(),
                'runtype': sp.params('execution_type'),
                'sample': sp.wildcards('sample'),
                'threads': sp.threads(),
            },
         },
        )
```

The corresponding slurmise toml would be
```toml
# slurmise.toml

[slurmise.job.monitored]
variables.infile = "file"
variables.runtype = "category"
variables.sample = "category"
variables.threads = "numeric"
default_mem = 1000
default_time = 60
file_parsers.infile = "file_size"
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

The `patch_snakemake_workflow` can also accept overwrites for the `benchmark_dir`,
which defaults to `slurmise/benchmarks` in the workdir of the workflow.  You can
also toggle `keep_benchmarks` to True to keep benchmark files after they are
recorded.  Finally, setting `record_benchmarks` to False will provide resource
estimates from slurmise without recording the actual usage or updating the job
database.

Each entry of the rules dictionary to `patch_snakemake_workflow` should contain
all the variables for the slurmise job.  You can also include a few overrides
on a per-rule basis.
 - `SLURMISE_attempt_exp` controls how quickly estimates scale with additional
 attempts; must enable with `--retries X` when running snakemake, where X is the
 maximum number of retries for each job.  Default is 1.
 - `SLURMISE_memory_scale` scale the estimate from slurmise by a constant value.
 Default is 1.1, e.g. aim for 90% efficiency.
 - `SLURMISE_runtime_scale` scale the estimate from slurmise by a constant value.
 Default is 1.25, e.g. aim for 80% efficiency.

The final estimate for a resource is `scale * estimate * attempt ** attempt_exp`

### Thread Scaling (WIP)
You can update the requested number of threads for each job dynamically based on
the amount of memory required for the job.  The idea is if a job can benefit
from multiple threads, you may want to provide more to limit how long large
memory jobs take to complete.  This keeps the memory per thread closer to
constant.  By default, the number of threads specified in the rule will be
used for all jobs.

To use thread scaling, set the `SLURMISE_thread_scaling` key of the given rule
in `patch_snakemake_workflow`.  Setting this to a numeric value will attempt to
scale the memory per thread to that value.  E.g. setting `SLURMISE_thread_scaling: 1000`
will cause a job that requires 8000 MB to request 8 threads, a job requiring
3200 MB to request 3, etc.  By default the runtime is scaled by the number of
threads and the memory is left as is.  The number of threads is kept between
1 and 20.

The behavior can be further adjusted by providing a `snake_parsers.ThreadScaler`
instead of a numeric value.
```python
class ThreadScaler:
    # used when a numeric is provided to SLURMISE_thread_scaling
    memory_per_thread: float
    # how to scale the runtime with additional threads, see below
    runtime_overhead: float = 1
    # how to scale the memory with additional threads, see below
    memory_overhead: float = 1
    # range of possible thread values to use
    thread_range: tuple[int, int] = (1, 20)
```

The overhead values have two interpretations depending on their values.  When less
than 2, the overhead is taken as a fractional scale value.  E.g. a value of
1.1 is interpreted as "provide 10% more memory/runtime per each additional thread".
Values over 2 are instead interpreted as offsets to add to the estimates.  If
each thread needs an additional 1000 MB for thread specific variables, set the
`memory_overhead` to 1000.
