# slurmise.toml reference

Condensed from the project README (the source of truth) and `tests/`. When this
disagrees with the README or with `slurmise validate`, trust those.

## Layout

```toml
[slurmise]
base_dir = "slurmise_dir"          # required; database and models live here
db_filename = "slurmise.h5"        # optional
retrain_warning_threshold = 0.2    # optional; fraction of new records before predict warns
retrain_warning_enable = true      # optional

[slurmise.runtime]                 # global bounds, minutes (built-in default 60)
default = 70                       # returned when no model exists
minimum = 10
maximum = 1440
multiply_prediction_by = 1.0       # must be > 0
retry_exponent = 1.0               # Snakemake attempt N: prediction *= N**retry_exponent
on_high_uncertainty_return = "prediction"   # "default" | "prediction" | "max" | "min"

[slurmise.memory]                  # same keys, MB (built-in default 1000)
default = 2000

[slurmise.job.<name>]              # name: no spaces
job_prefix = "git checkout"        # optional; defaults to the job name
job_spec = "monomer -T {threads} -C {complexity}"

[slurmise.job.<name>.runtime]      # per-job overrides of any key above
[slurmise.job.<name>.memory]
[slurmise.job.<name>.model]
model = "poly"                     # or "knn"

[slurmise.job.<name>.variables]
threads = {type = "numeric"}
complexity = {type = "category"}
```

Constraints enforced at load: `minimum >= 0`, `maximum >= minimum`,
`multiply_prediction_by > 0`, and `on_high_uncertainty_return = "max"` needs a
finite `maximum`.

## Variable types

| type | meaning |
| --- | --- |
| `numeric` | A number (ints, decimals, `1e-4`). Regression input. No `pattern`. |
| `category` | A string; selects a separate model per value. `1` and `1.0` differ. |
| `file` / `gzip_file` | A file, processed by `file_parsers`. |
| `file_list` | A file listing files, each processed in turn. |
| `ignore` | Matched but not recorded. Usually via `{ignore}` in the spec. |

Built-in file parsers: `file_size` (bytes, numeric), `file_lines` (numeric),
`file_basename` (category), `file_md5` (category). A parser's output is stored as
`<variable>_<parser>`, e.g. `reads_file_size`.

## Examples

Basic numeric and category:
```toml
[slurmise.job.nupack]
job_spec = "monomer -T {threads} -C {complexity}"
[slurmise.job.nupack.variables]
threads = {type = "numeric"}
complexity = {type = "category"}
```

Discarding arguments, and absorbing a variable-length tail:
```toml
[slurmise.job.cmd]
job_spec = "{input} {ignore} {ignore} {output} {rest}"
[slurmise.job.cmd.variables]
input = {type = "file", file_parsers = "file_size"}
output = {type = "category"}
rest = {type = "ignore", pattern = ".+"}
```

Quoted path with spaces:
```toml
[slurmise.job.quoted]
job_spec = '"{input}" {threads}'
[slurmise.job.quoted.variables]
input = {type = "file", file_parsers = "file_size", pattern = '[^"]+'}
threads = {type = "numeric"}
```

Literal braces:
```toml
job_spec = "awk '{{print $1}}' {input}"
```

Two prefixed subcommands:
```toml
[slurmise.job.git_checkout]
job_prefix = "git checkout"
job_spec = "{branch} -j {jobs}"
[slurmise.job.git_checkout.variables]
branch = {type = "category"}
jobs = {type = "numeric"}
```

Custom awk parsers:
```toml
[slurmise.file_parsers.epochs]
type = "numeric"
awk_script = "/^epochs:/ {print $2}"

[slurmise.file_parsers.fasta_length]
type = "numeric"
awk_script = "/path/to/file.awk"
script_is_file = true
```

Snakemake (variables read from the rule; `source` is required on every
variable and `job_spec` is not used):
```toml
[slurmise.job.monitored]
[slurmise.job.monitored.variables]
infile = {type = "file", source = "input", file_parsers = "file_size"}
runtype = {type = "category", source = "params", key = "execution_type"}
sample = {type = "category", source = "wildcards", key = "sample"}
threads = {type = "numeric", source = "threads"}
```
Snakemake options live in `[slurmise.extras.snakemake]`; see
`src/slurmise/extras/README.md`.

## Validation

```bash
slurmise --toml slurmise.toml validate --json "<command>" ["<command>" ...]
```
Output keys: `ok`, `toml_path`, `errors`, `warnings`, `parsed`. Exit code 1 when
`ok` is false. Warnings cover shadowed job prefixes, `file_parsers` sections with
no `awk_script`, parsers replacing a built-in, and missing `awk`.
