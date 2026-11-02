---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Bash
  language: bash
  name: bash
---

```{code-cell}
:tags: [remove-cell]
# Work in a scratch directory so the files this guide writes stay out of docs/
cd "$(mktemp -d)"
```

# Writing a configuration from scratch

slurmise needs a `slurmise.toml` that tells it how to read each command you want
to predict resources for. This guide builds one up a piece at a time and checks
each step with `slurmise validate`, which reads the configuration and your example
commands but never touches the job database. You don't need SLURM to follow along.

If you would rather have an AI assistant draft the file for you, see
{doc}`config-skill`. It is worth reading this page first so you can review what the
assistant produces.

```{note}
slurmise looks for `slurmise.toml` in the current directory, then in
`~/.slurmise/slurmise.toml`. Pass `--toml path/to/file.toml` to use a different one,
as the commands below do.
```

## Start from a real command

Suppose you run a program called `nupack` like this, varying the thread count and a
complexity setting between runs:

```text
nupack monomer -T 4 -C high
nupack monomer -T 16 -C low
```

Collect two or three real invocations before you write anything. The configuration
is a template that has to match all of them.

## The smallest configuration

Every configuration needs a `[slurmise]` table with a `base_dir`, which is where
slurmise keeps its database and fitted models. Each job is then a table under
`[slurmise.job.<name>]`.

```{code-cell}
cat > slurmise.toml <<'EOF'
[slurmise]
base_dir = "."

[slurmise.job.nupack]
job_spec = "monomer -T {threads} -C {complexity}"

[slurmise.job.nupack.variables]
threads = {type = "numeric"}
complexity = {type = "category"}
EOF
```

Three things are happening here:

- The job is named `nupack`. slurmise recognises a command by checking that it
  starts with the job name, and strips the name before matching the rest.
- `job_spec` is the rest of the command with each changing value replaced by a
  `{placeholder}`.
- Every placeholder is declared in `variables` with a type. The next section says
  which type to pick.

Check it against the example commands:

```{code-cell}
slurmise --toml slurmise.toml validate "nupack monomer -T 4 -C high" "nupack monomer -T 16 -C low"
```

Each line of the `Parsed` output shows what slurmise extracted from a command.
Read it carefully, because this is the step that confirms the configuration means
what you intended. Values read from a file are
named after the variable and the parser, for example `reads_file_size` or
`config_epochs`.

## Numeric or category?

Only you know how a value affects resource use, so this is a modelling decision:

- **`numeric`** values go into a regression as independent variables. Use them for
  quantities that scale cost smoothly, such as thread counts, epochs, or replicates.
- **`category`** values choose which model is used. Use them for modes, flags and
  algorithm names. slurmise fits a separate model for each distinct value, and
  stores the value as text, so `1` and `1.0` are different categories.
- **`ignore`** values are matched but not recorded. Put `{ignore}` in a `job_spec`
  to skip one token without declaring a variable.

Every job needs at least one numeric value, either a `numeric` variable or a file
parser that produces a number (see below).

## When validation fails

`validate` exits with a non-zero status and an aligned diff when a command does not
match the `job_spec`. Here the `-C` option is missing from the command:

```{code-cell}
slurmise --toml slurmise.toml validate "nupack monomer -T 4" || echo "exit status $?"
```

The diff shows where the spec and the command stopped lining up. Other common
mistakes are a placeholder that is not declared in `variables`, a variable that
never appears in `job_spec`, and putting the job name inside `job_spec`.

## Setting resource limits

Without any resource settings slurmise falls back to built-in defaults of 60 minutes
and 1000 MB. Add `[slurmise.runtime]` and `[slurmise.memory]` to set your own.
Runtime is in **minutes** and memory is in **MB**.

```{code-cell}
cat > slurmise.toml <<'EOF'
[slurmise]
base_dir = "."

# Global bounds for every job
[slurmise.runtime]
default = 70       # returned when no model has been fit yet
maximum = 1440     # never predict more than the 24 hour wall-time limit

[slurmise.memory]
default = 2000
maximum = 500000

[slurmise.job.nupack]
job_spec = "monomer -T {threads} -C {complexity}"

# Per-job override of a global value
[slurmise.job.nupack.memory]
default = 4000

[slurmise.job.nupack.variables]
threads = {type = "numeric"}
complexity = {type = "category"}
EOF
slurmise --toml slurmise.toml validate "nupack monomer -T 4 -C high"
```

Set `maximum` to what your cluster allows so predictions never ask for more than a
job can be given. If you don't know a sensible value, leave the key out rather than
guessing. The full list of keys, including `minimum`, `multiply_prediction_by` and
`retry_exponent`, is in the
[README](https://github.com/PrincetonUniversity/slurmise#configuration).

## Using a file as an input

Often the cost of a job depends on an input file, not a number on the command line.
Declare the variable as a `file`, and say which feature of the file to use with
`file_parsers`. The built-in parsers are:

| Parser | Produces | Type |
| --- | --- | --- |
| `file_size` | size on disk, in bytes | numeric |
| `file_lines` | number of lines | numeric |
| `file_basename` | the file name | category |
| `file_md5` | digest of the contents | category |

You can list several parsers for one file. The files must exist when you validate,
because the parsers really run. Note that `file_lines` counts the last line even when
the file ends in a newline, so a file ending in `\n` reports one more than `wc -l`.

```{code-cell}
printf 'ACGT\nACGT\nACGT' > reads.txt
cat > slurmise.toml <<'EOF'
[slurmise]
base_dir = "."

[slurmise.job.align]
job_spec = "--threads {threads} {reads}"

[slurmise.job.align.variables]
threads = {type = "numeric"}
reads = {type = "file", file_parsers = ["file_size", "file_lines"]}
EOF
slurmise --toml slurmise.toml validate "align --threads 8 reads.txt"
```

### A custom feature with awk

When the feature you care about is inside the file, write an `awk` parser. It goes in
`[slurmise.file_parsers.<name>]`, with a `type` of `numeric` or `category`. slurmise
runs awk directly rather than through a shell, so the script needs no extra quoting.
Set `script_is_file = true` if `awk_script` is a path to a script instead.

```{code-cell}
printf 'epochs: 30\nnetwork type: dense\n' > train.cfg
cat > slurmise.toml <<'EOF'
[slurmise]
base_dir = "."

[slurmise.file_parsers.epochs]
type = "numeric"
awk_script = "/^epochs:/ {print $2}"

[slurmise.job.train]
job_spec = "{config}"

[slurmise.job.train.variables]
config = {type = "file", file_parsers = "epochs"}
EOF
slurmise --toml slurmise.toml validate "train train.cfg"
```

A numeric awk parser can print several numbers. Every job must then produce the same
count, or training fails later.

## Commands with several words, or unusual arguments

**Subcommands.** A job name cannot contain spaces, so use `job_prefix` when the
program has subcommands. slurmise then infers the job from the prefix and strips the
prefix before matching `job_spec`.

```{code-cell}
cat > slurmise.toml <<'EOF'
[slurmise]
base_dir = "."

[slurmise.job.git_checkout]
job_prefix = "git checkout"
job_spec = "{branch} -j {jobs}"
[slurmise.job.git_checkout.variables]
branch = {type = "category"}
jobs = {type = "numeric"}

[slurmise.job.git_merge]
job_prefix = "git merge"
job_spec = "{branch} -j {jobs}"
[slurmise.job.git_merge.variables]
branch = {type = "category"}
jobs = {type = "numeric"}
EOF
slurmise --toml slurmise.toml validate "git checkout my_branch -j 4"
```

Job names are inferred with a plain `startswith`, in the order jobs appear in the
file. A prefix that starts with an earlier job's prefix is never reached, and
`validate` warns about it.

**Naming the job explicitly.** If the command has no recognisable prefix, pass the job
name yourself and leave the prefix off the command:

```{code-cell}
slurmise --toml slurmise.toml validate --job-name git_merge "my_branch -j 4"
```

**Values that are not one word.** Each placeholder matches one whitespace-delimited
token. For anything else, such as a quoted path or a tail of extra arguments, give a
non-numeric variable a `pattern`, which is a Python regular expression. Numeric
variables cannot take a pattern. See the README for worked examples, and double any
literal braces (`{{` and `}}`) in a `job_spec`.

## Choosing a model

Each job is fit with a degree 2 polynomial regression by default (`poly`), which
suits resource use that scales smoothly with the numeric variables. For usage that
does not follow a simple curve, switch the job to k-nearest neighbors:

```toml
[slurmise.job.nupack.model]
model = "knn"
```

`knn` can only predict from jobs it has already seen. Models are stored per type,
so after switching, run `slurmise update-model --job-name nupack` to fit the new one.

## Where to go next

- {doc}`tutorial` records real jobs, fits a model and predicts with the configuration
  you have just written.
- {doc}`config-skill` shows how to have an AI assistant draft and validate a
  configuration for you.
- The [README](https://github.com/PrincetonUniversity/slurmise#configuration) lists
  every option, and the
  [Snakemake extras](https://github.com/PrincetonUniversity/slurmise/tree/main/src/slurmise/extras)
  describe how variables are read from a rule instead of a command string.
