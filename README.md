# slurmise

[![PyPI - Version](https://img.shields.io/pypi/v/slurmise.svg)](https://pypi.org/project/slurmise)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/slurmise.svg)](https://pypi.org/project/slurmise)
![CI workflow](https://github.com/princetonuniversity/slurmise/actions/workflows/test.yaml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/PrincetonUniversity/slurmise/badge.svg?branch=main)](https://coveralls.io/github/PrincetonUniversity/slurmise?branch=main)
-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install slurmise
```

## Usage

### Configuration

Slurmise requires a configuration file for every command which tells slurmise
where to store important files and how to process input files.  A general CLI
parser is hard to get correct and still doesn't account for all use cases, for
example is the input file size important or number of lines?  In addition to
some built-in file parsers, you have the option of utilizing `awk` to extract
more complex information from a file.

#### Example toml
```toml
[slurmise]
# base directory to store database and optimized models
base_dir = "slurmise_dir"

# default resources to return if a model doesn't exist or isn't trained
# can be overwritten by subsequent jobs
# if not set, will use 1000 for default memory and 60 for default runtime
default_mem = 2000
default_time = 70

# for each job you want to track, give a unique job name
[slurmise.job.job_name]
# the job spec determines how to parse commands to extract their relevant,
# dependent variables
job_spec = "subcommand -T {threads:numeric} -C {complexity:category}"
# jobs of `job_name` will now return default memory of 3000 and time of 80
default_mem = 3000
default_time = 80
```

#### Matching job names
The job name of a command can be set in various ways.  First, if the command
starts with the job name, the job name will be detected and removed from the command.
```toml
# slurmise.toml
[slurmise.job.git]
job_spec = "checkout {branch:category}"
```
This job specification will match any of the following with the branch set to `my_branch`
```bash
# infer from command
slurmise record "git checkout my_branch"
# tell explicitly
slurmise record --job-name git "checkout my_branch"
```

Since the job name cannot have spaces and some commands have several subcommands,
you can set unique prefixes for certain jobs.
```toml
# slurmise.toml
[slurmise.job.git_checkout]
job_prefix = "git checkout"
job_spec = "{branch:category}"

[slurmise.job.git_merge]
job_prefix = "git merge"
job_spec = "{branch:category}"
```
With a job prefix, slurmise will use the prefix instead of job name to infer
job names.
```bash
slurmise record "git checkout my_branch"
# explicit job name
slurmise record -j git_checkout "my_branch"
```
Note that the job name and prefix should not be included in the job specification.
When a job name is explicitly given to slurmise, the corresponding command should
not have the prefix or job name included.

#### Job specifications
When constructing the job specification, tokens that should be recorded use
curly braces as placeholders:
```
{variable_name:variable_type}
```
The name should be unique within a job and contain no spaces.  The type can be
one of:
- `numeric`: A single number, used in regression as an independent variable.
Examples include the number of threads, epochs, or replicates to perform.
- `category`: A string which is used to select the correct model.  Examples include
what algorithm to choose, switches or flags.  Note that a category can be a number,
but will be stored as a string, e.g. "1.0" is different from "1".  For inference,
the categories will be matched to particular, independent model.
- `ignore`: A placeholder for a token that shouldn't be considered.  Ignored tokens
do not require a variable name.
- `file`: An input file in plain text.  Can be processed further as described below.
- `gzip_file`: An input file in gzip format.  During processing, the file will
be decompressed to read it's contents, note this can incur memory and cpu drain.
- `file_list`: An input file that contains a list of files to process in turn.

#### File Parsers
Each file can have one or more parsers associated with its variable name.
Slurmise comes with several built-in options for parsing files:
- `file_size`: The size of the file on disk, in bytes, numeric
- `file_lines`: The number of lines (newlines)  in the file, numeric
- `file_basename`: The base filename, category
- `file_md5`: The md5 digest of the file contents, category

Additionally, custom file parsers can be made using awk.  While somewhat limited,
awk prevents security issues with running arbitrary code.  File parsers require
a unique name in the `slurmise.file_parsers` collection.  The return type is
categorical by default.  The awk command can be supplied as a string or file path,
which is used with the `-f` flag of awk.  Here are some examples:
```toml
[slurmise.file_parsers.epochs]
return_type = "numerical"
awk_script = "/^epochs:/ {print $2}"

[slurmise.file_parsers.network]
return_type = "categorical"
awk_script = "/^network type:/ {print $3}"

[slurmise.file_parsers.fasta_length]
return_type = "numerical"
awk_script = "/path/to/awk/file.awk"
script_is_file = True

# contents of file.awk
# /^>/ {if (seq) print seq; seq=0} 
# /^>/ {next} 
# {seq = seq + length($0)} 
# END {if (seq) print seq}
```
The first extracts the token after `epochs: ` as a number and could be used for
getting metadata from a configuration file.  Similarly, the network parser
extracts the `network type` but this time returns the result as a category.

Finally, the `fasta_length` parser takes an awk script file that prints the
length of each sequence in a fasta file, returning the list of numbers as numerics.

To specify which parser a file uses, add them to the job entry:
```toml
[slurmise.job.sample_files]
job_spec = "--reference {reference:file} {fasta:file}"
file_parsers.reference = "file_md5,file_lines"
file_parsers.fasta = "file_size,fasta_length"
```
Each file name (`reference` and `fasta`) needs a `file_parsers` entry within the job
specification.  The name can take a comma separated list of parsers.  Here the
reference file is parsed with the md5 and number of lines.  The md5 will create a new
category based on the file contents while the lines will be an independent variable.
In practice, matching md5 will ensure the same number of lines which doesn't provide
additional information to the mode.  The fasta file returns the file size in bytes
and the number of nucleotides in each fasta entry.


## License

`slurmise` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
