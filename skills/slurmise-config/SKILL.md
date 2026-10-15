---
name: slurmise-config
description: Write or edit a slurmise.toml for a command or workflow, then validate it against real example commands. Use when the user wants to set up slurmise for a job, add a job to an existing slurmise.toml, or fix a configuration that fails to load or parse.
---

# Writing a slurmise configuration

A slurmise configuration says how to turn a command line into model inputs
(numeric and category variables) and what resource bounds apply. You cannot get
this right from a command alone: whether a value is `numeric` or `category`, and
which file features matter, is the user's modelling decision. Ask, then verify
with `slurmise validate`.

Read [reference.md](reference.md) for the schema and examples before drafting.

## Process

1. **Find the target file.** `slurmise.toml` in the working directory, else
   `~/.slurmise/slurmise.toml`. If one exists, read it and add or edit the job in
   place. Never overwrite an existing file; ask before creating one elsewhere.
2. **Collect real example commands.** Ask for 2-3 actual invocations with
   different values, not a description. Also ask:
   - Run from the CLI or from Snakemake? (Snakemake variables use `source`
     instead of parsing a command string; see reference.md.)
   - Job name, and whether commands start with the program name or need a
     `job_prefix` (e.g. `git checkout`).
3. **Ask the modelling questions you cannot infer**, offering a default for each:
   - Each varying value: `numeric` (thread count, epochs), `category` (algorithm,
     flag, mode), or not recorded (`{ignore}`).
   - Each input file: which feature predicts cost? File size (default), line
     count, a value read by a custom awk script, or the basename/md5 to select
     a model per file. For awk, ask for a sample of the file contents.
   - Resource bounds: runtime in **minutes**, memory in **MB**: a `default` for
     when no model exists, a `maximum` matching the cluster limit. Do not invent
     these numbers; omit the section if the user does not know.
4. **Draft the TOML.** Follow the rules below, keep it minimal, and only set
   resource keys the user asked for.
5. **Validate against every example command:**
   ```bash
   slurmise --toml slurmise.toml validate --json "cmd one" "cmd two"
   ```
   Pass `--job-name NAME` when commands omit the program name or prefix. Files
   referenced by example commands must exist, because file parsers really run.
   Fix every entry in `errors` and rerun until `ok` is true. Mismatch errors
   include an aligned diff of the spec against the command; use it. Read each
   `parsed` entry and confirm with the user that the extracted numerics and
   categories are what they meant.
6. **Report** warnings to the user and explain them. Do not run `record`,
   `update-model` or `predict`; they touch the job database, which is out of scope.

## Rules for the TOML

- `[slurmise] base_dir` is required.
- Every job needs a `variables` table, and at least one numeric value (a
  `numeric` variable, or a file with a numeric parser such as `file_size`).
- Every declared variable must appear as a `{placeholder}` in `job_spec`, and
  every placeholder except `{ignore}` must be declared. Names are unique and
  contain no spaces.
- Do not put the job name or `job_prefix` in `job_spec`; slurmise strips it.
- Each placeholder matches one whitespace-delimited token by default. For
  anything else (quoted paths, trailing arguments) add `pattern = "<python regex>"`
  to a non-numeric variable. `numeric` variables cannot take a `pattern`.
- Literal braces in a spec are doubled: `{{` and `}}`.
- Every `file`, `gzip_file` and `file_list` variable needs `file_parsers`
  (a name or list of names).
- Custom awk parsers live in `[slurmise.file_parsers.<name>]` with `awk_script`
  and `type` of exactly `numeric` or `category`. Awk is run without a shell, so
  do not wrap the script in extra quotes. Set `script_is_file = true` (lowercase
  TOML boolean) when `awk_script` is a path. A numeric awk parser yields a list
  of numbers, and every job must produce the same count or training fails.
- Job names are inferred by `command.startswith(prefix)` in file order, with no
  word boundary. Do not give a job a prefix that starts with an earlier job's
  prefix, and avoid short names that other commands begin with.
- `[slurmise.job.<name>.model]` with `model = "poly"` (default) or `"knn"`.
  Degree and neighbor count are not configurable.

## Do not

- Guess `numeric` vs `category` or resource limits silently.
- Edit anything outside the configuration file.
- Report success without a passing `validate` run on the user's real commands.
