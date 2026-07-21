# Slurmise Tutorial

This tutorial walks through using slurmise to track SLURM job resource usage
and predict requirements for future jobs. Each subfolder is a self-contained
example that can be run independently.

You likely already have `slurmise` installed and ran `slurmise-generate-tutorial`
to access this file.

## Prerequisites

- Access to a SLURM-enabled cluster (with `sbatch` and `srun`)
- slurmise installed (`pip install slurmise`)

## Examples

- [01_single_job](01_single_job/) — Record a single job and see what
  `slurmise predict` does before any model has been trained.
- [02_jobs_in_loop](02_jobs_in_loop/) — Generate enough recordings in a loop
  to actually train a model, then use `slurmise predict`.
- [03_array_jobs](03_array_jobs/) — Run jobs in parallel using a SLURM array,
  including a categorical job to show how slurmise handles non-numeric inputs.
- [lazy_recording](lazy_recording/) — **Not a slurmise feature**: a
  self-contained prototype of a proposed `lazy-record` command that wraps the
  job instead of requiring the command to be specified twice. Here temporarily
  so it can ride the tutorial CI while the idea is discussed.

## The test jobs

The tutorials share three small Python scripts that allocate memory and sleep,
kept in the top-level [`bin/`](bin/) directory. Each example's `.sbatch`
invokes them by relative path (e.g. `../bin/perfectScaler`).

- `perfectScaler` — deterministic: memory and runtime exactly match the
  `--intensity` and `--duration` arguments.
- `complexMemScaler` — same interface as `perfectScaler` but adds ±20% random
  noise to memory. Useful for showing that slurmise can model noisy data.
- `categoricalScaler` — takes `--intensity` and `--duration` as category
  labels (`1`, `2`, or `3`) instead of raw numbers, mapping them internally.

All scripts use a short `--duration` (a few seconds) so each tutorial finishes
quickly. This means runtime predictions aren't very interesting; **memory is
the variable worth predicting**.

## Usage

`cd` into the `01_single_job/` directory and follow along with the README there.
THen continue on to the other numbered examples.

Your SLURM setup may require additional flags such as `--account` when you're
submitting the included sbatch files.
