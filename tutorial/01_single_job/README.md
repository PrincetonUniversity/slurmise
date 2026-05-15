# 01 — Single job

Record a single `perfectScaler` job and see what `slurmise predict` does
*before* slurmise has enough data to train a model.

## Files

- `perfectScaler` — allocates `--intensity` MB and sleeps `--duration` seconds.
- `run_perfectScaler.sbatch` — runs it once at intensity 5000, duration 10.
- `slurmise.toml` — declares `perfectScaler` with two numeric parameters.

## Run it

```bash
sbatch run_perfectScaler.sbatch
```

Total wall time: ~30 seconds.

## Inspect

After the job completes, look at what was recorded:

```bash
slurmise --toml slurmise.toml print
```

You should see one record for `perfectScaler` with intensity 5000, duration 10.

## Predict

Now ask slurmise what it would predict for a *different* intensity:

```bash
slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 4000 --duration 10"
```

The prediction will not actually use the recorded value — slurmise needs at
least 13 records per job (10 after an 80/20 train/test split) before it will
fit a model. With one record it falls back to the toml defaults.

This sets up the next tutorial: generate enough records to actually train a
model.
