## 01 — what `./slrmise run` is

The goal of this tutorial is to show a prototype of how slurmise might work
using an "all-in-one" approach where `slurmise` actually submits `sbatch`
jobs for you.

Type the `$` commands below yourself, making sure you're in this directory — or run
`./tutorial.py` to be walked through these exact commands interactively.

Prereqs: run on a login node from this directory, with `slurmise` importable by
your `python3`, and export any cluster submit vars first (`SBATCH_ACCOUNT`, ...).

Send SLURM's job output somewhere tidy while you're at it. `sbatch` writes
`slurm-<jobid>.out` into whatever directory you submitted from unless you tell it
otherwise, so the tutorial directory fills up with stray files:

    mkdir -p out_slurm_logs
    export SBATCH_OUTPUT=out_slurm_logs/slurm-%j.out

`./tutorial.py` does exactly that for you before it runs anything, and it only
sets `SBATCH_OUTPUT` if you haven't — so your own setting always wins.

The `#>` lines inside the code blocks say what each command is expected to do.
They're shell comments, so pasting a whole block into your own shell is harmless;
`./tutorial.py` reads them and checks the command output, waiting on
the scheduler where a `retry=` or `repeat-until` says to.

If the queue is busy — or you have no scheduler at all — set `SLRMISE_MOCK=1`
(or run `./tutorial.py --mock`) and `slrmise` will pretend: it skips `sbatch` and
works out how each job would have gone instead. The whole tour then takes under
a minute. Prediction, self-heal and model fitting all still run for real; only
the scheduler is imaginary.

## 02 — what's here, and `predict`

Important files for this tutorial:

- `./slrmise` — the prototype. It's a stand-in for a hypothetical
  `slurmise run`, so the name is deliberately misspelled (no `u`). It
  is not installed on the PATH so we'll always run as `./slrmise`.
  It predicts a job's resources, submits it, and records what it used; its
  subcommands are `run`, `predict`, and `display`:

```bash
$ ./slrmise --help
#> expect /Commands:/
```

- `../bin/perfectScaler` — the example command whose time and memory we're
  trying to predict. It's an uninteresting 25-line python script that just
  allocates `--intensity` MB of memory and holds it for `--duration` seconds,
  then exits.
  In real use you'd point slurmise at your command of interest
  instead. Take a look, then run it directly on the login node

```bash
$ cat ../bin/perfectScaler
#> expect /intensity/
$ ../bin/perfectScaler
```

  By default, it holds ~200 MB for ~2 s and prints nothing. Try it again with
  `../bin/perfectScaler --duration 10` if you're patient.

- `slurmise.toml` — the config. `job_spec` tells slurmise how to parse the
  command into features; `default_mem` / `default_time` are the initial
  guesses used until a model has been trained:

```bash
$ cat slurmise.toml
#> expect /job_spec/
```

  The `job_spec` has to agree with the command: it names `--intensity` and
  `--duration`, so slurmise expects to find exactly those on the wrapped command
  line.

Now use `predict` — it estimates resources and submits nothing. The new syntax
wraps the command of interest after a `--` separator, and `--toml` points at the config
above.

First, watch what happens when the command doesn't supply the features the
`job_spec` declares:

```bash
$ ./slrmise --toml slurmise.toml predict -- \
    ../bin/perfectScaler
#> expect fail
```

That error is the toml and the command disagreeing — the `job_spec` in the toml wants
`--intensity` and `--duration`, and we gave neither. Now let's supply them:

```bash
$ ./slrmise --toml slurmise.toml predict -- \
    ../bin/perfectScaler --intensity 2000 --duration 5
#> expect /memory:/
```

That prints a prediction, but only after a warning:

    WARNING: Not enough fitting data points in the fits. Returning default values.

The database is empty, so there is nothing to fit a model to. This is not a
caveat about accuracy — no model is consulted at all, and what comes back is
`default_mem` and `default_time` straight from the toml. slurmise wants 10 data
points to train on, and holds back 20% of the runs to test against, so it takes
about 13 COMPLETED runs before predictions stop being the defaults.

Finally, ask again with wildly different features:

```bash
$ ./slrmise --toml slurmise.toml predict -- \
    ../bin/perfectScaler --intensity 99 --duration 1
#> expect /memory:/
```

Same prediction. Until a model exists (lesson 07), the inputs don't move the
estimate at all — it's just the cold defaults from the toml.

## 03 — one run: stub now, metrics later

`run` predicts, submits via `sbatch`, and immediately records a stub — the
row exists but its actual memory/time are unknown (`-`) until the job finishes
and a later sync pulls them from `sacct`.

```bash
$ ./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler --intensity 2000 --duration 5
#> expect /Submitted job/
```

Right away, `display --no-sync` shows the stub with `-` for the metrics:

```bash
$ ./slrmise --toml slurmise.toml display --no-sync
#> expect /2000/
```

Once the job finishes, `display` (which syncs first) fills it in. There's no
`squeue` polling here — `display` is the check, so you just re-run it until the
row fills in. That's what the `retry=` below tells `./tutorial.py` to do, and it
watches this job's own row rather than anything else you happen to have queued:

```bash
$ ./slrmise --toml slurmise.toml display
#> expect /COMPLETED.*\b2000\b/ retry=180 delay=10
```

Compare `alloc_mem` (what the job was actually given — 4500M, the 3000M default
times the 1.5 margin) against `mem` (what it really used, ~2015M). That ~2500M
gap is real waste — lesson 06 reclaims it.

## 04 — a failure is data, not a loss

The toml's 3000M default becomes a 4500M allocation, so we ask for far more than
that — intensity 8000 needs about 8015M. It has to be far more: a job that only
slightly overshoots gets the excess reclaimed and survives, so a small overshoot
would quietly COMPLETE instead of failing. The job can't fit and is
recorded `OUT_OF_MEMORY` — it didn't vanish, the failure is evidence, and it's excluded from model training.

```bash
$ ./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler --intensity 8000 --duration 5
#> expect /Submitted job/
$ ./slrmise --toml slurmise.toml display
#> expect /OUT_OF_MEMORY.*\b8000\b/ retry=180 delay=10
```

## 05 — self-heal: double the memory until it fits

Re-run the same intensity-8000 command. With a failure on record and no
success, `slrmise` escalates past the largest failing allocation — doubling the
memory (4500 → 9000) until the job COMPLETES. No model is involved; this
is pure exact-history logic.

`TIMEOUT` self-heals the same way, on the other resource: a job that outruns its
time limit gets the time doubled (2 → 4 min) on the next run, from the same
exact-param history. Only the resource differs.

Run this block again each time it's still `OUT_OF_MEMORY` — the `repeat-until`
below is `./tutorial.py` doing exactly that for you until it completes:

```bash
$ ./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler --intensity 8000 --duration 5
$ ./slrmise --toml slurmise.toml display
#> expect /COMPLETED.*\b8000\b/ retry=180 delay=10
#> repeat-until /COMPLETED.*\b8000\b/ max=3
```

## 06 — right-size reuse: reclaim the headroom

Re-run lesson 03's intensity-2000 command — the one allocated ~4500M but that
only used ~2100M. Because a success is on record, reuse now right-sizes to what
it actually used × the margin (~3150M), ratcheting the allocation down.
(Contrast lesson 05, which ratcheted up to fit.)

```bash
$ ./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler --intensity 2000 --duration 5
#> expect /Submitted job/
$ ./slrmise --toml slurmise.toml display
#> expect /(COMPLETED.*\b2000\b[\s\S]*){2}/ retry=180 delay=10
```

The newest intensity-2000 row's `alloc_mem` is ~3023M, down from ~4500M — the
allocation ratcheted down to what the job really used, times the margin. (That
comparison is for your eyes; the check above only confirms the re-run
completed.)

## 07 — it learns: predictions become model-based

slurmise only predicts from a model once the fit has at least 10 training
points. It holds back 20% of the runs to test against, so that means about 13
COMPLETED runs, not 10. Submit a spread of varied jobs to cross that threshold:

```bash
$ for i in 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2500 2600; do \
      ./slrmise --toml slurmise.toml run -- \
          ../bin/perfectScaler --intensity $i --duration 6; \
  done
#> expect /Submitted job/
```

Now a dry-run syncs, fits the model, and predicts on a brand-new intensity —
the estimate tracks intensity instead of returning the flat cold default. Re-run
it until the `Not enough fitting data points` warning stops appearing; each
attempt syncs, so this doubles as the wait for that batch to land:

```bash
$ ./slrmise --toml slurmise.toml run --dry-run -- \
    ../bin/perfectScaler --intensity 1500 --duration 6
#> expect /^(?![\s\S]*Not enough fitting data points)[\s\S]*Fit perfectScaler on \d+ completed/ retry=180 delay=10
```

Look for a `--mem` near 2270M instead of the flat 4500M cold default. That is
the model: intensity 1500 really needs about 1515M, so 1515 × the 1.5 margin is
where it lands — tighter than the cold default, and derived from intensity
rather than ignoring it.

Keep the query inside the range you trained on. Ask for a `--duration` the model
has never seen (the runs above were all 5s or 6s) and it extrapolates badly —
`--duration 9` predicts several times the memory actually needed. Runtime also
still falls back to the default here; memory is the resource this example
teaches.

## Starting over

The lessons only hold from a clean slate — lesson 04 is only `OUT_OF_MEMORY` if
there's no earlier success at that intensity for self-heal to have learned from.
So throw away the database and the fitted models before starting. `./tutorial.py`
runs exactly this block for you, every time, before lesson 02:

```bash
#> reset
$ rm -f slurmise.h5 fits.json poly_runtime_model.pkl poly_memory_model.pkl
$ rm -f out_slurm_logs/*.out
```
