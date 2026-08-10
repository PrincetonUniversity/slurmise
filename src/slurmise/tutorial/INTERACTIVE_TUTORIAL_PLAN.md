# The interactive tutorial: what was built, and what's still open

Status: **done.** Every lesson is a `tutorial.md` driven by the single
`tutorial.py` in this directory. This file is internal notes — why the shape is
what it is, which assumptions turned out to be wrong (several), and what is
still open. It is excluded from the shipped tutorial via `EXCLUDE_PATTERNS` in
`generate_tutorial.py`.

## Why

Each example used to be a `README.md` telling you what to type plus a `.sbatch`
that actually did it, and `tests/integration/run_tutorial_slurm.sh` submitted
those `.sbatch` files so CI noticed when they broke. The content was therefore
specified twice — once as prose for a reader, once as a script for CI — and the
two drifted. `03_array_jobs/README.md` ended up describing files that did not
exist (`run_array.sbatch`) and task counts that were wrong (8 vs
`--array=0-25`). Nothing failed. (It was
right about per-category sub-models, though -- see below, where I got that
backwards first.)

One `tutorial.md` per lesson is now the only source of truth: each command
carries a `#>` expectation, and a reader can either read the markdown and type
along or let `tutorial.py` drive it. The tutorial is its own integration test.

## Shape

One runner, `tutorial.py`, discovering `*/tutorial.md`. Each lesson directory
keeps its own `slurmise.toml`, its own `slurmise.h5`, and its own `#> reset`
block, which runs immediately before that lesson.

That per-lesson database is the decision everything else follows from. A single
shared database would be more honest about how slurmise accumulates history, but
it makes the lessons ordered — after 02 has trained a model, 01's `predict` no
longer returns a cold default — and it would force the three tomls together,
whose defaults cannot be reconciled without changing what lesson 01 shows.
Keeping them separate means any lesson runs alone, in any order, and resume
(`--from` / `--stop-after`) is not needed even though a full pass submits real
jobs and is not cheap.

`slurmise_run/` was absorbed as `04_slurmise_run`. It is the one lesson that
mocks the scheduler (`SLRMISE_USED_*` declared usage, `--slurm` to really
submit), and the only one `generate_tutorial.py` excludes — so it exists from a
clone and not from a generated tutorial, and the menu simply shows three lessons
instead of four.

`record` stays inside the job in 01–03. It is *meant* to be called from there —
that is the lesson — so the interesting output lands in
`out_slurm_logs/` and the markdown reads it back with `cat … #> expect /re/`
rather than restructuring the lesson to record from the login node.

## Walking a lesson without a cluster

Each `run_x.sbatch` has a `mock_x.sh` beside it that states what the job would
have used rather than running it, via `slurmise raw-record --memory --runtime`
(both given, and sacct is not consulted -- `api.raw_record` treats usage already
on the record as authoritative). `tutorial.py --mock` substitutes one for the
other: `sbatch [--wait] run_x.sbatch` becomes `bash mock_x.sh`. The full tour
drops from most of an hour to about two minutes, which is what makes it usable
as a check on every push (`test.yaml`) rather than only on the cluster workflow.

Two shapes were tried before this one. Folding the mock into each `.sbatch`
behind an `if [ -n "$SLURM_JOB_ID" ]` works -- and `sbatch` setting that
variable is a sound way to tell the modes apart -- but it makes the `.sbatch`
files hard to read, and those are the artifact a newcomer copies and adapts. A
dedicated `slurmise mock` CLI command was also tried and dropped: `raw-record`
gaining two optional flags is a smaller API than a second command that records.

The cost of the split: the mock scripts restate each job's features as
`--numerics` JSON, and the memory model of `complexMemScaler` and
`categoricalScaler` in shell. Both can drift from `bin/`. The mock scripts say
so, at the line that would need changing.

What `--mock` cannot cover, and why the cluster workflow still matters: `srun`,
`sacct`, and `record` reading back what a job really used -- i.e. everything
`raw-record --memory --runtime` deliberately steps around.

That real path is checked on GHA (`tutorial-slurm.yaml`) for 01 and 02 only.
03's three arrays are 112 tasks of ~10s each, which serialize on the workflow's
single-node cluster and would exceed its 30 minute budget by themselves. So
**03's srun/sacct path has never run against a real scheduler** -- its logic is
covered by `--mock`, its submission is not. Raising `timeout-minutes` and adding
it to `LESSONS` is the fix if that gap matters.

## Assumptions that turned out to be wrong

Both were caught only by running on a real cluster. Worth remembering the next
time this file says something confident.

- **`SBATCH_OUTPUT` vs `#SBATCH --output=`.** The plan said the script's
  directive wins and the runner's env var was harmless. It is the other way
  round — slurm's precedence is command line > environment > script directive —
  so the runner was silently renaming logs the `.sbatch` had asked for, and 01's
  `cat` of them failed. `Tour.__init__` now sets `SBATCH_OUTPUT` only for a
  lesson with no `.sbatch` of its own (i.e. only 04).
- **"`sbatch --wait` means no `retry=` is needed here."** True for 01 and 03.
  Not for 02, which submits two ~9-minute loops and wants them running at the
  same time — so neither gets `--wait`, and the wait is a `squeue … | wc -l`
  with `#> expect /^0$/ retry=120 delay=15`. The `retry=` machinery this plan
  called "simply not required" is what makes that lesson bearable.

## Found while converting: one model per `base_dir`, not per job

`api.py` passes `path=self.configuration.slurmise_base_dir` at both call sites
(`raw_predict`, `_update_model`), so every job's fit is written to the same
`fits.json` + `poly_*.pkl`. `update-all` therefore fits each job in turn into
one file and only the last survives, and `predict` answers *every* job name from
whatever model is sitting there. Two different jobs return byte-identical
predictions — which `02_jobs_in_loop/README.md` had already noticed without
diagnosing ("NOTE!! Not sure why the exact same memory prediction is returned").

`ResourceFit._make_model_path()` exists to prevent exactly this: it hashes
`job_name` **and `query.categories`** into a per-model directory under
`~/.slurmise/models/`. Because `path` is always passed explicitly, `load()`
takes the `case (str(path), _)` branch and that function is unreachable outside
its own unit tests (`tests/fit/test_polynomial_fit.py`).

The lessons work around it rather than depending on it: 02 and 03 `rm -f
fits.json *.pkl` and `update-model` the single job they are about to ask about,
and say plainly why. That is honest and teaches a real constraint, but it means
`update-all` is not demonstrated anywhere — it cannot be, usefully, until this
is fixed.

If it is fixed, both lessons get simpler: one `update-all`, then predict each
job, no `rm`.

## Categories partition; the one-hot encoder is a red herring

Settled by experiment, after an initial reading of the code got it backwards.

`slurmise print` groups a categorical job's records by category value
(`scaling=linear`, `scaling=quadratic`, `scaling=cubic`), and `update-model`
fits only the pile matching the category in the command it is given. Fitting and
predicting each in turn returns 20.75 / 385.75 / 7960 MB against true values of
20 / 400 / 8000 -- so each category really does carry its own model and its own
~13-run threshold, which is what `03_array_jobs/README.md` always claimed.

The `OneHotEncoder` in `fit/resource_fit.py:150` is what misled the first
reading. It is real, but every fit sees rows from exactly one category, so it
never has more than one value to encode. It would matter only if a fit were
handed a mixed pile, which the database layer does not do.

`_make_model_path()` hashing `job_name` **and** `query.categories` is consistent
with all of this -- it is the storage layout the design implies. It is also
unreachable (see above), which is why every category's model still lands in the
same file and the lessons have to `rm` between them.

What is left open is only whether 60 tasks at a single `(intensity, duration)`,
varying nothing but `--scaling`, is the best demonstration. It shows the
per-category split clearly and shows nothing about how a category interacts with
the numeric features.

## Prior art: is the markdown runner reinventing a wheel?

Reviewed before committing to keep it. Partly yes, but not in the part that
matters:

| tool | asserts on output | interactive walkthrough | waits for a scheduler |
|---|---|---|---|
| [byexample](https://byexamples.github.io/byexample/) | yes — `$ cmd` + expected output, `<...>` placeholders, `+timeout` | no | no |
| [Runme](https://github.com/runmedev/runme) | no | yes — TUI over markdown blocks | no |
| [mdsh](https://github.com/zimbatm/mdsh), [mdtest](https://github.com/crabtw/mdtest), [md-cli-test](https://lib.rs/crates/md-cli-test) | yes | no | no |
| [pytest-markdown-docs](https://github.com/modal-labs/pytest-markdown-docs), [phmdoctest](https://github.com/tmarktaylor/phmdoctest) | yes, Python blocks | no | no |

byexample is the closest match and covers `#> expect /re/` and `#> expect fail`
almost exactly. What nothing off-the-shelf has is `retry=` / `repeat-until`,
which is the whole point here — these tutorials wait on a scheduler. byexample's
answer would be `$ while ! cmd | grep -q X; do sleep 10; done`, which is exactly
the polling loop the tutorials deliberately do not teach.

So: the *parser* (~110 lines) is reinvented and could have been borrowed; the
*scheduler-waiting* could not. Keeping the custom runner. Do not re-litigate
without new evidence that one of these grew retry support.

Related: `tutorial.py` uses `rich`, not Textual. The tour is a linear scrolling
transcript — output must stay in native scrollback and in plain CI logs, and
Textual would take the alt-screen and make both its own problem. Revisit only if
resume becomes a requirement, which the per-lesson database above is what avoids.
