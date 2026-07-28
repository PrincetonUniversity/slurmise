# Plan: one interactive tutorial, driven by a markdown runner

Status: proposal, not started. Prompted by the `slurmise_run/` prototype, where
`tutorial.md` holds the prose, the commands, and each command's expected outcome,
and `tutorial.py` interprets it. This plan is about bringing `01_single_job`,
`02_jobs_in_loop`, and `03_array_jobs` onto that same footing.

## Why

Today each example is a `README.md` telling you what to type plus a `.sbatch`
that actually does it, and `tests/integration/run_tutorial_slurm.sh` submits
those `.sbatch` files so CI notices when they break. The content is therefore
specified twice — once as prose for a reader, once as a script for CI — and the
two drift. The README can describe output the job no longer produces and nothing
fails.

The runner collapses that: one `tutorial.md` per tutorial is the only source of
truth, each command carries a `#>` expectation, and a reader can either read the
markdown and type along or let `tutorial.py` drive it. The tutorial becomes its
own integration test.

## The one real design decision

`slurmise_run` runs everything on the login node: `./slrmise` submits, syncs, and
prints, so every interesting command and its output happen in the reader's
session and are trivially checkable.

These tutorials are the opposite. `slurmise record` is *meant* to be called from
inside the job — that is the lesson — so the interesting work happens on a
compute node and lands in `out_slurm_logs/slurm-*.out`:

```bash
srun ../bin/perfectScaler --intensity 5000 --duration 10
slurmise --toml slurmise.toml record "perfectScaler --intensity 5000 --duration 10"
slurmise --toml slurmise.toml print
```

**Recommendation: keep `record` inside the job and have the markdown orchestrate
around it.** Do not restructure these lessons to record from the login node —
that would teach the wrong thing. The markdown drives:

```bash
$ sbatch --wait run_perfectScaler.sbatch
#> expect /Submitted batch job/
$ cat out_slurm_logs/slurm-perfectScaler-*.out
#> expect /perfectScaler.*intensity/
```

`sbatch --wait` blocks until the job finishes, so this needs **none** of
`slurmise_run`'s `retry=` / `repeat-until` machinery — the hardest part of that
prototype is simply not required here. The runner should work as-is.

## Section mapping

| today | becomes | commands |
|---|---|---|
| `01_single_job` | 01 — record one job | `cat` the toml and sbatch; `sbatch --wait`; read the `.out`; `slurmise predict` showing the cold default |
| `02_jobs_in_loop` | 02 — enough runs to fit | the loop sbatch; `sbatch --wait`; `slurmise print`; `slurmise predict` now model-based |
| `03_array_jobs` | 03 — arrays and categories | array sbatch; the `categoricalScaler` job showing a non-numeric feature |

Keep the `.sbatch` files exactly as they are — they are the artifact being
taught, and they are what a reader adapts for their own work. Only the READMEs
are replaced.

## Consolidating the tomls

The three `slurmise.toml` files differ only in which `job_spec`s they declare and
their defaults:

- `01`: `perfectScaler`, `default_time = 234`, no `default_mem`
- `02`: adds `complexMemScaler`, `default_mem = 5000`, `default_time = 30`
- `03`: adds `categoricalScaler` (with a `{scaling:category}` feature), no defaults

One toml declaring all three `job_spec`s is straightforward. The defaults are
not: lesson 01's "no model yet, here is the cold default" depends on the specific
numbers, so reconciling them changes what the lesson shows.

Treat this as the riskiest part of the migration. In `slurmise_run/`, folding
three tomls into one silently broke lesson 04 — the OOM demonstration stopped
producing an OOM, and only a real-scheduler run caught it. Whatever defaults are
chosen here, re-derive each lesson's expected numbers on a real cluster rather
than by arithmetic.

## Sequencing: the significant behaviour change

Each example currently has its own `slurmise.h5`, so any of them can be run
alone. A single folder means a single database, and the lessons become ordered:
after 02 has trained a model, 01's `predict` no longer returns a cold default.

That is arguably more honest — it is how slurmise actually accumulates history —
and it is how `slurmise_run` already works. But it means:

- a `#> reset` block must run first (as `slurmise_run/tutorial.md` does), and
- you can no longer jump to lesson 03 in isolation.

`tutorial.py` currently has no `--from` / `--stop-after`; they were deliberately
removed because `--mock --yes` made a full re-run cheap. These tutorials submit
real jobs with no mock, so a full re-run is *not* cheap, and resume may need to
come back. **Decide this before migrating**, since it shapes whether one combined
`tutorial.md` or three (one per subfolder, sharing a runner) is the better shape.

## What the runner needs

Probably nothing. `tutorial.py` imports nothing from slurmise and takes its
content entirely from `tutorial.md`. Worth confirming during the work:

- It hardcodes `TUTORIAL_MD = HERE / "tutorial.md"`; sharing one runner across
  three folders means either a copy per folder or a path argument.
- It sets `SBATCH_OUTPUT`, which these `.sbatch` files override with their own
  `#SBATCH --output=`. Harmless, but the prose should not claim otherwise.
- `#>` has no "assert on a file" directive — `cat … #> expect /re/` covers it.

## Migration steps

1. Decide one combined `tutorial.md` vs three; decide whether resume returns.
2. Pick the merged toml and re-derive each lesson's numbers **on a real cluster**.
3. Convert `01_single_job/README.md` first, end to end, as the pattern.
4. Convert 02 and 03 against that pattern.
5. Replace the `EXAMPLES` / `example_path` token machinery in
   `run_tutorial_slurm.sh` with a single `tutorial.py` invocation — the markdown
   supplies the ordering, so the mapping disappears. The accounting probe stays
   in shell: it is a precondition, not a lesson.
6. Update the top-level `README.md`. It currently links `lazy_recording/`, which
   no longer exists — that folder is now `slurmise_run/`.

## Verification

- A green `--mock` run proves lesson *logic* only. It cannot prove lesson
  *waiting*, because mocked jobs settle instantly — that gap hid two real bugs in
  `slurmise_run`. There is no mock here anyway, which is a point in this
  migration's favour.
- Run the full tour on a real cluster before and after each lesson is converted,
  and compare the recorded table — the numbers should not move.
- `sbatch --wait` on an array job waits for every task; confirm lesson 03's
  expectations still hold when tasks finish out of order.

## Note

This file is internal planning. `generate_tutorial.py` currently copies the whole
tutorial tree, so it will be shipped to users unless it is added to
`EXCLUDE_PATTERNS` alongside `slurmise_run`.
