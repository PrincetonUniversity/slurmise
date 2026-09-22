# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "slurmise",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # slurmise record a job

    ## 01 — about this tutorial

    Now we'll see how to record a single job with slurmise. This is the first
    lesson, so it starts from nothing: one job, one recording, and a `predict` that
    can't yet answer from your data.

    `00_introduction/` covers how the lessons work.

    This is the same lesson as `README.md`, as a marimo notebook. Every command is
    shown before it runs, and the cell below it checks the command did what this
    page says it should — the notebook equivalent of the `#> expect` lines in the
    markdown version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 — important files

    `../bin/perfectScaler` is the command we're recording. It's a small python
    script that simulates using a certain amount of time and memory, controlled by
    the `--duration` and `--intensity` command line arguments respectively.

    This uninteresting process is good for a tutorial since we know ahead of time
    how much time and memory the job needs.

    `slurmise.toml` is the config. `job_spec` tells slurmise how to parse the
    command into features, and `default_time` / `default_mem` are the guesses
    slurmise falls back on until a model has been trained:
    """)
    return


@app.cell
def _(mo, show):
    toml_text = show("slurmise.toml")
    mo.md(f"```toml\n{toml_text}\n```")
    return (toml_text,)


@app.cell
def _(toml_text):
    assert "job_spec" in toml_text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `job_spec` has to agree with the command: it names `--intensity` and
    `--duration`, so slurmise expects to find exactly those on the command line it
    is asked to record. `base_dir = "."` is where the `slurmise.h5` database will be
    created.

    `run_perfectScaler.sbatch` runs `perfectScaler` once:
    """)
    return


@app.cell
def _(mo, show):
    sbatch_text = show("run_perfectScaler.sbatch")
    mo.md(f"```bash\n{sbatch_text}\n```")
    return (sbatch_text,)


@app.cell
def _(sbatch_text):
    assert "slurmise --toml slurmise.toml record" in sbatch_text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that `slurmise record` is called **inside the job**, after the `srun`
    finishes. That is the whole lesson: `record` asks slurm what the step it just
    ran actually used, so it has to run on the compute node alongside the work.

    One consequence worth keeping in mind as you go: the command is written out
    twice — once for `srun` to execute, once as a string for `record` to parse
    against the `job_spec`. They have to match.

    The `#SBATCH --output=` line keeps the job's log in `out_slurm_logs/` rather
    than dropping it in this directory. Nothing here prints to it, but your own jobs
    will.

    ## 03 — run it

    On a cluster, submit it. `--wait` blocks until the job finishes, so there's no
    polling loop to write — the command simply doesn't return for about ten seconds.

    Without a cluster, take the `mock` path instead: it calls `slurmise raw-record`
    to assert what the job *would* have used, so nothing is submitted and nothing is
    waited on. That is the only difference; everything below reads the same either
    way.
    """)
    return


@app.cell(hide_code=True)
def _(IN_BROWSER, mo):
    how = mo.ui.radio(
        ["mock"] if IN_BROWSER else ["mock", "cluster"],
        value="mock",
        label="How should the job run?",
    )
    go = mo.ui.run_button(label="Run it")
    mo.vstack([how, go])
    return go, how


@app.cell
def _(MOCK_RECORD, go, how, mo, run):
    mo.stop(
        mo.running_in_notebook() and not go.value,
        mo.md("*Choose above and press **Run it**.*"),
    )

    if how.value == "cluster":
        command = "sbatch --wait run_perfectScaler.sbatch"
    else:
        command = MOCK_RECORD

    recorded = run(command)
    mo.md(f"```console\n$ {command}\n{recorded}\n```")
    return (recorded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Neither `perfectScaler` nor `slurmise record` prints anything, so on the cluster
    path the job's log in `out_slurm_logs/` is empty and there is nothing to read
    there. The recording went into the database instead, which is where we look
    next.

    ## 04 — inspect

    The record outlives the job — it's in `slurmise.h5` now, so you can ask from
    the login node:
    """)
    return


@app.cell
def _(mo, recorded, run):
    _ = recorded
    printed = run("slurmise --toml slurmise.toml print")
    mo.md(f"```console\n$ slurmise --toml slurmise.toml print\n{printed}\n```")
    return (printed,)


@app.cell
def _(printed):
    assert "intensity" in printed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You should see one record for `perfectScaler` with intensity 5000 and duration
    10, alongside the memory and runtime it really used:

    ```
    perfectScaler
    |--- 8844261
    |    |--- duration: () float64 10.0
    |    |--- intensity: () float64 5000.0
    |    |--- memory: () int64 5021
    |    |--- runtime: () int64 12
    attrs:
    ```

    The job id is slurm's, and the memory and runtime are measured, so your numbers
    won't match those exactly on the cluster path. The `intensity` and `duration`
    values will: they were parsed straight out of the command by the `job_spec`.

    ## 05 — predict

    Now ask slurmise what it would predict for a *different* intensity:
    """)
    return


@app.cell
def _(mo, recorded, run):
    _ = recorded
    query = 'slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000 --duration 10"'
    predicted = run(query)
    mo.md(f"```console\n$ {query}\n{predicted}\n```")
    return (predicted,)


@app.cell
def _(predicted):
    assert "Not enough fitting data points" in predicted
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unfortunately, you should see a warning about not enough data points.

    This is because slurmise needs many records per job before it will fit a model,
    and with one it falls back to the defaults. So `234` is `default_time` straight
    out of `slurmise.toml` — no model was consulted at all. `default_mem` isn't
    specified there, so the memory figure is slurmise's own built-in default of 1
    GB.

    The slurmise job-agnostic built-in defaults of 60 minutes and 1 GB are
    arbitrary. It is good practice to set both defaults in your toml so the default
    guess is at least in the right range for your job.

    Now you're ready for the next tutorial, `../02_jobs_in_loop/`: where we
    actually generate enough records to train a model and get good predictions.

    ## Starting over

    Section 05 only holds from a clean slate — with records left over from a
    previous pass, `predict` may have enough data to fit a model and the "not enough
    fitting data points" warning won't appear. So the setup cell at the bottom of
    this notebook throws the database away every time the notebook starts, the same
    way `tutorial.py` runs the `#> reset` block for the markdown version. The logs
    go too, so a second pass doesn't leave you sifting through the last one's.
    """)
    return


@app.cell
def _():
    # --- lesson setup -----------------------------------------------------
    # Clears the database, then defines how this notebook runs a command.
    # Both live in one cell so that every cell calling `run` necessarily runs
    # after the reset -- marimo orders cells by what they use, not by where
    # they sit on the page.
    import re
    import shlex
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    # Pyodide has no fork/exec, so `subprocess` raises there. Nothing in the
    # mock path actually needs a process: it is one `slurmise` call, and
    # slurmise only shells out to `sacct` when it has to look a job up, which
    # `raw-record` does not. So in the browser we call the CLI in-process
    # instead, and the lesson reads the same.
    IN_BROWSER = sys.platform == "emscripten"

    MOCK_RECORD = (
        "slurmise --toml slurmise.toml raw-record "
        "--job-name perfectScaler "
        "--slurm-id 12345 "
        '--numerics \'"intensity":5000,"duration":10\' '
        "--used-minutes 12 "
        "--used-mbs 5020"
    )

    _ANSI = re.compile(r"\033\[[0-9;]*m")

    # marimo bundles the lesson's files under `public/`; copy them next to the
    # notebook so that every command below is spelled exactly as it would be on
    # a cluster, with no browser-only paths leaking into the lesson.
    if IN_BROWSER:
        for _bundled in Path("public").glob("*"):
            shutil.copy(_bundled, _bundled.name)

    for _name in ("slurmise.h5", "fits.json"):
        Path(_name).unlink(missing_ok=True)
    for _stale in list(Path().glob("*.pkl")) + list(Path("out_slurm_logs").glob("*.out")):
        _stale.unlink()
    Path("out_slurm_logs").mkdir(exist_ok=True)

    def show(name: str) -> str:
        """The contents of one of the lesson's files -- `cat`, without a shell."""
        return Path(name).read_text().strip()

    def run(cmd: str) -> str:
        """Run one command and return its output, stdout and stderr together.

        Merged because slurmise reports its warnings on stderr, and the warning
        is often the point -- section 05 is built around one of them.
        """
        if IN_BROWSER:
            from click.testing import CliRunner

            from slurmise.__main__ import main

            if not cmd.startswith("slurmise "):
                msg = f"only slurmise commands can run in the browser, not: {cmd}"
                raise RuntimeError(msg)
            # click >= 8.2 keeps the two streams apart, so stderr is appended
            # explicitly; on 8.1 and earlier `output` already held both.
            result = CliRunner().invoke(main, shlex.split(cmd)[1:])
            code, output = result.exit_code, result.output + result.stderr
        else:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            code, output = proc.returncode, proc.stdout + proc.stderr

        output = _ANSI.sub("", output).strip()
        if code != 0:
            msg = f"`{cmd}` exited {code}\n{output}"
            raise RuntimeError(msg)
        return output

    return IN_BROWSER, MOCK_RECORD, run, show


if __name__ == "__main__":
    app.run()
