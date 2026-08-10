"""Standalone entrypoint that copies the bundled getting-started tutorial.

Kept separate from :mod:`slurmise.__main__` so it needs no ``--toml`` file and
pulls in none of the heavy modeling imports.
"""

from __future__ import annotations

import stat
from fnmatch import fnmatch
from importlib.resources import as_file, files
from pathlib import Path

import click

# Artifacts produced by running the tutorial; ship the starting point only.
# In editable installs the source tree is copied directly, so these must be
# filtered here rather than relying on the wheel's exclude-package-data.
EXCLUDE_PATTERNS = (
    "slurm*.out",
    "*.h5",
    "*.pkl",
    "fits.json",
    "slurm_outs",
    "out_slurm_logs",
    "local.sql",
    "__pycache__",
    "05_slurmise_run",  # wip, only run through a clone of the repo
    "INTERACTIVE_TUTORIAL_PLAN.md",  # internal notes about the tutorial, not part of it
)


def _copy_tree(src: Path, dest: Path) -> None:
    """Recursively copy ``src`` into ``dest`` using only pathlib."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if any(fnmatch(item.name, pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        target = dest / item.name
        if item.is_dir():
            _copy_tree(item, target)
        else:
            target.write_bytes(item.read_bytes())


@click.command()
@click.option(
    "--dest",
    type=click.Path(file_okay=False),
    default="slurmise-tutorial",
    help="Directory to copy the tutorial into (created if needed).",
)
def main(dest):
    """Copy the slurmise getting-started tutorial into DEST."""
    with as_file(files("slurmise") / "tutorial") as src:
        _copy_tree(Path(src), Path(dest))

    # Restore the executable bit on the example job scripts; wheel installs
    # don't reliably preserve it, and the sbatch files invoke them directly.
    # Same for the `tutorial.py` walkthrough, which the README tells the reader
    # to run as `./tutorial.py`, and each lesson's no-cluster `mock_*.sh`.
    executables = [p for p in (Path(dest) / "bin").glob("*") if p.is_file()]
    executables += [p for p in [Path(dest) / "tutorial.py"] if p.is_file()]
    executables += [p for p in Path(dest).glob("*/mock_*.sh") if p.is_file()]
    for script in executables:
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    click.echo(f"Tutorial written to {dest}/")
    click.echo(f"Run `cd {dest} && ./tutorial.py` to begin, or read 01_single_job/README.md.")


if __name__ == "__main__":
    main()
