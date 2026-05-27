"""Standalone entrypoint that copies the bundled getting-started tutorial.

Kept separate from :mod:`slurmise.__main__` so it needs no ``--toml`` file and
pulls in none of the heavy modeling imports.
"""

from __future__ import annotations

import stat
from importlib.resources import as_file, files
from pathlib import Path

import click


def _copy_tree(src: Path, dest: Path) -> None:
    """Recursively copy ``src`` into ``dest`` using only pathlib."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
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
    bin_dir = Path(dest) / "bin"
    if bin_dir.is_dir():
        for script in bin_dir.iterdir():
            if script.is_file():
                script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    click.echo(f"Tutorial written to {dest}/")
    click.echo("Open the README there, then `cd` into 01_single_job/ to begin.")


if __name__ == "__main__":
    main()
