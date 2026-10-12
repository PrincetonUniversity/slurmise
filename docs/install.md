# Installation

::::{tab-set}

:::{tab-item} pip
```bash
pip install slurmise
```
:::

:::{tab-item} pipx
```bash
pipx install slurmise
```

pipx gives the `slurmise` command an environment of its own, which suits using the
CLI by itself. You won't be able to `import slurmise` from your own Python, so for
the Python API or the Snakemake integration use pip or uv instead.
:::

:::{tab-item} uv
```bash
uv tool install slurmise
```

Like pipx, this installs only the `slurmise` command. To use slurmise from your own
code, add it to your project instead:

```bash
uv add slurmise
```
:::

::::

slurmise needs Python 3.11 or newer. It talks to SLURM by running `sacct`, so it
has to be installed somewhere that SLURM's accounting commands are on the
`PATH` — normally a login or compute node of the cluster itself.

## Getting the tutorial files

The tutorial lessons in this book run real commands against real files: a
`slurmise.toml` config, a `.sbatch` script, and a couple of toy programs that
burn a known amount of time and memory. Those files ship as a tarball attached
to each release:

```bash
wget -qO- https://github.com/PrincetonUniversity/slurmise/releases/latest/download/slurmise-tutorial.tar.gz | tar -xz
cd slurmise-tutorial
```

That gives you one directory per lesson. Work through the lessons in this book
and type — or copy — the commands into a shell inside the matching directory.
