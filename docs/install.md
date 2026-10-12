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

To work through the tutorial, {doc}`lessons/00_introduction/lesson` covers
getting its files.
