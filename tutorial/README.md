# Quickstart

Download the tutorial files as folder named slurmise-tutorial/ in your current
directory:

```bash
wget -qO- https://github.com/PrincetonUniversity/slurmise/releases/latest/download/slurmise-tutorial.tar.gz | tar -xz
cd slurmise-tutorial
```

Run `./tutorial.py` to get an interactive tutorial.

Or alternatively, `cd lesson-01_single_job/` and work through `README.md`
if you want to type the commands yourself.

# Introduction and motivation of slurmise

Knowing how much time and memory to assign a job is very difficult.
Even if you're just running the same command a thousand times,
the runtime might vary depending on what parameters are used.

For example, you might be a biologist who has a script that predicts the
protein structue of a gene and you want to run that script on all ~20,000
genes. You could submit an ARRAY job, but how much time and memory should you
provide? Some genes are small, others are huge.

Or maybe you're an astrophysicst interested in estimating the number of black holes
across different regions of space.

slurmise aims to help you request an efficient amount of
time and memory when you submit jobs to the SLURM scheduler.

To accomplish this task, slurmise records how much time and memory was used
for a job, as well as whatever parameters you think might be important. It
then attempts to learn from these prior runs.

slurmise might be a good fit if you need to run the same code thousands of
times with different parameters or inputs. It won't be helpful if you just
need to run a script twice because it won't see enough examples to make a
helpful prediction.

Why should you care about being efficient with SLURM resources?
The less resources you ask for, the shorter your queue times!

# About the tutorials

Each subfolder like `lesson-01_single_job/` is a self-contained
lesson and each `lesson-*/README.md` is the tutorial.

We've made a python script `tutorial.py` which parses these `README.md`'s
to save you some typing, but you can work through them manually if you'd prefer.

## Tutorial specific oddities

1. You'll see some lines like `#> expect` which are parsed by `tutorial.py`
to verify the output of the commands, but you can ignore these lines.

2.  Wherever a lesson submits a job, its block offers two ways to do it:

```bash
#> option cluster
$ sbatch --wait run_perfectScaler.sbatch
#> expect ok
#> option mock
$ bash mock_perfectScaler.sh
#> expect ok
```

`cluster` really submits. `mock` runs a script shipped beside the `.sbatch`
that states what the job would have used. The benefit of `mock` is that it's
much faster than waiting on a SLURM queue.

If you were running slurmise for real, you'd never use the mock approach.

3. Your SLURM setup may require additional flags such as `--account` when you're
submitting the included sbatch files. `sbatch` honours `SBATCH_ACCOUNT`,
`SBATCH_PARTITION` and `SBATCH_QOS` from your environment, which is the easiest
way to supply them without editing every `.sbatch`.
