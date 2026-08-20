# Why slurmise, and how this tutorial works

## 01 — the problem

Knowing how much time and memory to assign a job is very difficult.
Even if you're just running the same command a thousand times,
the runtime might vary depending on what parameters are used.

For example, you might be a biologist who has a script that predicts the
protein structure of a gene and you want to run that script on all ~20,000
genes. You could submit an ARRAY job, but how much time and memory should you
provide? Some genes are small, others are huge.

Or maybe you're an astrophysicist interested in estimating the number of black
holes across different regions of space.

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

## 02 — how the lessons work

Each numbered folder is a self-contained lesson, and the `README.md` inside it
*is* the tutorial — the narration, the commands, and what each command should
do. It's setup for  you to take them in order, but you can skip around if you want.

There are two ways of working through any lesson:

1. Run `./tutorial.py` for an interactive walkthrough. It shows you each
   command, waits for you to press Enter, runs it, and checks the result.
2. `cd` into the lesson folder and type the `$` commands yourself.

Both approaches work through the `README.md` for that lesson, so the content
of the tutorial will be the same regardless of which approach you take.

## 03 — oddities in the code blocks

The lessons' code blocks carry a few lines that might look strange such as:

    $ echo "Hello World!"
    #> expect /World!/

Here `#> expect` describes the expected output of the command above it. For example
this block runs `echo` and checks that "World!" appears somewhere in the output.
Note that this is a shell comments, so if you copy and paste them it should be fine.


The purpose of `#>expect` are for `tutorial.py` to validate against, and they
are why the tutorial can double as an integration test. You can ignore them
when running manually.

Wherever a lesson submits a job, its block offers two ways to do it:

    #> option cluster
    $ sbatch --wait run_perfectScaler.sbatch
    #> expect ok
    #> option mock
    $ bash mock_perfectScaler.sh
    #> expect ok

`cluster` really submits. `mock` runs a script meant to mimic the `.sbatch`
and pre-defines what the job would have used instead instead of submitting.

The benefit of `mock` is that it's much faster than waiting on a SLURM queue,
and allows you to work through the tutorial without even having `sbatch` available.
If you were running slurmise for real, you'd never use the mock approach. This is
just for the tutorial.

## 04 — before you start

Your SLURM setup may require additional flags such as `--account` when you're
submitting the included sbatch files. `sbatch` honours `SBATCH_ACCOUNT`,
`SBATCH_PARTITION` and `SBATCH_QOS` from your environment, which is the easiest
way to supply them without editing every `.sbatch`:

    export SBATCH_ACCOUNT=myaccount

The lessons use short sleep durations so every job finishes quickly. That makes
runtime predictions uninteresting, so lessons focus on predicting **memory**.
