# Why slurmise, and how this tutorial works

## The problem

Knowing how much time and memory to assign a job is very difficult. Even if
you're just running the same command a thousand times, the runtime might vary
depending on what parameters are used.

For example, you might be a biologist who has a script that predicts the protein
structure of a gene and you want to run that script on all ~20,000 genes. You
could submit an ARRAY job, but how much time and memory should you provide? Some
genes are small, others are huge.

Or maybe you're an astrophysicist interested in estimating the number of black
holes across different regions of space.

slurmise aims to help you request an efficient amount of time and memory when
you submit jobs to the SLURM scheduler.

To accomplish this task, slurmise records how much time and memory was used for
a job, as well as whatever parameters you think might be important. It then
attempts to learn from these prior runs.

slurmise might be a good fit if you need to run the same code thousands of times
with different parameters or inputs. It won't be helpful if you just need to run
a script twice, because it won't see enough examples to make a helpful
prediction.

Why should you care about being efficient with SLURM resources? The less
resources you ask for, the shorter your queue times.

## How the lessons work

Each lesson in this book corresponds to a directory in the tutorial tarball —
see {doc}`../../install` for how to get it. Take the lessons in order; each
builds on the one before.

To follow along, open a shell, `cd` into the lesson's directory, and type the
commands as you meet them. Every command block has a copy button in its top
right corner.

**The outputs you see on these pages are real.** Each command block was executed
when this book was built, and what appears beneath it is what the command
actually printed. If a lesson's commands stopped working, or stopped printing
what the surrounding prose claims, the documentation would fail to build rather
than quietly telling you something untrue.

Two consequences worth knowing:

- The pages take the **mock** path wherever a lesson would submit a job, because
  the machine building this book has no scheduler. Each lesson also shows the
  `sbatch` command the mock stands in for.
- Your numbers will not match the page exactly. Measured memory and runtime vary
  from run to run, and on the mock path they are asserted rather than measured.
  The parsed parameters — intensity, duration — will match, because those come
  straight out of the command.

{doc}`../../how-it-works` covers the record / update / predict loop and the
`--account` flags your cluster may need. Start with
{doc}`../01_single_job/lesson` when you're ready.
