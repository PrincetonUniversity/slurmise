# slurmise

Predicting how much time, memory, and cpu cores a SLURM job will need is
notoriously difficult on an HPC. Requesting too little risks out of time or out of memory
failures, while requesting too much results in inefficient cluster
usage and unnecessarily long queue times.

Knowing what resources to allocate is especially difficult when submitting an
unfamiliar job. Only after multiple submissions, and careful monitoring with
tools like `seff` and `reportseff`, can HPC users compare allocated vs. utilized
resources and build an intuition for efficient job submissions in the future.
However, this intuition is difficult to scale for commands that can have widely
varying requirements depending on input data and parameters.

slurmise attempts to address these issues by maintaining a database of previous
job submissions, with extensive metadata missing from SLURM's default database,
that is used to predict the requirements of the current job submission.

## Where to start

- {doc}`install` — getting slurmise onto your cluster.
- {doc}`overview` — whether slurmise fits your work, and the record /
  update / predict loop.
- The **Tutorial** lessons — a hands-on walkthrough, starting with
  {doc}`tutorial`. Every command on those pages was really
  run when this page was built, and the output shown beneath it is what the
  command really printed.
- {doc}`api` — the Python API, for using slurmise as a library.
