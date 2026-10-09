# slurmise

Predicting how much time, memory, and cpu cores a SLURM job will need is
notoriously difficult, especially if the same job needs to be run multiple
times with different parameters and on differently sized inputs.

Requesting too little risks out of time or out of memory failures for some jobs,
while requesting too much results in inefficient cluster usage and unnecessarily
long queue times.

slurmise attempts to address these issues by maintaining a database of previous
jobs that is used to predict the requirements of the current job submission.

## Where to start

- {doc}`install` — getting slurmise onto your cluster.
- {doc}`how-it-works` — the record / update / predict loop, and the vocabulary
  the tutorial uses.
- The **Tutorial** lessons — a hands-on walkthrough. Every command on those
  pages was really run when this page was built, and the output shown beneath it
  is what the command really printed.
- {doc}`api` — the Python API, for using slurmise as a library.
