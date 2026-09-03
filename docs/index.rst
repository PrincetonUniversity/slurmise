.. Slurmise documentation master file, created by
   sphinx-quickstart on Tue Sep 17 13:01:01 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Slurmise
========

Predicting how much time, memory, and cpu cores a SLURM job will need is
notoriously difficult, especially if the same job needs to be run multiple
times with different parameters and on differently sized inputs.

Requesting too little risks out of time or out of memory
failures for some jobs, while requesting too much results in inefficient cluster usage and
unnecessarily long queue times.

slurmise attempts to address these issues by maintaining a database of previous jobs
that is used to predict the requirements of the current job submission.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :caption: Submodules and API

   modules
