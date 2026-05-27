Tutorial
========

This page gives a high-level picture of how slurmise works and how to work
through the hands-on tutorial yourself.

How slurmise works
------------------

slurmise predicts the time and memory a SLURM job will need by learning from
jobs you have already run. The workflow is a loop:

0. **Run.** Use standard srun to launch your command like ``srun myCommand --param1 12``.
   This submits the command to run on a slurm cluster. The tutorial shows
   how to launch sruns from within .sbatch files.
1. **Record.** After a job finishes, ``slurmise record`` reads the job's
   resource usage from SLURM's accounting and stores it in a local hdf5
   database, tagged with the parameters that job ran with (for example the
   input size or a mode flag).
2. **Update.** ``slurmise update-all`` (or ``update-model`` for a single job)
   fits a model to the recorded data for each job.
3. **Predict.** ``slurmise predict`` uses the fitted model to estimate the
   runtime and memory for a new set of parameters, so you can request sensible
   resources before submitting.

A few things that shape what you will see in the tutorial:

- **Configuration lives in a toml file.** Every command takes ``--toml`` (there
  is no auto-discovery). The toml declares each job's ``job_spec`` and any
  default time/memory values.
- **Parameters can be numeric or categorical.** Numeric parameters (declared as
  ``{name:numeric}`` in the toml) let the model interpolate across nearby
  values. Categorical parameters (``{name:category}``) partition the database
  into a separate model per category combination, so each combination needs its
  own data.
- **A model needs enough data before it will fit.** Until a job has accumulated
  enough records, ``predict`` falls back to the defaults from your toml rather
  than guessing from too few points.

Get the tutorial files
-----------------------

After installing slurmise (see :doc:`install`), generate the tutorial sub-directory
in the current directory with:

.. code-block:: bash

   slurmise-generate-tutorial

This writes a ``slurmise-tutorial/`` directory. Use ``--dest`` to choose a
different location.

If you installed slurmise with a runner that isolates the environment, invoke
the script through it, for example:

.. code-block:: bash

   uvx --from slurmise slurmise-generate-tutorial
   # or
   pipx run --spec slurmise slurmise-generate-tutorial

Work through it
---------------

Inside the generated directory you will find a top-level ``README`` and three
self-contained examples. Do them in order — each builds on the last:

1. ``01_single_job/`` — record a single job and see ``predict`` return the toml
   defaults, because one record is not enough to fit a model.
2. ``02_jobs_in_loop/`` — generate enough records in a loop to train models for
   two jobs, then predict and compare a noiseless job against a noisy one.
3. ``03_array_jobs/`` — run jobs in parallel with a SLURM array, and see how a
   categorical job is modeled per category combination.

``cd`` into each directory and follow its ``README.md``. The examples use short
sleep durations so every job finishes quickly; this makes runtime predictions
uninteresting, so the examples focus on predicting **memory**.
