Tutorial
========

This page gives a high-level picture of how slurmise works and how to work
through the hands-on tutorial yourself.

Get the tutorial files
-----------------------

The tutorial is attached to each slurmise release as a tarball. Unpack it
anywhere:

.. code-block:: bash

   wget -qO- https://github.com/PrincetonUniversity/slurmise/releases/latest/download/slurmise-tutorial.tar.gz | tar -xz
   cd slurmise-tutorial

That gives you a ``slurmise-tutorial/`` directory. Also make sure you
so install slurmise too (see :doc:`install`).

Inside the ``slurmise-tutorial`` you will find a top-level ``README`` and
multiple self-contained lessons that you can work through in order.

Either run ``./tutorial.py``, which walks a lesson with you — showing each
command before it runs it and checking the result — or ``cd`` into a lesson and
type the commands from its ``README.md`` yourself.

Every lesson offers a no-cluster path, so you can take the whole tutorial
without submitting anything.

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
