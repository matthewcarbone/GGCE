.. _parallel:

========================
Running GGCE in parallel
========================

Available parallel schemes
--------------------------

When using a GGCE solver to obtain the spectral function through a solution of a system of
equations, there are two ways in which parallelization might be helpful.

1. When computing the spectral function at a variety of :math:`k, \omega` points,
parallelizing across those points gives a linear speed-up -- because the calculations
at different points are fully independent on each other. This is the so-called
*embarrassingly parallel* situation, and we call this parallelization scheme
the **across-points scheme**.

2. In a large variational space (i.e. with large values for ``phonon_number`` and
``phonon_extent``), the matrices to be solved at each :math:`k,\omega` point can
become large even for sparse solvers. The advantage of sparse solvers is that it is
possible to parallelize the solution of a sparsely-represented matrix across MPI
ranks, speeding up the process (albeit in general sub-linearly). We call this the
**matrix-level** scheme.

If you have configured ``mpi4py``, you can directly run standard GGCE Solver classes,
``SparseSolver`` and ``DenseSolver``, in parallel according to the first scheme --
across different :math:`k,\omega` points.

If you have configured ``PETSc``, you can actually have both across-points and
matrix-level parallelization simultaneously, by employing groups or *brigades* of MPI
ranks together. We refer to this combination as the *double-parallel* scheme
and it is described in more detail in :ref:`petsc-advanced`.

Parallelization primer
----------------------

Two things are needed to enable parallelization within ``Solver`` operation.

First, you need to run the script with an MPI executable. For example, to run
a Python GGCE script with MPI, using four processes

.. code-block:: bash

  mpirun -np 4 python script.py

Alternatively, ``mpiexec`` can be used depending on your cluster's configuration.
(Locally, these are equivalent.)

Second, we need to provide the ``Solver`` with an ``MPI`` communicator. This object
is wrapped in ``mpi4py`` and can be accessed in the following way

.. code-block:: python

  from mpi4py import MPI

  COMM = MPI.COMM_WORLD

.. note::

  It is not enough to install ``mpi4py`` to have access to MPI objects and be able
  to run MPI parallel calculations. By itself, ``mpi4py`` is merely a wrapper for
  actual executables provided by a particular implementation of the MPI standard
  such as `openmpi <https://www.open-mpi.org/>`__
  or `mpich <https://www.mpich.org/>`. Even if a ``pip`` installation of ``mpi4py``
  does not throw an error, a good test to make sure that ``mpi4py`` has been
  installed and linked properly against an installation of an MPI implementation
  are the following two commands one can run in the terminal

  .. code-block:: bash

    python -c "import mpi4py"

    python -c "from mpi4py import MPI"

  If both these commands execute without errors, you are likely ready to execute MPI
  calculations.

The object ``COMM`` is an MPI global communicator wrapper. It provides a variety of
methods for interacting with and controlling different MPI processes. For example,
you can get the ``WORLD_SIZE`` -- i.e. the total number of processes running the
calculation

.. code-block:: python

  print(COMM.Get_size())

In our calculation above with four processes, this would return, predictably

.. code-block:: bash

  4
  4
  4
  4

Each process executes the print command separately.

We can also print the rank (sequential label) of each MPI process running the script.

.. code-block:: python

  print(COMM.Get_rank())

What will this print? Since ``WORLD_SIZE`` is a variable that has the same value
on all MPI processes, the same number is printed. But each process has a different rank,
so different numbers will be printed.

.. code-block:: bash

  3
  1
  4
  2

The order will not necessarily be sequential: all processes rush to write to the
output at once, and contingent on the situation on a given CPU, will get there
at different times. This might even change from execution to execution.

Getting the rank of a given process can be useful if in the same
script there are sequential and parallel parts. The easiest way to execute part
of the code sequentially (for example, for printing the results at the end)
is to introduce an ``if`` block

.. code-block:: python

  (setting up Models, Systems, Solvers, running parallel calculations)

  if COMM.Get_rank() == 0:
    (do sequential stuff here)

The ``== 0`` part is convention -- usually sequential portions of the code are
reserved for the so-called "head rank" -- but could of course be any of the processes.

With this, we are ready for a parallel GGCE script.

Across-points scheme
--------------------

As mentioned above, we import the communicator and pass it to the Solver during
instantiation.

.. code-block:: python

  from ggce import Model, System, DenseSolver
  from mpi4py import MPI

  COMM = MPI.COMM_WORLD

  mymodel = Model.from_parameters(...)
  mysystem = System(mymodel)
  mysolver = Solver(system=mysystem, mpi_comm=COMM)

And that's it! When we run ``.greens_function()`` on some momentum and frequency
arrays, the ``Solver`` class instance will automatically parallelize the calculation
across available ranks. In particular, if we do

.. code-block:: python

  results = mysolver.greens_function(kgrid, wgrid, eta = 0.005, pbar = True)

we will see linear speed-up, with the ranks splitting up the work. The progress
bar will note this automatically and be proportionally shorter.

One important idiosyncracy of the ``.greens_function()`` method is that only the
head node (rank = 0) returns the result -- the others result a pythonic ``None``.
Subsequent processing of results -- such as taking the imaginary part to get the
spectral function -- must be restricted to an ``if COMM.Get_rank() == 0`` block
for this reason.

Matrix-level scheme
-------------------

As mentioned at the top of this tutorial, this scheme is not available without PETSc.
The SciPy sparse solver does have some rudimentary multithreading controlled by
the ``OMP_NUM_THREADS`` parameter (see :ref:`Multithreading SciPy solvers <scipythread>` for more details).

See the next tutorial titled :ref:`petsc-intro` about using the matrix-level scheme with PETSc.
The advanced double-parallel scheme will be described in :ref:`petsc-advanced`.
