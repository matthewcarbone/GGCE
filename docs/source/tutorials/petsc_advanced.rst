.. _petsc-advanced:

=======================================
GGCE-PETSc interface: advanced features
=======================================

In this tutorial we go over two advanced features enabled by the GGCE-PETSc interface:
the double-parallel scheme and the basis-matrix separation, which together unlock
large-scale phonon configuration calculations.

Double-parallel scheme
----------------------

In Sec. :ref:`parallel`, we alluded to the double-parallel scheme of running GGCE.
To recap: in the double-parallel scheme, the calculation is MPI-parallelized BOTH across different
:math:`(k,\omega)` points, AND within the solution of the matrix equation at each individual
point. To enable and control this feature when running GGCE-PETSc, we use the ``brigade_size``
keyword when instantiating the Solver.

.. code-block:: python

  COMM = MPI.COMM_WORLD
  solver = MassSolverMUMPS(system=system, mpi_comm=COMM, brigade_size=1)

This is the only syntactic change required to enable this scheme.

By a "brigade", we mean a group of MPI processes
that are collectively working on a single :math:`(k,\omega)` point. The keyword
``brigade_size`` is used in the code to split all MPI processes into brigades of
**equal** size ``brigade_size``. Each brigade works collectively on the matrix for a single
:math:`(k,\omega)` point, and together the brigades cover the entire :math:`(k,\omega)`
space.

By default, ``brigade_size=WORLD_SIZE``, meaning that there is only one brigade,
and the Solver proceeds sequentially through all :math:`(k,\omega)` points whilst
maximally distributing the matrix of equations across all available MPI processes.
The main limitation for this keyword is that the number of MPI processes should be
evenly divisible by ``brigade_size``, otherwise an Error is raised and the calculation
is terminated.

.. note ::

  Due to the specifics of the PETSc
  API, the number of "jobs" fed into ``Solver.greens_function(k, w, eta)`` --
  that is, ``len(k)*len(w)`` -- is required to be evenly divisible by the number of brigades
  (otherwise the PETSc solvers hang indefinitely). The Solver automatically extends (pads)
  the :math:`k, \omega` arrays with the minimum needed number of extra points to
  prevent this. The padding is done to the :math:`omega` array by default, unless it
  is of length 1: in that case, the :math:`k` array is padded. The padding is always
  done outside the range specified by the user, so the input :math:`k,\omega` arrays
  are NOT redefined but merely extended, so that the user still has perfect
  control over where the calculations are being carried out.

  All calculations, including over the padded points, are returned by
  ``greens_function``. This is why sometimes the output array of ``greens_function()``
  may have slightly larger dimensions than the original input array. This is of
  course easily fixed by merely indexing the output array up to the lengths of the
  original :math:`k,\omega` arrays

  .. code-block:: python

    result = solver.greens_function(k, w, eta)
    result_notpadded = result[:len(k),:len(w)]

The optimal choice of ``brigade_size`` is ultimately up to the user. For small
systems where the matrix can be easily stored and solved by a single CPU, the
optimal speed-up is accomplished with maximum splitting, i.e. ``brigade_size=1``.
When dealing with larger matrices (say of dimension more than 100,000), optimal
speed-up will have to be determined by experimenting. In practice, ``brigade_size``
should be just enough for the matrix solution to be possible within MUMPS in terms of
memory and time, so that the largest number of independent brigades can be formed.

For the calculation in the previous section, we see that the sizes of matrices
being solved are quite small (merely hundreds of equations). This allows us to set
``brigade_size=1`` and enjoy linear speed-up in the calculation.

Basis-matrix separation
-----------------------

Without a double-parallel scheme, GGCE is limited to phonon configurations where
the matrix produced can be reasonably quickly solved by SciPy's sparse direct
solver (<100,000 or so). Using the double-parallel scheme removes this restriction --
if a suitable number of MPI processes (CPUs) is available, the matrix can be
appropriately segmented, stored and solved in parallel, allowing much larger matrices
to be handled.

However, one memory bottleneck still remains in GGCE: the basis. The basis, as
computed and stored inside the ``System`` object, is at core a Python ``dict``
that encodes the structure of equations induced by a chosen phonon cloud variational
configuration. It does this agnostic of particular values of :math:`k, \omega, \eta`,
which is a great advantage as it allows to only compute the basis once and then easily
generate all matrices for a range of :math:`k,\omega` values.

And yet the basis itself constitutes a problem for calculations with large configurations.
Being a ``dict``, the basis is inherently not parallelizable: it cannot be shared across
MPI processes. Instead, each process **has an individual copy** of the entire basis,
resulting in incredible redundancy and memory usage.

One solution to this problem is to change the basis generation and storage approach:
this is one of the goals of the next release. In the meantime, another solution is to
separate the computation of the basis from the actual calculation of the Green's
function, by pre-computing and storing on disk the matrices to be solved by the
sparse direct solver.

This is accomplished within the ``MassSolverMUMPS`` in two stages. First,
we must define the directory for saving both the matrices to be solved and the
``Model`` and ``System`` objects

.. code-block:: python

    matr_dir = p["root_matr_dir"]
    root_dir = p["root_sys"]

    model = Model.from_parameters(...)
    model.add_(...)

Next, we create the first ``MassSolverMUMPS`` instance, the purpose of which is
to create the matrices and save them to disk.

.. code-block:: python

    sysgen_petsc = MassSolverMUMPS(
                                  system=System(model),
                                  root=root_dir,
                                  mpi_comm=COMM,
                                  matr_dir=matr_dir,
                                  brigade_size=1,
                                  )

Setting the ``matr_dir`` keyword determines the location where the matrices will
be saved. Notice that the creation of matrices is in principle parallelizable:
while the basis is not shareable, in the sense that each MPI process will still
have a redundant copy of it, groups of MPI processes can still work on
generating and writing different matrices to disk. In especially dire cases,
to save memory, a smaller number of MPI ranks should be used while keeping
allocated memory constant.

To prepare the matrices and save them to disk, we use the following command

.. code-block:: python

    sysgen_petsc.prepare_greens_function(k, w, eta)

    del sysgen_petsc  # to free up memory used to store the basis

The matrices are dumped to disk using ``pickle`` in sparse matrix format,
meaning that we save three arrays -- row indices, column indices, and values --
for nonzero entries only. They are saved with a filename formatted as
``matr_at_k_{kval:.10f}_w_{wval:.10f}_e_{etaval:.10f}.pkl``.

Once the matrices are written to disk, the solver object, which contains the
basis, is deleted to free up memory.

Next, we create the Solver that will actually do the solving. Notice that we need
to create a new System object with ``autoprime=False`` keyword AND to pass this
keyword to ``MassSolverMUMPS`` to make sure that the basis is not generated when
we pass the System to the Solver.

.. code-block:: python

    system_unprimed = System(model, autoprime=False)
    executor_petsc = MassSolverMUMPS(
                                    system=system_unprimed,
                                    root=root_dir,
                                    mpi_comm=COMM,
                                    matr_dir=matr_dir,
                                    autoprime=False,
                                    brigade_size=1,
                                    )

    results_petsc = executor_petsc.greens_function(k, w, eta)

This second Solver then looks in the ``matr_dir`` directory and loads the matrices
from there, constructing filenames from the ``k,w,eta`` it was given.

We showed how to run basis-matrix separate calculations within a single Python
script. Alternatively, this can be run as two separate, sequential calculations
(i.e. with different batch job scripts), using different memory and CPU counts to
optimize resource usage.
