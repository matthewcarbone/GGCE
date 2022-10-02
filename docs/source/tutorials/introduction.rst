========================
Introduction to the GGCE
========================

The GGCE API consists of three major components:

- :class:`ggce.model.Model`: contains all aspects of the Model Hamiltonian to be solved. It also contains information about the level of theory used 
- :class:`ggce.engine.system.System`: a lightweight wrapper for the ``Model`` which initializes the system of equations to be solved
- :class:`ggce.executors.solvers.Solver` (and classes which inherit from this base class): solves the system of equations in various ways, using either SciPy or PETSc
  
In this tutorial, we will demonstrate a simple application of the GGCE method.

Step 1: initialize the model
----------------------------

We begin by initializing a model. Specifically, the model corresponds to an electron-phonon Hamiltonian with Edwards Fermion Boson coupling,

.. math::

    H = -t \sum_{\langle i, j \rangle} c_i^\dagger c_i + \Omega \sum_i b_i^\dagger b_i - g \sum_{\langle i, j \rangle} c_i^\dagger c_j \left( b_j^\dagger + b_i \right)

where in the above, :math:`t` is the electron hopping strength, :math:`\Omega` is the phonon frequency and :math:`g` is the coupling strength. The corresponding model can be initialized in the GGCE API as follows

.. code-block:: python

    from ggce import Model
    model = Model.from_parameters(hopping=0.1)
    model.add_(
        "EdwardsFermionBoson",
        phonon_frequency=1.25,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=2.5
    )

where ``dimensionless_coupling_strength`` is related to :math:`g`. Above, ``1.25`` is the value for :math:`\Omega`, and ``3`` and ``9`` are the values of the maximum cloud extent and maximum number of allowed phonons.

.. hint::

    While we highly recommend reviewing the literature as presented in `Carbone, Reichman & Sous, PRB 104, 035106 (2021) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.035106>`__ for more background on these parameters, the maximum phonon extent and total number of phonons can if nothing else be thought of as convergence parameters. In the limit as these two values approaches infinity, the calculation becomes numerically exact.

Step 2: initialize the system
-----------------------------

The next step is extremely simple. Initializing the ``System`` object will trigger a build of all of the necessary Python objects representing a system of equations to solve:

.. code-block:: python

    from ggce import System
    system = System(model)


.. note::

    The GGCE code implements a comprehensive logger through Loguru. Debugging mode can be controlled using the :class:`ggce.logger.debug` context manager.
