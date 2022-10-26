============
Installation
============

The following text documents how one should install the GGCE package. Note that currently only support for **Linux** and **MacOS (Intel)** is supported.

.. warning::

   We cannot guarantee that the package works correctly on Windows operating systems (although it has been tested in `Windows Subsystem for Linux (WSL/WSL2) <https://learn.microsoft.com/en-us/windows/wsl/about>`__). We also cannot guarantee that the package will work correctly on ARM-based Mac M1 or M2 machines, as these are still quite new, and occasionally some packages that GGCE depends on may break.

Standard installation
---------------------

You can install the GGCE package via pip like many others. This will install the default dependencies for using GGCE on e.g. your laptop or on other "single-node" machines.

.. code-block:: bash

   pip install ggce

Standard development installation
---------------------------------

If you wish to help develop GGCE, you can install a local version via

TK

.. code-block:: bash

   pip install e ".[dev]"

which will also install all the optional development requirements in ``requirements-dev.txt``. We generally recommend using a virtual environment via e.g. ``conda`` to ensure reproducibility and so that there are no package conflicts.

Advanced installation via pip
-----------------------------

The GGCE code can be installed with advanced capabilities. There are two particular options: MPI and PETSc (which requires MPI). The following documents the installation procedure for each of these.

TK

Advanced manual installation
----------------------------

When all else fails, which can happen for specific computing architectures or edge cases, we can always resort to manual installation. Here, we detail much more specifically how to install the GGCE dependencies correctly for massively parallel computation.

We use ``conda`` for our software development environment, and ``pip`` whenever necessary for installing new packages. In principle any distribution of ``conda`` should work. We recommend ``miniforge3``.


MPI installation
^^^^^^^^^^^^^^^^

Some of the functionality of the GGCE code can be improved by parallelizing calculations across multiple machines. We use MPI and the Python wrapper ``mpi4py`` to accomplish this. The following will detail the installation instructions for MPI-accelerated GGCE.

#. First, create and activate a fresh ``conda`` `environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__ using ``python=3.9``.

   .. code-block:: bash

      conda create -n my_env python=3.9 -y && conda activate my_env

#. Next, make sure you have the correct MPI module loaded. Usually this is done via some ``module load``, where generally the package is something like ``openmpi/gcc/64/3.1.1``.

#. Point the ``MPICC`` path to the executables corresponding to the loaded modules. This can usually be done by ``export MPICC=$(which mpicc)``, where you'll want to check via ``echo $MPICC`` that this is indeed the correct path.

#. Install ``mpi4py`` using ``pip``, *not* ``conda``. ``conda`` tends to ignore the cluster executables in favor of its own locally installed versions, and then when running calculations, the correct executables will not be detected and all processes will correspond to rank 0.

   .. code-block:: bash

      pip install mpi4py

#. Install ``numpy`` and ``scipy`` packages. You'll also want to check that this linked successfully to e.g. ``mkl`` (you can use ``numpy.show_config()`` and ``mp4py.get_config()``).



Enable usage of PETSc for massively parallel computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`PETSc <https://www.mcs.anl.gov/petsc/index.html>`__ is a software library created and maintained at the Argonne National Laboratory for massively parallel solution of large linear systems of equations (e.g. finite-difference approaches to PDEs). It provides an interface to a long list of linear system solution methods (MUMPS, PARDISO, Krylov), preconditioners, and data structures that allow rapid optimization of the solution method for the problem at hand.

Since PETSc relies on MPI for multi-core parallelization, just as above, do not use ``conda`` to install mpi4py. Below are installation instructions
for a typical Linux cluster (tested on the LISA cluster at University of British
Columbia's Stewart Blusson Quantum Matter Institute, the Cedar cluster at the WestGrid consortium (Compute Canada network) and the Institutional Cluster at the Scientific Data and Computing Center, Brookhaven National Laboratory).


Institutional Cluster installation (Brookhaven National Lab / LISA SBQMI)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Unlike other examples as presented here, we will use ``pip``'s software environment manager, as often times ``conda`` does not play nice with many high performance computing clusters.

#. Load the correct modules and create a software environment.

    .. code-block:: bash

        module load python/3.7
        python3 -m pip install --user virtualenv
        python3 -m venv ggce_env
        source ggce_env/bin/activate

#. Load the *correct* ``openmpi`` module.

    .. code-block:: bash

        module load openmpi

   This will probably be something like ``/hpcgpfs01/software/openmpi/3.1.1-gnu/bin/mpicc`` (at least as of July 2021).

#. Using ``pip``, install ``mpi4py``.

    .. code-block:: bash

        pip install mpi4py

   This should result in something like the following when checking the ``mpi4py`` config in Python:

    .. code-block:: python

        import mpi4py
        mpi4py.get_config()
        {
            'mpicc': '/hpcgpfs01/software/openmpi/3.1.1-gnu//bin/mpicc',
            'mpicxx': '/hpcgpfs01/software/openmpi/3.1.1-gnu//bin/mpicxx',
            'mpifort': '/hpcgpfs01/software/openmpi/3.1.1-gnu//bin/mpifort',
            'mpif90': '/hpcgpfs01/software/openmpi/3.1.1-gnu//bin/mpif90',
            'mpif77': '/hpcgpfs01/software/openmpi/3.1.1-gnu//bin/mpif77'
        }

#. Set required environment variables.

    .. code-block:: bash

        export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack"

    .. warning::

        This step is extremely important. For example, if the scalar type is not set to complex, PESTc will compute all quantities using real numbers only *but will not warn the user*. This can cause all spectral functions to inadvertently be 0, and of course the Green's functions will be totally incorrect as well.

#. Finally, install both ``petsc`` and ``petsc4py``.

    .. code-block:: bash

        pip install petsc petsc4py

   This step might fail quite a few times as ``pip`` tries to figure out the right files to use to build these packages, but usually it succeeds in the end.


LISA cluster installation (University of British Columbia)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#. Again, make sure to be in your freshly installed conda environment, with the
   correct MPI packages and their corresponding compilers loaded. Either MPICH or
   OPENMPI should work: currently tested with MPICH and gcc compilers.
#. Install ``cython`` via ``conda``. This is needed for the PETSc C-oriented Python
   bindings.
#. `Download PETSc source <https://www.mcs.anl.gov/petsc/download/index.html>`_ (git clone recommended).
#. Unzip and navigate to the directory.
#. Run ./configure with the following flags (see the `PETSc list of common usage <https://www.mcs.anl.gov/petsc/documentation/installation.html#exampleusage>`_ for a complete list of possible flags).

   * ``--with-batch`` -- is needed on cluster systems which only allow job batch submissions.
   * ``--with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90`` -- this lets PETSc know to use the same compilers as were used in the creation of your loaded MPI module.
   * ``--download-fblaslapack --download-mumps --download-scalapack`` -- download any external solvers that are desired and their dependencies (for a complete list see the `PETSc manual, sec. 2.3.7 <https://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf>`_\ )
   * ``--with-scalar-type=complex`` -- by default PETSc is compiled for real number vectors and matrices, this switches it to complex data types.
   * ``--download-petsc4py`` -- this downloads the Python bindings petsc4py, allowing to access PETSc routines and data structures easily from within Python

#. Once the commands executes, in its final output it will give you a new command to execute. The command will involve make, and at the same time setting the PETSc local directory. In my case the command is

   * ``make PETSC_DIR=/home/stepanfomichev/local/petsc PETSC_ARCH=arch-linux-c-debug all``

#. The previous make command will again in its final output give another command to execute -- this time a make check command. In my case

   * ``make PETSC_DIR=/home/stepanfomichev/local/petsc PETSC_ARCH=arch-linux-c-debug check``

#. Finally, to be able to use the installed petsc4py, one needs to set the environment variable PYTHONPATH to point to the location where petsc4py was installed, which will be in its local directory (this is again returned in the final output of make check). In my case

   * ``export PYTHONPATH=$HOME/local/petsc/arch-linux-c-debug/lib``

After these steps, PETSc and the ParallelSparseExecutor class are ready to be used.

Windows installation
""""""""""""""""""""

.. warning::

   We cannot guarantee that Windows installations will work correctly. All that follows is highly experimental.

For a Windows installation, we recommend using the Windows Subsystem for Linux
(WSL2 as of Apr 2021). This provides the easiest interface for a Linux shell with minimal system overhead (compared to e.g. a virtual machine). For most users, downloading and making compilers will likely be required. This drastically simplifies the process of working with compilers, specifying environment variables, and enables one to rely on the vast community of practice that exists around the Linux OS. Moreover, many of conda installers are only available on Linux, so using WSL simplifies the python package managing process. We describe the entire process below for Ubuntu 20.04 on Windows.

(A very similar process can be followed for a personal \*nix machine, minus the WSL instructions.)


#. First, follow the Microsoft instructions to enable the Windows Subsystem for Linux feature and install the \*nix distribution of your choice (we used Ubuntu 20.04). There are many websites detailing the installation process: we found `this Medium post <https://medium.com/using-valgrind-on-windows-in-clion-with-wsl/install-windows-subsystem-for-linux-windows-10-3ea33c535625>`_ to be a convenient reference. The instructions at the beginning of the link refer specifically to WSL.
#. Again begin by creating the ``conda`` environment following the instructions above.
#. You will likely not have an MPI library installed. There are a number of choices, including ``openmpi`` and ``mpich``. We used ``mpich``\ , downloaded and compiled according to the instructions on the `ABINIT website <https://docs.abinit.org/tutorial/compilation/#installing-mpi>`_\ , specifically the "Installing MPI" section.
#. With an MPI library in place, point the ``MPICC`` environment variable to the MPI executables (to find the path on \*nix, call for example ``which mpiexec``\ ) and use pip to install ``mpi4py`` (see the first section above).
#. It is now time to install PETSc. Unlike the server where we must compile from source, here we can simply use pip. In order to still be able to pass various flags to PETSc's configure, we define a special environment variable by

   * ``export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack"``
   * this sets the PETSc data type to complex, as well as downloads a particular parallel sparse solver (MUMPS) with its dependencies

#. Once the flags are passed, install PETSc and its Python binding petsc4py by ``pip install petsc petsc4py``.

After these steps, PETSc and the ParallelSparseExecutor class are ready to be used.
