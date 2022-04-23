---
title: GGCE Installation
---

The following documents how one should install the GGCE package. Note that currently only support for Linux and Mac OS is supported, we cannot guarantee that the package works correctly on Windows operating systems.

# Standard installation
**Once the package is released on PyPI**, You can install GGCE via pip:

```bash
pip install ggce
```

# Standard development installation
If you wish to help develop GGCE, you can install a local version via

```bash
pip install e ".[dev]"
```

which will also install all the optional development requirements in `requirements-dev.txt`. We generally recommend using a virtual environment via e.g. `conda` to ensure reproducibility and that there are no package conflicts.


# Advanced installation

The GGCE code can be installed with advanced capabilities. There are two particular options: MPI and PETSc (which requires MPI). The following documents the installation procedure for each of these.

## Necessary packages (MPI installation)
We use `conda` for all package management except for `mpi4py`. The following process should generally explain how to install the `GGCE` repository and its dependencies in a way that should work on most clusters.

1. [Create a fresh `conda` environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using `python=3.7`.
2. Make sure you have the correct MPI module loaded. Usually this is done via some `module load`, where generally the package is something like `openmpi/gcc/64/3.1.1` (this example is the one on the Columbia Habanero cluster that I use).
3. Point the `MPICC` path to the executables corresponding to the loaded modules. I do this with `export MPICC=$(which mpicc)`, where you'll want to check via `echo $MPICC` that this is indeed the correct path.
4. Install `mpi4py` using `pip`, _not_ `conda`. I have found that `conda` ignores the cluster executables in favor of it's own locally installed versions, and then when submitting jobs, the compute nodes will not detect the correct executables and all processes will correspond to rank 0.
5. Install `numpy` and `scipy` via `conda`. You'll also want to check that this linked successfully to e.g. `mkl` (you can use `numpy.show_config()`).
    * `conda install -c anaconda numpy`
    * `conda install -c anaconda scipy`

After these steps, everything should work properly.

## Enable usage of PETSc for massively parallel computations
[PETSc](https://www.mcs.anl.gov/petsc/index.html) is a software library created and maintained at the Argonne National Laboratory for
massively parallel solution of large linear systems of equations (e.g.
finite-difference approaches to PDEs). It provides an interface to a long list
of linear system solution methods (MUMPS, PARDISO, Krylov), preconditioners, and data
structures that allow rapid optimization of the solution method for the problem at hand.

Since PETSc relies on MPI for multicore parallelization, the same caveats about
using the conda package management apply. Below are installation instructions
for a typical Linux cluster (I used the LISA cluster at University of British
Columbia's Quantum Matter Institute).

1. Again, make sure to be in your freshly installed conda environment, with the
correct MPI packages and their corresponding compilers loaded. Either MPICH or
OPENMPI should work: currently tested with MPICH and gcc compilers.
2. Install `cython` via `conda`. This is needed for the PETSc C-oriented Python
bindings.
3. [Download PETSc source](https://www.mcs.anl.gov/petsc/download/index.html) (git clone recommended).
4. Unzip and navigate to the directory.
5. Run ./configure with the following flags (see the [PETSc list of common usage](https://www.mcs.anl.gov/petsc/documentation/installation.html#exampleusage) for a complete list of possible flags).
  * `--with-batch` -- is needed on cluster systems which only allow job batch submissions.
  * `--with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90` -- this lets PETSc know to use the same compilers as were used in the creation of your loaded MPI module.
  * ` --download-fblaslapack --download-mumps --download-scalapack` -- download any external solvers that are desired and their dependencies (for a complete list see the [PETSc manual, sec. 2.3.7](https://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf))
  * `--with-scalar-type=complex` -- by default PETSc is compiled for real number vectors and matrices, this switches it to complex data types.
  * ` --download-petsc4py` -- this downloads the Python bindings petsc4py, allowing to access PETSc routines and data structures easily from within Python
6. Once the commands executes, in its final output it will give you a new command to execute. The command will involve make, and at the same time setting the PETSc local directory. In my case the command is
  * `make PETSC_DIR=/home/stepanfomichev/local/petsc PETSC_ARCH=arch-linux-c-debug all`
7. The previous make command will again in its final output give another command to execute -- this time a make check command. In my case
  * `make PETSC_DIR=/home/stepanfomichev/local/petsc PETSC_ARCH=arch-linux-c-debug check`
8. Finally, to be able to use the installed petsc4py, one needs to set the environment variable PYTHONPATH to point to the location where petsc4py was installed, which will be in its local directory (this is again returned in the final output of make check). In my case
  * `export PYTHONPATH=$HOME/local/petsc/arch-linux-c-debug/lib`

After these steps, PETSc and the ParallelSparseExecutor class are ready to be used.

For a Windows installation, we recommend using the Windows Subsystem for Linux
(WSL2 as of Apr 2021). This provides the easiest interface for a Linux shell with minimal system overhead (compared to e.g. a virtual machine). For most users, downloading and making compilers will likely be required. This drastically simplifies the process of working with compilers, specifying environment variables, and enables one to rely on the vast community of practice that exists around the Linux OS. Moreover, many of conda installers are only available on Linux, so using WSL simplifies the python package managing process. We describe the entire process below for Ubuntu 20.04 on Windows.

(A very similar process can be followed for a personal *nix machine, minus the WSL instructions.)

1. First, follow the Microsoft instructions to enable the Windows Subsystem for Linux feature and install the *nix distribution of your choice (we used Ubuntu 20.04). There are many websites detailing the installation process: we found [this Medium post](https://medium.com/using-valgrind-on-windows-in-clion-with-wsl/install-windows-subsystem-for-linux-windows-10-3ea33c535625) to be a convenient reference. The instructions at the beginning of the link refer specifically to WSL.
2. Again begin by creating the `conda` environment following the instructions above.
3. You will likely not have an MPI library installed. There are a number of choices, including `openmpi` and `mpich`. We used `mpich`, downloaded and compiled according to the instructions on the [ABINIT website](https://docs.abinit.org/tutorial/compilation/#installing-mpi), specifically the "Installing MPI" section.
4. With an MPI library in place, point the `MPICC` environment variable to the MPI executables (to find the path on *nix, call for example `which mpiexec`) and use pip to install `mpi4py` (see the first section above).
5. It is now time to install PETSc. Unlike the server where we must compile from source, here we can simply use pip. In order to still be able to pass various flags to PETSc's configure, we define a special environment variable by
  * `export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack"`
  * this sets the PETSc data type to complex, as well as downloads a particular parallel sparse solver (MUMPS) with its dependencies
6. Once the flags are passed, install PETSc and its Python binding petsc4py by `pip install petsc petsc4py`.

After these steps, PETSc and the ParallelSparseExecutor class are ready to be used.
