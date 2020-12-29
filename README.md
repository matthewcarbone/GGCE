
# GGCE
Generalized Green's function Cluster Expansion

## Necessary packages
We use `conda` for all package management except for `mpi4py`. The following process should generally explain how to install the `GGCE` repository and its dependencies in a way that should work on most clusters.

1. [Create a fresh `conda` environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using `python=3.7`.
2. Make sure you have the correct MPI module loaded. Usually this is done via some `module load`, where generally the package is something like `openmpi/gcc/64/3.1.1` (this example is the one on the Columbia Habanero cluster that I use).
3. Point the `MPICC` path to the executables corresponding to the loaded modules. I do this with `export MPICC=$(which mpicc)`, where you'll want to check via `echo $MPICC` that this is indeed the correct path.
4. Install `mpi4py` using `pip`, _not_ `conda`. I have found that `conda` ignores the cluster executables in favor of it's own locally installed versions, and then when submitting jobs, the compute nodes will not detect the correct executables and all processes will correspond to rank 0.
5. Install `numpy` and `scipy` via `conda`. You'll also want to check that this linked successfully to e.g. `mkl`. 
    * `conda install -c anaconda numpy`
    * `conda install -c anaconda scipy`

After these steps, everything should work properly.
