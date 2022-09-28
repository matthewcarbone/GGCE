#!/usr/bin/env python3

import numpy as np

import sys

sys.path.append("../..")

from ggce import Model, System, DenseSolver  # noqa
from ggce.executors.petsc4py.parallel import MassSolverMUMPS
from ggce.logger import logger
import matplotlib.pyplot as plt

from mpi4py import MPI

if __name__ == "__main__":

    root_dir = f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"\
               f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expansion/"\
               f"fixing_h_pl_p_bug/scripts/petsc_test"

    k = 0.5 * np.pi
    w = np.linspace(-2, 5, 8)

    COMM = MPI.COMM_WORLD

    # Check the true T=0 case
    model = Model.from_parameters(hopping=1., temperature=0.0)
    model.add_("Holstein", 0.5, 2, 2, dimensionless_coupling_strength=1.)
    model.add_("Peierls", 1., 2, 2, dimensionless_coupling_strength=1.)
    solver = MassSolverMUMPS(system=System(model), mpi_comm = COMM, \
                                    brigade_size = 2)
    results = solver.spectrum(k, w, eta=0.05, pbar=False)
    if COMM.Get_rank() == 0:
        results = results.squeeze()
        np.savetxt("T0.txt", np.array([w, -results.imag / np.pi]).T)
        plt.plot(w, -results.imag / np.pi, label = f"Tzero")

    # Check the true T=epsilon case
    model = Model.from_parameters(hopping=1., temperature=1e-6)
    model.add_(
        "Holstein",
        0.5,
        2,
        2,
        phonon_extent_tfd=1,
        phonon_number_tfd=1,
        dimensionless_coupling_strength=1.,
    )
    model.add_(
        "Peierls",
        1.,
        2,
        2,
        phonon_extent_tfd=1,
        phonon_number_tfd=1,
        dimensionless_coupling_strength=1.,
    )
    solver = MassSolverMUMPS(system=System(model), mpi_comm = COMM, \
                                                    brigade_size = 2)
    results = solver.spectrum(k, w, eta=0.05, pbar=False)

    if COMM.Get_rank() == 0:
        results = results.squeeze()
        np.savetxt("Tepsilon.txt", np.array([w, -results.imag / np.pi]).T)
        plt.plot(w, -results.imag / np.pi, label = "Tepsilon")

        plt.legend()

        plt.show()
