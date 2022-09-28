#!/usr/bin/env python3

import numpy as np

import sys

sys.path.append("..")

from ggce import Model, System, DenseSolver  # noqa
import matplotlib.pyplot as plt

if __name__ == "__main__":

    k = 0.5 * np.pi
    w = np.linspace(-2, 5, 90)

    # Check the true T=0 case
    model = Model.from_parameters(hopping=1., temperature=0.0)
    model.add_("Holstein", 0.5, 2, 2, dimensionless_coupling_strength=1.)
    model.add_("Peierls", 1., 2, 2, dimensionless_coupling_strength=1.)
    solver = DenseSolver(System(model))
    results = solver.spectrum(k, w, eta=0.05, pbar=False).squeeze()
    np.savetxt("T0.txt", np.array([w, -results.imag / np.pi]).T)
    plt.plot(w, -results.imag / np.pi, label = f"Tzero")

    # Check the true T=epsilon case
    # model = Model.from_parameters(hopping=1., temperature=1e-6)
    # model.add_(
    #     "Holstein",
    #     0.5,
    #     2,
    #     2,
    #     phonon_extent_tfd=1,
    #     phonon_number_tfd=1,
    #     dimensionless_coupling_strength=1.,
    # )
    # model.add_(
    #     "Peierls",
    #     1.,
    #     2,
    #     2,
    #     phonon_extent_tfd=1,
    #     phonon_number_tfd=1,
    #     dimensionless_coupling_strength=1.,
    # )
    # solver = DenseSolver(System(model))
    # results = solver.spectrum(k, w, eta=0.005, pbar=False).squeeze()
    # np.savetxt("Tepsilon.txt", np.array([w, -results.imag / np.pi]).T)
    #
    # plt.plot(w, -results.imag / np.pi, label = "Tepsilon")

    plt.show()
