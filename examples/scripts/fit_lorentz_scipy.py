import numpy as np
from scipy.optimize import curve_fit

# uncomment if want to visualize PETSc speed benchmark
# import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import shutil
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
ggce_head_dir = (
    f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"
    f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expan"
    f"sion/new_lorentzfit"
)
data_dir = (
    f"/mnt/c/Users/bobst/Documents/University_of_British_Columbia/"
    f"Physics/Mona_Berciu/Generalized_Green_function_cluster_expan"
    f"sion/fin_T_hol"
)
sys.path.append(ggce_head_dir)
import ggce

# load the mpi communicator
from mpi4py import MPI

# load the Model and Executor classes to store parameters and manage the calculation
from ggce.model import Model  # noqa: E402
from ggce.executors.petsc4py.parallel import ParallelSparseExecutorMUMPS
from ggce.executors.serial import SerialDenseExecutor
from ggce.utils.utils import lorentzian, peak_location_and_weight_scipy

server = f"lisa"
Mval = 3
Nval = 9
Mval2 = 3
Nval2 = 5
lam, om, t = 1, 0.5, sys.argv[1]
eta = 0.05
date = sys.argv[2]
aux = f"polaron_dispersion"
year = f"22"
krange = [
    "0.000",
    "0.008",
    "0.017",
    "0.025",
    "0.033",
    "0.041",
    "0.050",
    "0.058",
    "0.066",
    "0.074",
    "0.083",
    "0.091",
    "0.099",
    "0.107",
    "0.116",
    "0.124",
    "0.132",
    "0.141",
    "0.149",
    "0.157",
]

results_dir = os.path.join(
    data_dir,
    f"lam_{lam}",
    f"om_{om}",
    f"t_{t}",
    aux,
    f"{Mval}{Nval}_{Mval2}{Nval2}_{server}_{date}_{year}",
    f"pol_dis_HfT_{Mval}{Nval}_{Mval2}{Nval2}",
)

energy_k = np.zeros(len(krange))
energy_k_err = np.zeros(len(krange))
eta_k = np.zeros(len(krange))

fig0, ax0 = plt.subplots()
for ii, k in enumerate(krange):
    result_file = os.path.join(results_dir, f"all_results_k_{k}.txt")
    wrange, Arange = np.loadtxt(result_file, unpack=True, skiprows=1)

    # restrict wrange to last 10 points near peak
    wrange = wrange[-10:]
    Arange = Arange[-10:]

    fit_params, error = peak_location_and_weight_scipy(wrange, Arange, eta)

    wrange_pred = np.linspace(wrange[0], wrange[-1], 100)
    Arange_pred = lorentzian(wrange_pred, *fit_params)

    if ii in [0, 10, 19]:
        ax0.plot(wrange_pred, Arange_pred)
        ax0.scatter(wrange, Arange)

    energy_k[ii] = fit_params[0]
    energy_k_err[ii] = error[0, 0]
    eta_k[ii] = fit_params[2]

# show together the plots of the lorentz fits
plt.show()

result_file = os.path.join(results_dir, f"polaron_dispersion.txt")
kgrid, energy_k_2point = np.loadtxt(result_file, unpack=True, skiprows=1)


def quadratic_band(k, m_star, offset):

    return k**2 / (2.0 * m_star) - offset


""" DOES ANYTHING CHANGE WHEN I FIT TO DOUBLE THE RANGE (SYMMETRIZE)?
NO, IT DOES NOT -- TINY EFFECTS IN THE 2-4 DECIMAL, NON-CRUCIAL """
# kgrid = np.array(list(kgrid)[::-1] + list(kgrid)[1:])
# energy_k = np.array( list(energy_k)[::-1] + list(energy_k)[1:] )
# energy_k_2point = np.array( list(energy_k_2point)[::-1] + list(energy_k_2point)[1:] )
# energy_k_err = np.array( list(energy_k_err)[::-1] + list(energy_k_err)[1:] )

best_fit_pars, covmat = curve_fit(
    quadratic_band, kgrid, energy_k, p0=[0.5, energy_k[0]]
)
best_fit_pars_2point, covmat_2point = curve_fit(
    quadratic_band, kgrid, energy_k_2point, p0=[0.5, energy_k_2point[0]]
)
dense_kgrid = np.linspace(0, kgrid[-1], 100)

""" IF YOU NEED TO SAVE THE SCIPY FITS (FOR EXAMPLE FOR PLOTTING PRETTIER E_p(K) BAND TRACKING) """
xx = np.array([kgrid, energy_k]).T
save_loc = os.path.join(results_dir, f"polaron_dispersion_scipy.txt")
np.savetxt(
    save_loc,
    xx,
    header=f"momentum        E_P(k) with m_star / m_0 = {best_fit_pars[0] / 0.5041061511616247} (m_0 = 0.5041061511616247)",
    delimiter="    ",
)
exit()
fig, ax = plt.subplots()
ax.plot(
    dense_kgrid,
    quadratic_band(dense_kgrid, *best_fit_pars),
    label=f"$k^2$ fit",
)

ax.plot(kgrid, energy_k_2point, label="two-point")
# plt.plot(krange_floats, all_locs, label = "scipy")
ax.errorbar(kgrid, energy_k, yerr=energy_k_err, label="scipy")
fig.legend()
plt.show()

print(
    f"scipy fit: m_star / m_0 = {best_fit_pars[0] / 0.5041061511616247} (m_0 = 0.5041061511616247)"
)
print(
    f"two-p fit: m_star / m_0 = {best_fit_pars_2point[0] / 0.5041061511616247} (m_0 = 0.5041061511616247)"
)
# # only collect and output results on the head node
# if COMM.Get_rank() == 0:
#
#     # plot and save the Lorentz fit for diagnostic purposes
#     for ii in range(len(kgrid)):
#         fig, ax = plt.subplots(1, 1)
#         loc = results[ii]["ground_state"]
#         scale = results[ii]["weight"]
#         ax.plot(results[ii]["w"], lorentzian(results[ii]["w"], loc, scale, eta), label = "Lorentz fit")
#         ax.scatter(results[ii]["w"], results[ii]["A"], label = "Calculation")
#         ax.set_ylabel(r"$A(k,w)$", fontsize=20)
#         ax.set_xlabel(r"$w$", fontsize=20)
#         ax.tick_params(which = 'both', labelsize = 16)
#         ax.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(results_dir,f'lorentzfit_k_{kgrid[ii]:.3f}.png'),\
#                         format='png')
#         plt.close(fig)
#
#     # process the solver output to extract time to solve and spectrum vals
#     fig, ax = plt.subplots(1, 1)
#     ax.scatter(kgrid, energy_k, label = 'PETSc')
#     ax.set_ylabel(r"$E_P(k)$", fontsize=20)
#     ax.set_xlabel(r"$k/\pi$", fontsize=20)
#     ax.tick_params(which = "both", labelsize = 16)
#
#     def quadratic_band(k, m_star, offset):
#
#         return k**2 / (2.*m_star) - offset
#
#     best_fit_pars, covmat = curve_fit(quadratic_band, kgrid, energy_k, \
#                                                     p0 = [0.5, energy_k[0]])
#     dense_kgrid = np.linspace(0, kgrid[-1], 100)
#     ax.plot( dense_kgrid, quadratic_band(dense_kgrid, *best_fit_pars), \
#                                                         label = f"$k^2$ fit")
#
#     # also output the energy data to disk for posterity / postprocess
#     xx = np.array([kgrid, energy_k]).T
#     np.savetxt(os.path.join(results_dir,f"polaron_dispersion.txt"), xx,\
#                 header = f"momentum        E_P(k) with m_star / m_0 = {best_fit_pars[0] / 0.5041061511616247} (m_0 = 0.5041061511616247)",\
#                 delimiter = '    ')
#
#     plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize=16)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(os.path.join(results_dir,f'polaron_dispersion.png'), format='png', bbox_inches='tight')
#
#     ## also save all data just in case
#     for ii, kval in enumerate(kgrid):
#         xx = np.array([results[ii]['w'], results[ii]['A']]).T
#         np.savetxt(os.path.join(results_dir,f"all_results_k_{kval:.2f}.txt"), xx,\
#                     header = f"omega            spectral function        for momentum = {kval}",\
#                     delimiter = '    ')
