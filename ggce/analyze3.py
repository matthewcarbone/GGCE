#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os

import numpy as np
import yaml

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ggce.utils import utils
from ggce.utils.logger import default_logger as dlog
from ggce.engine.structures import InputParameters


def lorentzian(x, x0, a, gam):
    return np.abs(a) * gam**2 / (gam**2 + (x - x0)**2)


class Trial:
    """Trials are single spectra A(w) for some fixed k, and all other
    parameters. This class is a helper for querying trials based on the
    parameters specified, and returning spectral functions A(w)."""

    def __init__(self, package_path, config_fname, res="res.npy"):

        # Load in the initial data
        c_dir_name = os.path.splitext(config_fname)[0]
        trial_directory = os.path.join(package_path, "results", c_dir_name)
        path = os.path.join(trial_directory, res)
        try:
            results_matrix = np.loadtxt(path)
        except UnicodeDecodeError:
            results_matrix = np.load(open(path, 'rb'))
        if not os.path.exists(os.path.join(trial_directory, "DONE")):
            if res != "res.txt":
                msg = "Trial is not confirmed finished: check STATE"
                dlog.warning(msg)

        # Parse the results matrix
        config_path = os.path.join(package_path, "configs", config_fname)
        self.inp = InputParameters(yaml.safe_load(open(config_path)))
        self.inp.init_attributes()
        self.k_grid = self.inp.get_k_grid()

        # Create a mapping based on the k-grid values.
        self.k_map = dict()
        for k in self.k_grid:
            where = np.where(np.abs(results_matrix[:, 0] - k) < 1e-8)[0]
            res = results_matrix[where, 1:]
            sorted_indices = np.argsort(res[:, 0])
            self.k_map[f"{k:.08f}"] = res[sorted_indices, :]

    def __call__(self, k):
        return self.k_map[f"{k:.08f}"]

    def get_params(self):
        """Returns the parameters corresponding to this config."""

        model_params = self.inp.model_params

        return {
            'AE': model_params.absolute_extent,
            'M': model_params.M,
            'N': model_params.N,
            't': model_params.t,
            'Omega': model_params.Omega,
            'lam': model_params.lambdas,
            'model': model_params.model
        }

    def info(self):
        for key, value in self.get_params().items():
            print(key, "\t", value)

        if len(self.k_grid) > 5:
            print(f"kpts: {self.k_grid[:5]}, ...")
        else:
            print(f"kpts: {self.k_grid}")

    def k_v_w_no_interp(self):
        band = []
        for k in self.k_grid:
            A = self(k)
            band.append(A[:, 2])
        Z = -np.array(band) / np.pi
        return A[:, 0], self.k_grid, Z

    def ground_state_dispersion(self, lorentzian_fit=None, offset=5):
        """Returns the ground state dispersion computed as the lowest
        energy peak energy as a function of k.

        Parameters
        ----------
        lorentzian_fit : tuple
            Whether or not to attempt to fit the ground state peak to a
            Lorentzian before finding the location of the state.
        """

        energies = []
        for k in self.k_grid:
            G = self(k)
            w = G[:, 0]
            A = -G[:, 2] / np.pi
            argmax = find_peaks(A)[0][0]
            w_loc = w[argmax]
            if lorentzian_fit:
                eta = self.inp.model_params.eta
                popt, _ = curve_fit(
                    lorentzian, w[argmax-offset:argmax+offset],
                    A[argmax-offset:argmax+offset], p0=[w_loc, A[argmax], eta]
                )
                w_loc = popt[0]
            energies.append(w_loc)

        return self.k_grid, energies


class Results:

    def __init__(self, package_path, res="res.npy"):

        self.path = package_path
        self.master = dict()

        config_path = os.path.join(self.path, "configs")
        config_directories = utils.listdir_fullpath(config_path)

        # Nesting begins with the configs
        for config_dir in config_directories:
            base_config_name = os.path.basename(config_dir)
            self.master[base_config_name] = Trial(
                self.path, base_config_name, res=res
            )

    def __call__(self, config_name):
        return self.master[config_name]

    def info(self):
        for key, value in self.master.items():
            print(f"{key}: {value.get_params()}")

    def query(self, M, N, lam, Omega, AE=None, k=None, spectra=False):
        """Attempts to find matching configs via searches for configs that
        match values for M, N, lambda and Omega, where AE == M if unspecified.
        """

        if not isinstance(M, list):
            M = [M]
        if not isinstance(N, list):
            N = [N]
        if not isinstance(lam, list):
            lam = [lam]
        if not isinstance(Omega, list):
            Omega = [Omega]

        if AE is None:
            AE = max(M)

        found = []
        for key, value in self.master.items():
            p = value.get_params()
            if (
                p['AE'] == AE and p['M'] == M and p['N'] == N and
                p['lam'] == lam and p['Omega'] == Omega
            ):
                found.append(value)

        if k is not None:
            for trial in found:
                try:
                    G = trial(k)
                    if spectra:
                        return G[:, 0], -G[:, 2] / np.pi
                    else:
                        return trial
                except KeyError:
                    continue

        if len(found) == 1:
            return found[0]
        return found
