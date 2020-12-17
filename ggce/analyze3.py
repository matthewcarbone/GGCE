#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import copy
import os

from itertools import product
import numpy as np
import pickle
import yaml

from scipy.signal import find_peaks
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from ggce.utils import utils
from ggce.utils.logger import default_logger as dlog
from ggce.engine.structures import InputParameters


class Trial:
    """Trials are single spectra A(w) for some fixed k, and all other
    parameters. This class is a helper for querying trials based on the
    parameters specified, and returning spectral functions A(w)."""

    def __init__(self, package_path, config_fname, res="res.txt"):

        # Load in the initial data
        c_dir_name = os.path.splitext(config_fname)[0]
        trial_directory = os.path.join(package_path, "results", c_dir_name)
        results_matrix = np.loadtxt(os.path.join(trial_directory, res))
        state_dir = os.path.join(trial_directory, "STATE")
        if os.path.isdir(state_dir):
            if not os.path.exists(os.path.join(state_dir, "DONE")):
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


class Results:

    def __init__(self, package_path):

        self.path = package_path
        self.master = dict()

        config_path = os.path.join(self.path, "configs")
        config_directories = utils.listdir_fullpath(config_path)

        # Nesting begins with the configs
        for config_dir in config_directories:
            base_config_name = os.path.basename(config_dir)
            self.master[base_config_name] = Trial(self.path, base_config_name)

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
