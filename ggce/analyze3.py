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
        k_grid = self.inp.get_k_grid()

        # Create a mapping based on the k-grid values.
        self.k_map = dict()
        for k in k_grid:
            where = np.where(results_matrix[:, 0] == k)[0]
            res = results_matrix[where, 1:]
            sorted_indices = np.argsort(res[:, 0])
            self.k_map[k] = res[sorted_indices, :]

    def __call__(self, k):
        return self.k_map[k]

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


class Results:

    def append_master(self, params):
        """Appends the master dictionary, which keeps track of the parameters
        contained in this class."""

        for key, value in params.items():
            self._vals[key].append(value)

    def determine_defaults(self):
        """Iterates through the master dictionary and sets defaults for the
        lists which only have one parameter."""

        self.defaults = dict()
        for key, value in self._vals.items():
            if len(np.unique(value)) == 1:
                self.defaults[key] = value[0]
            else:
                self.defaults[key] = None

    @staticmethod
    def key_str(AE, M, N, t, Omega, lam, model):
        if isinstance(N, int):
            N = [N]
        if isinstance(M, int):
            M = [M]
        if isinstance(Omega, float):
            Omega = [Omega]
        if isinstance(lam, float):
            lam = [lam]
        if isinstance(model, str):
            model = [model]
        st = f"{AE}__{M}__{N}__{t:.08f}__{str(Omega)}__"
        st += f"{str(lam)}__{str(model)}"
        return st

    def __init__(self, package_path):

        self.path = package_path
        self.master = dict()
        self._vals = {
            'AE': [], 'M': [], 'N': [], 't': [], 'Omega': [],
            'lam': [], 'model': []
        }

        config_path = os.path.join(self.path, "configs")
        config_directories = utils.listdir_fullpath(config_path)

        # Nesting begins with the configs
        for config_dir in config_directories:
            base_config_name = os.path.basename(config_dir)
            trial = Trial(self.path, base_config_name)
            params = trial.get_params()
            self.append_master(params)
            self.master[Results.key_str(**params)] = trial

        self.determine_defaults()

    def info(self):
        for key, value in self._vals.items():
            print(key, "\t", value)

    def __call__(self, **kwargs):

        query = dict()
        for key, value in self.defaults.items():
            v = kwargs.get(key)
            if v is None:
                v = value
            query[key] = v

        return self.master[Results.key_str(**query)]
