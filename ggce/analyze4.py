#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ggce.engine.structures import GridParams
from ggce.utils import utils


def lorentzian(x, x0, a, gam):
    return np.abs(a) * gam**2 / (gam**2 + (x - x0)**2)


class Results:
    """Trials are single spectra A(w) for some fixed k, and all other
    parameters. This class is a helper for querying trials based on the
    parameters specified, and returning spectral functions A(w)."""

    def __init__(self, package_path, res=Path("res.npy")):

        # Load in the initial data
        package_path = Path(package_path)

        self.paths = {
            'results': package_path / Path("results"),
            'bash_script': package_path / Path("submit.sbatch"),
            'configs': package_path / Path("configs"),
            'grids': package_path / Path("grids.yaml")
        }

        # Load in the configurations and set a mapping between the parameters
        # and results
        self.master = pd.DataFrame({
            str(f.stem): yaml.safe_load(open(f, 'r'))
            for f in self.paths['configs'].iterdir()
        }).T.astype(str)
        self.master.drop(columns=['info', 'model'], inplace=True)

        # Load in the grids
        gp = GridParams(yaml.safe_load(open(self.paths['grids'], 'r')))
        self.w_grid = gp.get_grid('w')
        self.k_grid = gp.get_grid('k')

        # Load in the results
        self.results = dict()

        for idx in list(self.master.index):
            self.results[idx] = dict()
            dat = np.load(open(self.paths['results'] / Path(idx) / res, 'rb'))
            for k_val in self.k_grid:
                where = np.where(np.abs(dat[:, 0] - k_val) < 1e-7)[0]
                loaded = dat[where, 1:]
                sorted_indices = np.argsort(loaded[:, 0])
                self.results[idx][k_val] = loaded[sorted_indices, :]

        # Set the default key values for convenience
        self.defaults = dict()
        for col in list(self.master.columns):
            unique = np.unique(self.master[col])
            self.defaults[col] = None
            if len(unique) == 1:
                self.defaults[col] = unique[0]

    def _query(self, **kwargs):
        """Returns the rows of the dataframe corresponding to the kwargs
        specified."""

        prio = {1: kwargs, 2: self.defaults}
        d = {**prio[2], **prio[1]}
        query_base_list = [f"{key} == '{value}'" for key, value in d.items()]
        query_base = " and ".join(query_base_list)
        return self.master.query(query_base)

    def spectrum(self, k, **kwargs):
        """Returns the spectrum for a specified k value."""

        queried_table = self._query(**kwargs)
        if len(queried_table.index) > 1:
            raise RuntimeError("Queried table has more than one row")

        result = self.results[list(queried_table.index)[0]]

        try:
            G = result[k]
        except KeyError:
            raise RuntimeError("Queried k value does not exist")

        return G[:, 0], -G[:, 2] / np.pi
