#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import yaml

from ggce.engine.structures import GridParams


class BaseResults:

    def __init__(self, package_path, res):

        # Load in the initial data
        package_path = Path(package_path)

        self.paths = {
            'results': package_path / Path("results"),
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

    def _set_defaults(self):
        """Set the default key values for convenience."""

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

    def __call__(self, **kwargs):
        queried_table = self._query(**kwargs)
        if len(queried_table.index) != 1:
            raise RuntimeError("Queried table has != 1 row")
        return self.results[list(queried_table.index)[0]]


class LowestEnergyBandResults(BaseResults):

    def __init__(self, package_path, res=Path("res_gs.pkl")):

        super().__init__(package_path, res)

        to_drop = []

        self.results = dict()
        for idx in list(self.master.index):
            try:
                self.results[idx] = pickle.load(
                    open(self.paths['results'] / Path(idx) / res, 'rb')
                )
            except FileNotFoundError:
                to_drop.append(idx)
                continue

        self.master = self.master.T.drop(columns=to_drop).T

        self._set_defaults()

    def ground_state(self, **kwargs):
        """Returns the ground state dispersion computed as the lowest
        energy peak energy as a function of k. Also returns the polaron
        weight."""

        result = self.__call__(**kwargs)
        return self.k_grid, np.array(result[2]), np.array(result[3])


class Results(BaseResults):
    """Trials are single spectra A(w) for some fixed k, and all other
    parameters. This class is a helper for querying trials based on the
    parameters specified, and returning spectral functions A(w)."""

    def __init__(self, package_path, res=Path("res.npy")):

        super().__init__(package_path, res)

        # A list of indexes to drop; these contain no data
        to_drop = []

        for idx in list(self.master.index):
            self.results[idx] = dict()
            try:
                dat = np.load(
                    open(self.paths['results'] / Path(idx) / res, 'rb')
                )
            except FileNotFoundError:
                to_drop.append(idx)
                continue
            for k_val in self.k_grid:
                where = np.where(np.abs(dat[:, 0] - k_val) < 1e-7)[0]
                loaded = dat[where, 1:]
                sorted_indices = np.argsort(loaded[:, 0])
                self.results[idx][k_val] = loaded[sorted_indices, :]

        self.master = self.master.T.drop(columns=to_drop).T

        self._set_defaults()

    def spectrum(self, k, **kwargs):
        """Returns the spectrum for a specified k value."""

        result = self.__call__(**kwargs)
        G = result[k]  # Query will throw a KeyError if k is not found
        return G[:, 0], -G[:, 2] / np.pi

    def band(self, **kwargs):
        """Returns the band structure for the provided run parameters."""

        band = []
        for k in self.k_grid:
            _, A = self.spectrum(k, **kwargs)
            band.append(A)
        return self.w_grid, self.k_grid, np.array(band)
