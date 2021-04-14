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
from ggce.executors import finalize_lowest_band_executor


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
            trg = self.paths['results'] / Path(idx)
            fname = trg / res
            if fname.exists():
                self.results[idx] = pickle.load(open(fname, 'rb'))
                continue

            # Try to finalize from state
            done_file = self.paths['results'] / Path(idx) / Path("DONE")
            state_dir = self.paths['results'] / Path(idx) / Path("STATE")
            if len(list(state_dir.iterdir())) == 0:
                to_drop.append(idx)
                continue

            r = finalize_lowest_band_executor(state_dir, trg, done_file)
            self.results[idx] = r

        self.master = self.master.T.drop(columns=to_drop).T

        self._set_defaults()

    def ground_state(self, truncate_val=None, lowest_only=False, **kwargs):
        """Returns the ground state dispersion computed as the lowest
        energy peak energy as a function of k. Also returns the polaron
        weight."""

        result = self.__call__(**kwargs)
        w_gs = np.array(result[2])
        weight = np.array(result[3])
        k = self.k_grid[:len(weight)]
        assert len(weight) == len(w_gs)

        if lowest_only:
            kstar = np.argmin(w_gs)
            return k[kstar], w_gs[kstar], weight[kstar]

        if truncate_val is not None:
            try:
                kstar = np.where(weight < truncate_val)[0][0]
            except IndexError:
                return k, w_gs, weight
            return k[:kstar], w_gs[:kstar], weight[:kstar]

        return k, w_gs, weight

    def effective_mass(self, npts=15, **kwargs):
        """Returns the effective mass and the k-location of the ground state.

        [description]

        Parameters
        ----------
        **kwargs : {[type]}
            [description]
        npts : {number}, optional
            [description] (the default is 15, which [default_description])
        """

        result = self.__call__(**kwargs)
        w_gs = np.array(result[2])
        k = self.k_grid[:len(w_gs)]
        kstar = np.argmin(w_gs)
        lpts = kstar - min(kstar, npts // 2)
        rpts = kstar + npts // 2

        # Note that the k-grid is in units of pi.
        p0 = np.polyfit(k[lpts:rpts] * np.pi, w_gs[lpts:rpts], deg=2)
        return (1.0 / p0[0] / 2.0, k[kstar])


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