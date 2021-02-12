#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import namedtuple
import copy
import itertools
import numpy as np
import math

import yaml


# Define a namedtuple which contains the shift indexes, x and y, the dagger
# status, d, the coupling term, g, and the boson frequency and type (index)
SingleTerm = namedtuple("SingleTerm", ["x", "y", "d", "g", "bv", "bt"])


def model_coupling_map(coupling_type, t, Omega, lam):
    if coupling_type == 'H':  # Holstein
        return math.sqrt(2.0 * t * Omega * lam)
    elif coupling_type == 'EFB':  # EFB convention lam = g for convenience
        return lam
    elif coupling_type == 'SSH':  # SSH
        return math.sqrt(t * Omega * lam / 2.0)
    elif coupling_type == 'bondSSH':  # bond SSH (note this is a guess)
        return math.sqrt(t * Omega * lam)
    else:
        raise RuntimeError(f"Unknown coupling_type type {coupling_type}")


class ModelParams:
    """Parameter permutations are dictionaries containing two values: val,
    which is the list of values, and cycle, which determines how the
    permutations of this parameter interact with the others.
        Specifically, cycle can be one of three values:
    * 'solo': a single value (not a list) which is applied to all
        calculations
    * 'zip': all permutations with this flag will be iterated over using the
        zip iterator, so all with this flag must have the same length
    * 'prod': all product-wise combinations of these flags will be run
        and therefore they do not require the same length.
    """

    @staticmethod
    def _assert_parameter(param_name, data, model_len):
        """Sanity checks the input parameters for any issues."""

        # These parameters require a list of lists (for prod or zip), or
        # simply a list (for solo)
        if param_name in ['M_extent', 'N_bosons', 'Omega', 'lam']:
            assert isinstance(data['vals'], list)

            if data['cycle'] == 'solo':
                assert len(data['vals']) == model_len

            elif data['cycle'] in ['zip', 'prod']:
                assert all(isinstance(xx, list) for xx in data['vals'])
                assert all(len(xx) == model_len for xx in data['vals'])

            elif data['cycle'] == 'prod-linspace':
                assert param_name in ['Omega', 'lam']
                assert len(data['vals']) == model_len
                assert all(len(xx) == 3 for xx in data['vals'])
                assert all(
                    data['vals'][ii][2] == data['vals'][ii + 1][2]
                    for ii in range(model_len - 1)
                )

            else:
                raise RuntimeError(f"Unknown cycle: {data['cycle']}")

        elif param_name in [
            'hopping', 'broadening', 'absolute_extent', 'max_bosons_per_site'
        ]:

            if data['cycle'] == 'solo':
                cond1 = isinstance(data['vals'], float)
                cond2 = isinstance(data['vals'], int)
                assert cond1 or cond2

            elif data['cycle'] in ['zip', 'prod']:
                assert isinstance(data['vals'], list)

            else:
                raise RuntimeError(f"Unknown cycle: {data['cycle']}")

    def __init__(self, model, info, params):
        self.model = model
        self.info = info
        self.solo = dict()
        self.prod = dict()
        self.zip = dict()
        zip_lens = []

        for param_name, data in params.items():
            ModelParams._assert_parameter(param_name, data, len(self.model))

            if data['cycle'] == 'solo':
                self.solo[param_name] = data['vals']
            elif data['cycle'] == 'zip':
                zip_lens.append(len(data['vals']))
                self.zip[param_name] = data['vals']
            elif data['cycle'] == 'prod':
                self.prod[param_name] = data['vals']
            elif data['cycle'] == 'prod-linspace':
                dat = np.array([np.linspace(*xx) for xx in data['vals']]).T
                dat = np.round(dat, 3)
                self.prod[param_name] = [
                    [float(yy) for yy in xx] for xx in dat
                ]
            else:
                raise RuntimeError(f"Unknown cycle {data['cycle']}")

        # Assert that all lists in zip have the same length
        assert all([
            zip_lens[ii] == zip_lens[ii + 1]
            for ii in range(len(zip_lens) - 1)
        ])

        try:
            zip_indexes = [ii for ii in range(zip_lens[0])]
        except IndexError:
            zip_indexes = [None]

        # Define a product over the zip_indexes product and single terms
        self.master = list(itertools.product(
            zip_indexes, *list(self.prod.values())
        ))
        self._counter = 0
        self._counter_max = len(self.master)

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._counter_max:
            raise StopIteration

        current_parameters = self.master[self._counter]

        # Deal with the solo parameters
        d = copy.deepcopy(self.solo)
        d['model'] = self.model
        d['info'] = self.info

        # The zipped parameters
        for key, value in self.zip.items():
            if current_parameters[0] is not None:
                d[key] = value[current_parameters[0]]

        # And the product parameters
        ii = 1
        for key in list(self.prod.keys()):
            d[key] = current_parameters[ii]
            ii += 1

        self._counter += 1
        return d


class GridParams:
    """Contains the grid information for running the loops outside of the
    System class. I.e., for every set of ModelParameters, the System is
    initialized only once, and the solver is utilized for each w-k point
    combination."""

    def __init__(self, grid_params):
        self.k_grid_info = grid_params['k']
        self.w_grid_info = grid_params['w']

    def get_grid(self, grid_type, round_values=8):
        """"""

        assert grid_type in ['k', 'w']

        if grid_type == 'k':
            vals = self.k_grid_info['vals']
            linspace = self.k_grid_info['linspace']
        elif grid_type == 'w':
            vals = self.w_grid_info['vals']
            linspace = self.w_grid_info['linspace']

        assert isinstance(linspace, bool)

        if linspace:
            assert all([isinstance(xx, list) for xx in vals])
            assert all([len(xx) == 3 for xx in vals])

            return np.round(np.sort(np.concatenate([
                np.linspace(*c, endpoint=True) for c in vals
            ])), round_values)

        else:
            assert isinstance(vals, list)
            return np.round(vals, round_values)

    def save(self, path):
        """Writes the dictionary to disk."""

        d = {
            "k": self.k_grid_info,
            "w": self.w_grid_info
        }

        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False)


def parse_inp(inp_path):
    """Parses the user-generated input yaml file and returns the ModelParams
    and GridParams classes."""

    p = yaml.safe_load(open(inp_path, 'r'))
    mp = ModelParams(p['model'], p['info'], p['model_parameters'])
    gp = GridParams(p['grid_parameters'])
    return mp, gp


class SystemParams:

    def __init__(self, d):

        self.M = d['M_extent']
        self.N = d['N_bosons']
        self.t = d['hopping']
        self.eta = d['broadening']
        self.a = 1.0  # Hard code lattice constant
        self.Omega = d['Omega']
        self.lambdas = d['lam']
        self.models = d['model']
        self.n_boson_types = len(self.models)

        absolute_extent = d.get('absolute_extent')
        if self.n_boson_types == 1:
            assert absolute_extent is None
            self.absolute_extent = self.M[0]
        else:
            assert absolute_extent is not None
            assert absolute_extent > 0
            self.absolute_extent = absolute_extent

        self.max_bosons_per_site = d.get('max_bosons_per_site')
        if self.max_bosons_per_site is not None:
            assert self.max_bosons_per_site > 0

            # This constraint is required for consistency
            assert self.N == self.max_bosons_per_site * self.n_boson_types

    def prime(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multipled by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        self.terms = []

        bt = 0
        for (m, o, lam) in zip(self.models, self.Omega, self.lambdas):
            g = model_coupling_map(m, self.t, o, lam)

            if m == 'H':
                self.terms.extend([
                    SingleTerm(x=0, y=0, d='+', g=-g, bv=o, bt=bt),
                    SingleTerm(x=0, y=0, d='-', g=-g, bv=o, bt=bt)
                ])
            elif m == 'EFB':
                self.terms.extend([
                    SingleTerm(x=1, y=1, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=-1, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=1, y=0, d='-', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=0, d='-', g=g, bv=o, bt=bt)
                ])
            elif m == 'bondSSH':
                self.terms.extend([
                    SingleTerm(x=1, y=0.5, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=1, y=0.5, d='-', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=-0.5, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=-0.5, d='-', g=g, bv=o, bt=bt)
                ])
            elif m == 'SSH':
                self.terms.extend([
                    SingleTerm(x=1, y=0, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=1, y=0, d='-', g=g, bv=o, bt=bt),
                    SingleTerm(x=1, y=1, d='+', g=-g, bv=o, bt=bt),
                    SingleTerm(x=1, y=1, d='-', g=-g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=-1, d='+', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=-1, d='-', g=g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=0, d='+', g=-g, bv=o, bt=bt),
                    SingleTerm(x=-1, y=0, d='-', g=-g, bv=o, bt=bt)
                ])
            else:
                raise RuntimeError("Unknown model type when setting terms")
            bt += 1
