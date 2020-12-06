#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import copy
import itertools
import numpy as np
import math

import yaml


class SingleTerm:
    """Container for a single term in the Hamiltonian. Also carries the
    parameters needed for the computation, such as the couplings."""

    def __init__(self, x, y, dagger, g, boson_type):
        self.x = x
        self.y = y
        self.dagger = dagger
        self.boson_type = boson_type
        self.g = g


def model_coupling_map(coupling_type, t, Omega, lam):
    if coupling_type == 'H':  # Holstein
        g = math.sqrt(2.0 * t * Omega * lam)
    elif coupling_type == 'EFB':  # EFB convention lam = g for convenience
        g = lam
    elif coupling_type == 'SSH':  # SSH
        g = math.sqrt(t * Omega * lam / 2.0)
    else:
        raise RuntimeError(f"Unknown coupling_type type {coupling_type}")
    return g


class ModelParams:
    """

    Attributes
    ----------
    M : int
        Maximal cloud extent. Must be > 0.
    N : int
        Maximum number of bosons allowed in the cloud. Must be > 0.
    t : float
        Hopping term.
    eta : float
        Broadening term.
    a : float
        Lattice parameter. Default is 1.0. Recommended not to change this.
    """

    def _get_n_boson_types(self):
        """Get's the total number of boson types and asserts that the lists
        are properly used."""

        assert len(self.model) == len(self.Omega) == len(self.lambdas) \
            == len(self.M) == len(self.N)

        return len(self.model)

    def __init__(
        self, absolute_extent, M, N, t, eta, a, Omega, lambdas, model
    ):
        self.M = M
        self.N = N
        self.t = t
        self.eta = eta
        self.a = a
        self.Omega = Omega
        self.lambdas = lambdas
        self.model = model
        self.n_boson_types = self._get_n_boson_types()

        if self.n_boson_types == 1:
            assert absolute_extent is None
            self.absolute_extent = self.M[0]
        else:
            assert absolute_extent is not None
            self.absolute_extent = absolute_extent

    def assert_attributes(self):
        assert isinstance(self.absolute_extent, int)
        assert isinstance(self.M, list)
        assert all([M > 0 for M in self.M])
        assert all([isinstance(M, int) for M in self.M])
        assert isinstance(self.N, list)
        assert all([N > 0 for N in self.N])
        assert all([isinstance(N, int) for N in self.N])
        assert self.t >= 0.0
        assert self.eta > 0.0
        assert self.a > 0.0
        assert isinstance(self.Omega, list)
        assert all([omega >= 0.0 for omega in self.Omega])
        assert isinstance(self.lambdas, list)
        assert all([ll >= 0.0 for ll in self.lambdas])
        assert isinstance(self.model, list)
        assert all([isinstance(mod, str) for mod in self.model])


class InputParameters:
    """Container for the full set of input parameters for the trial. The
    Input parameters themselves are read either from a single config file,
    command line arguments during the prime step, or a combination of both.

    Attributes
    ----------
    model_params : ModelParams
        Contains the information, along with the terms attribute, for
        initializing the System class.
    w_grid_info, k_grid_info : list
        A list, or list of list in the case of w_grid_info, representing the
        linspace parameters for constructing the frequency and k-space grids,
        respectively. The k_grid_info must take the form of either
            1) a list of k-points in units of pi (if linspacek is False)
            2) a list of 3 elements representing the linspace parameters
               (otherwise)
        The w_grid_info must take the form of a list of lists, where each
        list is 3 elements long representing linspace parameters, and the
        corresponding created grids are stitched together. This allows the
        user to make grids which are non-linear in w-space.
    terms : list
        A list of SingleTerm objects. This corresponds directly with the
        coupling term of the Hamiltonian.
    """

    AVAILABLE_KEYS = [
        't', 'eta', 'model', 'Omega', 'lam', 'M_extent', 'N_bosons',
        'w_grid_info', 'k_grid_info', 'linspacek'
    ]

    def __init__(self, d1, d2=dict()):
        self._params = d1  # d1 should never contain incorrect keys
        self.terms = None
        for key, value in d2.items():
            if key in InputParameters.AVAILABLE_KEYS:
                if d2[key] is not None:
                    self._params[key] = value
        self.N_M_permutations = list(itertools.product(
            self._params['M_extent'], self._params['N_bosons']
        ))
        self.counter_max = len(self.N_M_permutations)

    def _from_dict(self, d):
        """Initializes the parameters from a yaml config. Note these serve as
        default parameters and are overridden by command line arguments."""

        self.w_grid_info = d.get('w_grid_info')
        self.k_grid_info = d.get('k_grid_info')
        self.linspacek = d.get('linspacek')

        # Get the number of boson types
        # absoute_extent, M, N, t, eta, a, Omega, lambdas, model
        self.model_params = ModelParams(
            absolute_extent=d.get("absolute_extent"),
            M=d.get("M_extent"),
            N=d.get("N_bosons"),
            t=d.get("t"),
            eta=d.get("eta"),
            Omega=d.get("Omega"),
            lambdas=d.get("lam"),
            a=1.0,
            model=d.get("model")
        )

    def get_w_grid(self):
        """"""

        return list(np.sort(np.concatenate([
            np.linspace(*c, endpoint=True)
            for c in self.w_grid_info
        ])))

    def get_k_grid(self):
        return list(np.linspace(*self.k_grid_info, endpoint=True)) \
            if self.linspacek else self.k_grid_info

    def init_attributes(self):
        self._from_dict(self._params)
        self.model_params.assert_attributes()
        assert isinstance(self.w_grid_info, list)
        assert isinstance(self.k_grid_info, list)
        assert isinstance(self.linspacek, bool)

    def init_terms(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multipled by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        self.terms = []

        boson_type = 0
        Omegas = self.model_params.Omega
        lambdas = self.model_params.lambdas
        models = self.model_params.model
        for (model, Omega, lam) in zip(models, Omegas, lambdas):
            g = model_coupling_map(model, self.model_params.t, Omega, lam)

            if model == 'H':
                self.terms.extend([
                    SingleTerm(
                        x=0, y=0, dagger='+', g=-g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=0, y=0, dagger='-', g=-g, boson_type=boson_type
                    )
                ])
            elif model == 'EFB':
                self.terms.extend([
                    SingleTerm(
                        x=1, y=1, dagger='+', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=-1, dagger='+', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=1, y=0, dagger='-', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=0, dagger='-', g=g, boson_type=boson_type
                    )
                ])
            elif model == 'SSH':
                self.terms.extend([
                    SingleTerm(
                        x=1, y=0, dagger='+', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=1, y=0, dagger='-', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=1, y=1, dagger='+', g=-g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=1, y=1, dagger='-', g=-g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=-1, dagger='+', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=-1, dagger='-', g=g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=0, dagger='+', g=-g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=-1, y=0, dagger='-', g=-g, boson_type=boson_type
                    )
                ])
            else:
                raise RuntimeError("Unknown model type when setting terms")
            boson_type += 1

    def prime(self):
        self.init_attributes()
        self.init_terms()

    def save(self, path):
        """Saves the config to the disk. These are the precise parameters
        needed to reconstruct the input parameters in its entirety."""

        with open(path, 'w') as outfile:
            yaml.dump(self._params, outfile, default_flow_style=False)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= self.counter_max:
            raise StopIteration
        d = copy.deepcopy(self._params)
        M_N_permutation = self.N_M_permutations[self.counter]
        d['M_extent'] = M_N_permutation[0]
        d['N_bosons'] = M_N_permutation[1]
        input_params = InputParameters(d)
        self.counter += 1
        return input_params
