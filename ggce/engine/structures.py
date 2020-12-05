#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

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

    def __init__(
        self, absolute_extent, M, N, t, eta, a, Omega, lambdas, model
    ):
        self.absolute_extent = absolute_extent
        self.M = M
        self.N = N
        self.t = t
        self.eta = eta
        self.a = a
        self.Omega = Omega
        self.lambdas = lambdas
        self.model = model

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
    """Container for the full set of input parameters for the trial.

    model_dict : dict
        Dictionary representing the boson frequencies and coupling terms for
        each model contribution in the Hamiltonian. This also encodes the
        models themselves. It also contains the maximum cloud extent and max
        number of bosons per model. Should be of the form, e.g.,
        {'H': [0.1, 0.2, 2, 5], 'SSH': [0.3, 0.4, 3, 10]}
    """

    @staticmethod
    def _get_n_boson_types(d):
        """Get's the total number of boson types and asserts that the lists
        are properly used."""

        assert len(d['model']) == len(d['Omega']) \
            == len(d['lam']) == len(d['M_extent']) \
            == len(d['N_bosons'])

        return len(d['model'])

    def _from_dict(self, d):
        """Initializes the parameters from a yaml config. Note these serve as
        default parameters and are overridden by command line arguments."""

        self.n_boson_types = InputParameters._get_n_boson_types(d)

        if self.n_boson_types == 1:
            absolute_extent = d.get('M_extent')[0]
        else:
            absolute_extent = d.get('absolute_extent')

        self.w_grid_info = d.get('w_grid_info')
        self.k_grid_info = d.get('k_grid_info')
        self.linspacek = d.get('linspacek')

        # Get the number of boson types
        # absoute_extent, M, N, t, eta, a, Omega, lambdas, model
        self.model_params = ModelParams(
            absolute_extent=absolute_extent,
            M=d.get("M_extent"),
            N=d.get("N_bosons"),
            t=d.get("t"),
            eta=d.get("eta"),
            Omega=d.get("Omega"),
            lambdas=d.get("lam"),
            a=1.0,
            model=d.get("model")
        )

    def get_w_wgrid(self):
        """"""

        return list(np.sort(np.concatenate([
            np.linspace(*c, endpoint=True)
            for c in self.w_grid_info
        ])))

    def get_k_grid(self):
        return list(np.linspace(*self.k_grid_info, endpoint=True)) \
            if list(self.linspacek) else self.k_grid_info

    def init_attributes(self, params_from_yaml, command_line_args):
        for key, value in command_line_args.items():
            if command_line_args[key] is not None:
                params_from_yaml[key] = value
        self._from_dict(params_from_yaml)
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

        terms = []

        boson_type = 0
        Omegas = self.model_params.Omega
        lambdas = self.model_params.lambdas
        models = self.model_params.model
        for (model, Omega, lam) in zip(models, Omegas, lambdas):
            g = model_coupling_map(model, self.model_params.t, Omega, lam)

            if model == 'H':
                terms.extend([
                    SingleTerm(
                        x=0, y=0, dagger='+', g=-g, boson_type=boson_type
                    ),
                    SingleTerm(
                        x=0, y=0, dagger='-', g=-g, boson_type=boson_type
                    )
                ])
            elif model == 'EFB':
                terms.extend([
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
                terms.extend([
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

        self.terms = terms

    def get_params(self):
        return {
            key: value for key, value in vars(self).items() if key != 'terms'
        }

    def save_config(self, path):
        """Saves the config to the disk. These are the precise parameters
        needed to reconstruct the input parameters in its entirety."""

        assert self.terms is None
        with open(path, 'w') as outfile:
            yaml.dump(self.get_params(), outfile, default_flow_style=False)
