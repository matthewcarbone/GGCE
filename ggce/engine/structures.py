#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

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
    def __init__(self, M, N, t, eta, a, Omega):
        self.M = M
        self.N = N
        self.t = t
        self.eta = eta
        self.a = a
        self.Omega = Omega

        # Checks on M
        assert(isinstance(self.M, int))
        assert(self.M > 0)

        # Checks on N
        assert(isinstance(self.N, int))
        assert(self.N > 0)

        # Assert the coupling
        assert(self.t >= 0.0)

        # Assert auxiliary parameters
        assert(self.eta > 0.0)
        assert(self.a > 0.0)


class InputParameters:
    """Container for the full set of input parameters for the trial.

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
    Omega_lambda_dict : dict
        Dictionary representing the boson frequencies and coupling terms for
        each model contribution in the Hamiltonian. This also encodes the
        models themselves. Should be of the form, e.g.,
        {'H': [0.1, 0.2], 'SSH': [0.3, 0.4]}
    """

    AVAIL_MODELS = ['H', 'EFB', 'SSH']

    def __init__(self, M, N, eta, t, Omega_lambda_dict, a=1.0):
        self.Omega_lambda_dict = Omega_lambda_dict
        self.terms = None
        self.model_params = ModelParams(
            M, N, t, eta, a,
            [Omega for _, (Omega, _) in Omega_lambda_dict.items()]
        )
        self.n_boson_types = len(self.model_params.Omega)

    def init_terms(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multipled by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        terms = []

        boson_type = 0
        for model, (Omega, lam) in self.Omega_lambda_dict.items():
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

        # We don't save the terms since those are reconstructed upon
        # instantiation
        config = self.get_params()

        with open(path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
