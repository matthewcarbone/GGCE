#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import math

import yaml


class ModelParams:

    def __init__(self, M, N, t, Omega, eta, a, g):
        self.M = M
        self.N = N
        self.t = t
        self.Omega = Omega
        self.eta = eta
        self.a = a
        self.g = g


class SingleTerm:
    """Container for a single term in the Hamiltonian. Also carries the
    parameters needed for the computation, such as the couplings."""

    def __init__(self, x, y, sign, dagger, model_params, alpha_idx=0):
        self.x = x
        self.y = y
        self.sign = sign
        self.dagger = dagger
        self.model_params = model_params
        self.alpha_idx = alpha_idx


class InputParameters:
    """Container for the full set of input parameters for the trial.

    Attributes
    ----------
    M : int
        Maximal cloud extent. Must be > 0.
    N : int
        Maximum number of bosons allowed in the cloud. Must be > 0.
    model : {'H', 'EFB', 'SSH'}
        The model type.
    t : float
        Hopping term.
    Omega: float
        Boson Einstein frequency.
    eta : float
        Broadening term.
    a : float
        Lattice parameter. Default is 1.0. Recommended not to change this.
    config_filter : {0, 1}
        Determines the filter/rule for boson clouds. 0 for no rule, 1 for
        A gaussian filter. Default is 0.
    lambd : float, optional
        If provided, represents the effective coupling which is
        model-dependent. Default is None. Note that the user should only
        provide lambd or g, not both, else it will raise a RuntimeError.
    k : float, optional
        The momentum value for the calculation.
    """

    AVAIL_MODELS = ['H', 'EFB', 'SSH']
    AVAIL_CONFIG_FILTERS = ['no_filter', 'gaussian']

    def __init__(
        self, M, N, eta, model, t, Omega, lambd, a=1.0,
        config_filter='no_filter', k=None
    ):

        # If specified, set k
        if k is not None:
            assert k >= 0.0
            self.k = k

        # Checks on M
        self.M = M
        assert(isinstance(self.M, int))
        assert(self.M > 0)

        # Checks on N
        self.N = N
        assert(isinstance(self.N, int))
        assert(self.N > 0)

        # Assert a valid model
        self.model = model
        assert(self.model in InputParameters.AVAIL_MODELS)

        # Assert the coupling
        self.t = t
        assert(self.t >= 0.0)

        # Assert the boson frequency
        self.Omega = Omega
        assert(self.Omega >= 0.0)

        # Assert auxiliary parameters
        self.eta = eta
        self.a = a
        assert(self.eta > 0.0)
        assert(self.a > 0.0)
        self.config_filter = config_filter
        assert(self.config_filter in InputParameters.AVAIL_CONFIG_FILTERS)

        self.lambd = lambd
        self.terms = None

    def _set_g(self):

        # Else, we need to compute what g actually is based on the model and
        # the provided lambda
        # H: lambd=g^2/2*t*Omega => g = sqrt(2*t*Omega*lambd)
        assert(self.lambd >= 0.0)
        if self.model == 'H':  # Holstein
            self.g = math.sqrt(2.0 * self.t * self.Omega * self.lambd)
        elif self.model == 'EFB':  # EFB convention lambd = g for convenience
            self.g = self.lambd
        elif self.model == 'SSH':  # SSH
            self.g = math.sqrt(self.t * self.Omega * self.lambd / 2.0)
        else:
            raise RuntimeError(f"Unknown model type {self.model}")

    def init_terms(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multipled by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        self._set_g()

        mp = ModelParams(
            self.M, self.N, self.t, self.Omega, self.eta, self.a, self.g
        )

        if self.model == 'H':
            self.terms = [
                SingleTerm(x=0, y=0, sign=-1.0, dagger='+', model_params=mp),
                SingleTerm(x=0, y=0, sign=-1.0, dagger='-', model_params=mp)
            ]
        elif self.model == 'EFB':
            self.terms = [
                SingleTerm(x=1, y=1, sign=1.0, dagger='+', model_params=mp),
                SingleTerm(x=-1, y=-1, sign=1.0, dagger='+', model_params=mp),
                SingleTerm(x=1, y=0, sign=1.0, dagger='-', model_params=mp),
                SingleTerm(x=-1, y=0, sign=1.0, dagger='-', model_params=mp)
            ]
        elif self.model == 'SSH':
            self.terms = [
                SingleTerm(x=1, y=0, sign=1.0, dagger='+', model_params=mp),
                SingleTerm(x=1, y=0, sign=1.0, dagger='-', model_params=mp),
                SingleTerm(x=1, y=1, sign=-1.0, dagger='+', model_params=mp),
                SingleTerm(x=1, y=1, sign=-1.0, dagger='-', model_params=mp),
                SingleTerm(x=-1, y=-1, sign=1.0, dagger='+', model_params=mp),
                SingleTerm(x=-1, y=-1, sign=1.0, dagger='-', model_params=mp),
                SingleTerm(x=-1, y=0, sign=-1.0, dagger='+', model_params=mp),
                SingleTerm(x=-1, y=0, sign=-1.0, dagger='-', model_params=mp)
            ]
        else:
            raise RuntimeError("Unknown model type when setting terms")

    def save_config(self, path):
        """Saves the config to the disk. These are the precise parameters
        needed to reconstruct the input parameters in its entirety."""

        assert self.terms is None

        # We don't save the terms since those are reconstructed upon
        # instantiation
        config = {
            key: value for key, value in vars(self).items() if key != 'terms'
        }

        with open(path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
