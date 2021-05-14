#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import namedtuple
import numpy as np
import math
import time

from ggce.utils.logger import Logger


def model_coupling_map(coupling_type, t, Omega, lam, ignore):
    """Returns the value for g, the scalar that multiplies the coupling in the
    Hamiltonian. Converts the user-input lambda value to this g.

    Parameters
    ----------
    coupling_type : {'H', 'SSH', 'bondSSH', 'EFB'}
        The desired coupling type. Can be Holstein, SSH (Peierls), bond SSH
        or Edwards Fermion Boson (EFB).
    t : float
        The hopping strength.
    Omega : float
        The (Einsten) boson frequency.
    lam : float
        The dimensionless coupling.
    ignore : bool
        If True, simply returns the value for lambda as g. Useful in some
        situations where the user wants to tune g directly.

    Returns
    -------
    float
        The value for the coupling (g).

    Raises
    ------
    RuntimeError
        If an unknown coupling type is provided.
    """

    if ignore:
        return lam

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


# Define a namedtuple which contains the shift indexes, x and y, the dagger
# status, d, the coupling term, g, and the boson frequency and type (index)
SingleTerm = namedtuple("SingleTerm", ["x", "y", "d", "g", "bt"])

# Define another namedtuple which contains only the terms that the f-functions
# need to calculate their prefactors, thus saving space.
fFunctionInfo = namedtuple("fFunctionInfo", ["a", "t", "Omega"])


class Model:

    def __init__(
        self, name="model", default_console_logging_level="info", log_file=None
    ):
        self._logger = Logger(log_file, mpi_rank=self.mpi_rank)
        self._logger.adjust_logging_level(default_console_logging_level)

        # Index uninitialized parameters
        self.t = None
        self.a = None
        self.M = None
        self.N = None
        self.M_tfd = None
        self.N_tfd = None
        self.temperature = None
        self.dimension = None

    def set_parameters(self, hopping=1.0, dimension=1, lattice_constant=1.0):
        """Initializes the core, model-independent parameters of the
        simulation. Note this also sets the temperature to 0 by default. Use
        set_temperature to actually change the temperature to something
        non-zero.

        Parameters
        ----------
        hopping : float, optional
            The nearest-neighbor hopping term (the default is 1.0).
        dimension : int, optional
            The dimensionality of the system (the default is 1).
        lattice_constant : float, optional
            The lattice constant (the default is 1.0).
        """

        # List all of the parameters necessary for the run
        self.t = hopping
        self.dimension = dimension
        self.temperature = 0.0
        self.a = lattice_constant

    def set_finite_temperature(self, temperature, M, N):
        """[summary]

        [description]

        Parameters
        ----------
        temperature : {[type]}
            [description]
        M : {[type]}
            [description]
        N : {[type]}
            [description]
        """

        if temperature == 0.0:
            self._logger.warning(
                "You have attempted to set thermo-field dynamics temperature "
                "to zero explicitly. This only bloats the calculation and "
                "does not change any results. Doing nothing."
            )
            return

        if temperature < 0.0:
            self._logger.error("Temperature must be non-zero")
            return

        if M_tfd or

        self.temperature = temperature
        self.M_tfd = M
        self.N_tfd = N

    def get_fFunctionInfo(self):
        return fFunctionInfo(a=self.a, t=self.t, Omega=self.Omega)

    def _extend_terms(self, model_type, g, bt):
        """Helper method to extent the self.terms list.

        This method contains the 'programmed' notation of the coupling terms.
        Every model must have a corresponding string matching the cases below.

        Parameters
        ----------
        model_type : {'H', 'EFB', 'bondSSH', 'SSH'}
            The model type.
        g : float
            The coupling term (multiplying V in the Hamiltonian).
        bt : int
            The boson type index. Indexes the place in the model list. For
            example, if the current boson type is 1 and the model is
            ['H', 'SSH'], then the boson corresponds to an SSH phonon.

        Raises
        ------
        RuntimeError
            If the model type is unknown.
        """

        if model_type == 'H':
            self.terms.extend([
                SingleTerm(x=0, y=0, d='+', g=-g, bt=bt),
                SingleTerm(x=0, y=0, d='-', g=-g, bt=bt)
            ])
        elif model_type == 'EFB':
            self.terms.extend([
                SingleTerm(x=1, y=1, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=0, d='-', g=g, bt=bt)
            ])
        elif model_type == 'bondSSH':
            self.terms.extend([
                SingleTerm(x=1, y=0.5, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0.5, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='-', g=g, bt=bt)
            ])
        elif model_type == 'SSH':
            self.terms.extend([
                SingleTerm(x=1, y=0, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
                SingleTerm(x=1, y=1, d='+', g=-g, bt=bt),
                SingleTerm(x=1, y=1, d='-', g=-g, bt=bt),
                SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-1, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=0, d='+', g=-g, bt=bt),
                SingleTerm(x=-1, y=0, d='-', g=-g, bt=bt)
            ])
        else:
            raise RuntimeError("Unknown model type when setting terms")

    def _get_coupling_prefactors(self, Omega):
        """Get's the TFD coupling prefactors.

        The TFD prefactors are defined clearly in e.g. JCP 145, 224101 (2016).

        Parameters
        ----------
        Omega : float
            The (Einstein) phonon frequency.

        Returns
        -------
        float, float
            The modifying prefactor to the real and fictitious couplings.
        """

        if self.temperature > 0.0:
            beta = 1.0 / self.temperature
            theta_beta = np.arctanh(np.exp(-beta * Omega / 2.0))
            V_prefactor = np.cosh(theta_beta)
            V_tilde_prefactor = np.sinh(theta_beta)
            return V_prefactor, V_tilde_prefactor
        else:
            return 1.0, None

    def _adjust_bosons_if_necessary(self):
        """Adjusts all attributes according to e.g. TFD.

        Note that this method essentially does nothing if T=0.
        """

        # Adjust the number of boson types according to thermofield
        if self.temperature > 0.0:
            self.n_boson_types *= 2  # Thermo field "double"
            assert isinstance(self.M, list)
            assert isinstance(self.N, list)
            assert isinstance(self.Omega, list)
            assert isinstance(self.lambdas, list)
            assert isinstance(self.models, list)

            new_M = []
            new_N = []
            new_Omega = []
            new_lambdas = []
            new_models = []

            for ii in range(len(self.models)):
                new_M.extend([
                    self.M[ii], self.M[ii]
                    if self.M_tfd is None else self.M_tfd[ii]
                ])
                new_N.extend([
                    self.N[ii], self.N[ii]
                    if self.N_tfd is None else self.N_tfd[ii]
                ])

                # Need the negative Omega here to account for the TFD truly.
                # the term's value for Omega is never actually called. Here, we
                # note that the boson frequency is NEGATIVE, indicative of the
                # fictitious space!
                new_Omega.extend([self.Omega[ii], -self.Omega[ii]])
                new_lambdas.extend([self.lambdas[ii], self.lambdas[ii]])
                new_models.extend([self.models[ii], self.models[ii]])

            self.M = new_M
            self.N = new_N
            self.Omega = new_Omega

            # Some of these parameters aren't used but we'll redfine them
            # anyway for consistency. Some of this is actually used in logging
            # so it's still useful.
            self.lambdas = new_lambdas
            self.models = new_models
            self.models_vis = []
            for ii, m in enumerate(self.models):
                if ii % 2 == 0:  # Even
                    self.models_vis.append(m)
                else:
                    self.models_vis.append(f"~{m}")
        else:
            self.models_vis = self.models

    def prime(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multiplied by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        t0 = time.time()

        self.terms = []

        bt = 0

        for (m, bigOmega, lam) in zip(self.models, self.Omega, self.lambdas):
            g = model_coupling_map(m, self.t, bigOmega, lam, self.use_g)

            # Handle the TFD stuff if necessary
            V_prefactor, V_tilde_prefactor = \
                self._get_coupling_prefactors(bigOmega)

            self._extend_terms(m, g*V_prefactor, bt)
            bt += 1

            # Now we implement the thermo field double changes to the
            # coupling prefactor, if necessary.
            if self.temperature > 0.0:
                self._extend_terms(m, g*V_tilde_prefactor, bt)
                bt += 1

        self._adjust_bosons_if_necessary()

        dt = time.time() - t0
        self._logger.info(
            "Parameters object primed; ready for compute", elapsed=dt
        )
