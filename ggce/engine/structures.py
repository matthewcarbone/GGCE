#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import namedtuple
import numpy as np
import math

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
fFunctionInfo = namedtuple("fFunctionInfo", ["a", "t", "Omega", "eta"])


class ParameterObject:

    def _set_coupling(self, d):
        """Handle the coupling, which can be defined as either lambda (the
        dimensionless coupling) or the coupling itself (g). In the latter case,
        use_g will be set to True so as to ignore any dimensionless coupling
        conversion.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        self.lambdas = d.get('dimensionless_coupling')
        self.use_g = False
        if self.lambdas is None:
            self.lambdas = d['coupling']
            self.use_g = True

    def _set_temperature(self, d):
        """Handle setting the temperature. The temperature in the original
        input file must not be defined for zero-temperature calculations, and
        must be a float > 0.0 for the finite-temperature calculations.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol in ["tfd", "trace"]:
            self.temperature = d['temperature']
            assert self.temperature is not None
            assert self.temperature > 0.0
        else:
            assert d.get("temperature") is None
            self.temperature = 0.0

    def _set_extra_boson_clouds(self, d):
        """Handles setting extra boson cloud information depending on the
        protocol.

        Notes
        -----
        The number of bosons is actually an optional quantity in the parameter
        file due to the option of setting a hard boson constraint
        (max_bosons_per_site). These possibilities are handled in a later
        assertion.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol not in ["tfd", "trace"]:
            assert d.get("M_tfd") is None
            assert d.get("N_tfd") is None
            assert d.get("M_trace") is None
            assert d.get("N_trace") is None
        elif self.protocol == "tfd":
            self.M_tfd = d["M_tfd"]
            self.N_tfd = d.get("N_tfd")
            assert d.get("M_trace") is None
            assert d.get("N_trace") is None
        elif self.protocol == "trace":
            assert d.get("M_tfd") is None
            assert d.get("N_tfd") is None
            self.M_trace = d["M_trace"]
            self.N_trace = d.get("N_trace")

    def _set_absolute_extent_information(self, d):
        """Sets the absolute extent and runs assertions on it.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol not in ["tfd", "trace"]:
            if self.n_boson_types > 1:
                self.absolute_extent = d["absolute_extent"]
                assert self.absolute_extent >= np.max(self.M)
            else:
                assert d.get("absolute_extent") is None
                self.absolute_extent = self.M[0]
        else:
            self.absolute_extent = d["absolute_extent"]
            assert self.absolute_extent >= np.max(self.M)
            if self.protocol == "tfd":
                assert self.absolute_extent >= np.max(self.M_tfd)
            elif self.protocol == "trace":
                assert self.absolute_extent >= np.max(self.M_trace)
        assert self.absolute_extent > 0

    def _set_max_bosons_per_site(self, d):
        """Handles the maximum bosons per site assertions (hard bosons).

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        self.max_bosons_per_site = d.get('max_bosons_per_site')
        if self.max_bosons_per_site is not None:
            assert self.max_bosons_per_site > 0
            assert isinstance(self.max_bosons_per_site, int)
            assert self.N is None
            self.N = [
                self.max_bosons_per_site * self.n_boson_types * m
                for m in self.M
            ]

            if self.protocol == "tfd":
                assert self.N_tfd is None
                self.N_tfd = [
                    self.max_bosons_per_site * self.n_boson_types * m
                    for m in self.M_tfd
                ]
            elif self.protocol == "trace":
                assert self.N_trace is None
                self.N_trace = [
                    self.max_bosons_per_site * self.n_boson_types * m
                    for m in self.M_trace
                ]
        else:
            assert self.N is not None
            if self.protocol == "tfd":
                assert self.N_tfd is not None
            elif self.protocol == "trace":
                assert self.N_trace is not None

    def __init__(self, d, logger=Logger(dummy=True)):

        self.logger = logger

        # Start with parameters that are required for all trials
        self.M = d['M_extent']
        self.N = d.get('N_bosons')
        self.t = d['hopping']
        self.eta = d['broadening']
        self.a = d.get("lattice_constant", 1.0)
        self.Omega = d['Omega']
        self.protocol = d['protocol']

        # Handle the coupling
        self._set_coupling(d)

        # Handle temperature
        self._set_temperature(d)

        # Handle extra boson clouds due to TFD or other finite-T methods
        self._set_extra_boson_clouds(d)

        # Set the model
        self.models = d['model']
        self.n_boson_types = len(self.models)
        assert self.n_boson_types == len(self.M)

        # Handle the absolute extent information
        self._set_absolute_extent_information(d)

        # Handle the hard boson constraints
        self._set_max_bosons_per_site(d)

        self.logger.info("Parameters object initialized successfully")

    def get_fFunctionInfo(self):
        return fFunctionInfo(
            a=self.a, t=self.t, Omega=self.Omega, eta=self.eta
        )

    def _extend_terms(self, m, g, bt):
        """Helper method to extent the self.terms list.

        This method contains the 'programmed' notation of the coupling terms.
        Every model must have a corresponding string matching the cases below.

        Parameters
        ----------
        m : {'H', 'EFB', 'bondSSH', 'SSH'}
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

        if m == 'H':
            self.terms.extend([
                SingleTerm(x=0, y=0, d='+', g=-g, bt=bt),
                SingleTerm(x=0, y=0, d='-', g=-g, bt=bt)
            ])
        elif m == 'EFB':
            self.terms.extend([
                SingleTerm(x=1, y=1, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=0, d='-', g=g, bt=bt)
            ])
        elif m == 'bondSSH':
            self.terms.extend([
                SingleTerm(x=1, y=0.5, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0.5, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='-', g=g, bt=bt)
            ])
        elif m == 'SSH':
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

        if self.protocol == "tfd":
            assert self.temperature > 0.0
            beta = 1.0 / self.temperature
            theta_beta = np.arctanh(np.exp(-beta * Omega / 2.0))
            V_prefactor = np.cosh(theta_beta)
            V_tilde_prefactor = np.sinh(theta_beta)
            return V_prefactor, V_tilde_prefactor
        else:
            return 1.0, None

    def _adjust_bosons_if_necessary(self):
        """Adjusts all attributes according to e.g. TFD.

        Note that this method essentially does nothing if the protocol is not
        TFD.
        """

        # Adjust the number of boson types according to thermofield
        if self.protocol == "tfd":
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
            if self.protocol == "tfd":
                self._extend_terms(m, g*V_tilde_prefactor, bt)
                bt += 1

        self._adjust_bosons_if_necessary()

        self.logger.info("Parameters object primed; ready for compute")
