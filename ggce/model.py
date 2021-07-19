#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import namedtuple
import numpy as np
import math

from ggce.utils.logger import Logger


def model_coupling_map(coupling_type, t, Omega, lam):
    """Returns the value for g, the scalar that multiplies the coupling in the
    Hamiltonian. Converts the user-input lambda value to this g. Uses
    pre-defined values for the dimensionless coupling to get g for a variety
    of pre-defined default models.

    Parameters
    ----------
    coupling_type : str
        The desired coupling type. Can be Holstein, Peierls, BondPeierls, or
        EdwardsFermionBoson
    t : float
        The hopping strength.
    Omega : float
        The (Einsten) boson frequency.
    lam : float
        The dimensionless coupling.

    Returns
    -------
    float
        The value for the coupling (g).

    Raises
    ------
    RuntimeError
        If an unknown coupling type is provided.
    """

    if coupling_type == 'Holstein':
        return math.sqrt(2.0 * t * Omega * lam)
    elif coupling_type == 'EdwardsFermionBoson':
        return lam
    elif coupling_type == 'Peierls':
        return math.sqrt(t * Omega * lam / 2.0)
    elif coupling_type == 'BondPeierls':
        return math.sqrt(t * Omega * lam)
    else:
        raise RuntimeError(f"Unknown coupling_type type {coupling_type}")


# Define a namedtuple which contains the shift indexes, x and y, the dagger
# status, d, the coupling term, g, and the boson frequency and type (index)
SingleTerm = namedtuple("SingleTerm", ["x", "y", "d", "g", "bt"])

# Define another namedtuple which contains only the terms that the f-functions
# need to calculate their prefactors, thus saving space.
fFunctionInfo = namedtuple("fFunctionInfo", ["a", "t", "Omega"])


class DefaultHamiltonians:

    ALLOWED_TYPES = [
        'Holstein', 'EdwardsFermionBoson', 'Peierls', 'BondPeierls'
    ]

    def Holstein(g, bt):
        return [
            SingleTerm(x=0, y=0, d='+', g=-g, bt=bt),
            SingleTerm(x=0, y=0, d='-', g=-g, bt=bt)
        ]

    def EdwardsFermionBoson(g, bt):
        return [
            SingleTerm(x=1, y=1, d='+', g=g, bt=bt),
            SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
            SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
            SingleTerm(x=-1, y=0, d='-', g=g, bt=bt)
        ]

    def BondPeierls(g, bt):
        return [
            SingleTerm(x=1, y=0.5, d='+', g=g, bt=bt),
            SingleTerm(x=1, y=0.5, d='-', g=g, bt=bt),
            SingleTerm(x=-1, y=-0.5, d='+', g=g, bt=bt),
            SingleTerm(x=-1, y=-0.5, d='-', g=g, bt=bt)
        ]

    def Peierls(g, bt):
        return [
            SingleTerm(x=1, y=0, d='+', g=g, bt=bt),
            SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
            SingleTerm(x=1, y=1, d='+', g=-g, bt=bt),
            SingleTerm(x=1, y=1, d='-', g=-g, bt=bt),
            SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
            SingleTerm(x=-1, y=-1, d='-', g=g, bt=bt),
            SingleTerm(x=-1, y=0, d='+', g=-g, bt=bt),
            SingleTerm(x=-1, y=0, d='-', g=-g, bt=bt)
        ]


class Model:

    def __init__(
        self, name="model", default_console_logging_level="info", log_file=None
    ):
        self._logger = Logger(log_file, mpi_rank=0)
        self._logger.adjust_logging_level(default_console_logging_level)
        self._parameters_set = False

        # Index uninitialized parameters
        self.t = None
        self.a = None
        self.M = []
        self.N = []
        self.M_tfd = []
        self.N_tfd = []
        self.temperature = None
        self.dimension = None
        self.max_bosons_per_site = None
        self.absolute_extent = None
        self.Omega = []
        self.terms = []
        self.n_boson_types = 0
        self._boson_counter = 0
        self.models_vis = []  # For visualizing the initialized parameters

    def visualize(self, silent=False):
        """Visualize the model you've initialized."""

        print(f"Model parameters initialized: {self._parameters_set}")
        print("Model globals:")
        print(f"\tt={self.t}, a={self.a}, T={self.temperature},", end=' ')
        print(f"dim={self.dimension},", end=' ')
        print(f"hard-core={self.max_bosons_per_site},", end=' ')
        print(f"absolute-extent={self.absolute_extent}")
        print("Model components:")
        for ii, model in enumerate(self.models_vis):
            model_type = model[0]
            M = model[1]
            N = model[2]
            Omega = model[3]
            g = model[4]
            print(f"\t({ii}) {model_type}")
            print(f"\t\tM={M}, N={N}, O={Omega:.05f}, g={g:.05f}")

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

    def _append_N(self, M, N, tfd=False):
        """Appends the list self.N or self.N_tfd based on whether or not a
        hard-core boson constraint was set.

        Parameters
        ----------
        N : int
        M : int
        tfd : bool, optional
            If True, modifies the self.N_tfd list, else modifies self.N (the
            default is False).
        """

        if self.max_bosons_per_site is None:
            if tfd:
                self.N_tfd.append(N)
            else:
                self.N.append(N)
        else:
            if tfd:
                self.N_tfd.append(self.max_bosons_per_site * M)
            else:
                self.N.append(self.max_bosons_per_site * M)

    def get_fFunctionInfo(self):
        return fFunctionInfo(a=self.a, t=self.t, Omega=self.Omega)

    def set_parameters(
        self, hopping=1.0, dimension=1, lattice_constant=1.0, temperature=0.0,
        max_bosons_per_site=None
    ):
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
        temperature : float, optional
            The temperature for a TFD simulation, if requested (the default is
            0.0).
        max_bosons_per_site : int, optional
            A hard core or partially hard core boson constraint: this is the
            maximum number of boson excitations per site on the lattice (the
            default is None, indicating no restriction).
        """

        if dimension > 1:
            raise NotImplementedError

        if temperature < 0.0:
            self._logger.error(
                "Temperature must be non-negative. Parameters remain unset."
            )
            return

        if hopping < 0.0:
            self._logger.error(
                "Hopping strength must be positive. Parameters remain unset."
            )

        if lattice_constant < 0.0:
            self._logger.error(
                "Lattice constant must be positive. Parameters remain unset."
            )

        # List all of the parameters necessary for the run
        self.t = hopping
        self.dimension = dimension
        self.temperature = temperature
        self.a = lattice_constant
        self.max_bosons_per_site = max_bosons_per_site
        self._parameters_set = True

    def _update_absolute_extent(self):

        ae1 = np.max(self.M)
        if self.temperature > 0.0:
            ae1 = max(np.max(self.M_tfd), ae1)

        if self.absolute_extent is None:
            self.absolute_extent = ae1
        else:
            self.absolute_extent = max(ae1, self.absolute_extent)

    def set_absolute_extent(self, ae):
        """The absolute_extent defines how far away boson clouds of different
        modes can be apart from each other. For example, two phonon modes each
        with M = 3 and absolute_extent = 4 can occupy a total length over the
        lattice of at most 4.

        Parameters
        ----------
        ae : int
        """

        ae1 = np.max(self.M)
        if self.temperature > 0.0:
            ae1 = max(np.max(self.M_tfd), ae1)

        if ae < ae1:
            self._logger.error(
                "Cannot set an absolute_extent less than the maximum "
                f"specified extent for any given cloud, {ae1}."
            )
            return

        self.absolute_extent = ae

    def add_coupling(
        self, coupling_type, Omega, M, N, M_tfd=None, N_tfd=None,
        coupling=None, dimensionless_coupling=None, boson_index_override=None
    ):
        """Adds an electron-phonon contribution to the Hamiltonian.

        Note that the user can override the boson index manually to construct
        more complicated models, such as single phonon-mode Hamiltonians but
        with multiple contributions to the coupling term.

        Parameters
        ----------
        coupling_type : {[type]}
            [description]
        Omega : {[type]}
            [description]
        M : {[type]}
            [description]
        N : {[type]}
            [description]
        M_tfd : {[type]}, optional
            [description] (the default is None, which [default_description])
        N_tfd : {[type]}, optional
            [description] (the default is None, which [default_description])
        coupling : {[type]}, optional
            [description] (the default is None, which [default_description])
        dimensionless_coupling : {[type]}, optional
            [description] (the default is None, which [default_description])
        boson_index_override : {[type]}, optional
            [description] (the default is None, which [default_description])
        """

        if not self._parameters_set:
            self._logger.error(
                "Run set_parameters(...) before adding couplings. No coupling "
                "was added."
            )
            return

        if self.temperature == 0 and (M_tfd is not None or N_tfd is not None):
            self._logger.warning(
                "Temperature is set to zero but M_tfd or N_tfd values were "
                "provided and will be ignored."
            )

        if self.temperature > 0.0 and (M_tfd is None or N_tfd is None):
            self._logger.error(
                "Temperature > 0 but M_tfd or N_tfd not set. No coupling "
                "was added."
            )
            return

        if coupling_type not in DefaultHamiltonians.ALLOWED_TYPES:
            self._logger.error(
                f"Provided coupling_type={coupling_type} must be part of the "
                "list or pre-defined couplings: "
                f"{DefaultHamiltonians.ALLOWED_TYPES}. No coupling was added."
            )
            return

        if M < 1 or N < 1:
            self._logger.error(
                "Provided M and N values for the doubled cloud  must "
                "both be > 1. No coupling was added."
            )
            return

        if coupling is None and dimensionless_coupling is None:
            self._logger.error(
                "Provided coupling and dimensionless_coupling cannot both "
                "be unset. No coupling was added."
            )
            return

        if coupling is not None and dimensionless_coupling is not None:
            self._logger.error(
                "Provided coupling and dimensionless_coupling cannot both "
                "be set. No coupling was added."
            )
            return

        # Get the term that multiplies the electron-phonon part of the
        # Hamiltonian
        g = coupling if coupling is not None else \
            model_coupling_map(
                coupling_type, self.t, Omega, dimensionless_coupling
            )

        # Get the TFD prefactors for the terms in the Hamiltonian. Note that
        # if temperature = 0, V_tilde_pf will be None.
        V_pf, V_tilde_pf = self._get_coupling_prefactors(Omega)

        # Extend the terms list with the zero-temperature contribution
        klass = eval(f"DefaultHamiltonians.{coupling_type}")
        self.terms.extend(klass(g * V_pf, self._boson_counter))
        self.Omega.append(Omega)
        self.M.append(M)
        self._append_N(M, N, tfd=False)
        self.models_vis.append([
            coupling_type, M, self.N[-1], Omega, g * V_pf
        ])
        self._boson_counter += 1

        # Finite temperature
        if V_tilde_pf is not None:
            assert self.temperature > 0.0  # Double check
            self.terms.extend(klass(g * V_tilde_pf, self._boson_counter))
            self.Omega.append(-Omega)  # Omega get's a negative sign!!
            self.M_tfd.append(M_tfd)
            self._append_N(M_tfd, N_tfd, tfd=True)
            self.models_vis.append([
                f"~{coupling_type}", M_tfd, self.N_tfd[-1], -Omega,
                g * V_tilde_pf
            ])

        self.n_boson_types = len(self.Omega)
        self._update_absolute_extent()

    def finalize(self):
        """Completes the model initialization and locks the ability to make
        any modifications."""

        if self.temperature > 0.0:
            N_tmp = []
            for N, N_tfd in zip(self.N, self.N_tfd):
                N_tmp.extend([N, N_tfd])
            self.N = N_tmp
            M_tmp = []
            for M, M_tfd in zip(self.M, self.M_tfd):
                M_tmp.extend([M, M_tfd])
            self.M = M_tmp
