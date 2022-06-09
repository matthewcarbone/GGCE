"""The ``equations`` module constructs intermediate objects for dealing with
individual equations in the overall closure.

An :class:`.Equation` can be loosely thought of in the following
pseudo-mathematical equation:

.. math::

    f(n) \\sim \\sum_i f_i(n-1) + \\sum_j f_j(n+1) + b

where Auxiliary Green's functions, :math:`f`, are indexed by the total number
of phonons, :math:`n`. Note that this is of course a "bad" quantum number and
that much more information is required to represent the Auxiliary Green's
function, and that this information is actually contained, but for the sake
of explanation we suppress those parameters.

For couplings of the forms compatible with the GGCE software package, only
a single phonon is created or annihilated at a time. Thus, the equations in
the closer couple Auxiliary Green's Functions with :math:`n` phonons to those
with :math:`n \\pm 1` phonons. There is also a bias term, :math:`b`, which is
generally 0 except in the case of the :class:`.GreenEquation`, which is a
special instance of the :class:`.Equation` used only for the true Green's
function.

.. tip::

    During the operation of the GGCE code, it is unlikely you will need to
    actually manipulate the :class:`.Equation` and :class:`.GreenEquation`
    objects at all, so for the pure user, this module is likely not so
    necessary to understand.

.. note::

    See the :class:`.Equation` and :class:`.GreenEquation` for more details on
    these two important objects.
"""

from copy import deepcopy
import numpy as np

from monty.json import MSONable

from ggce import logger
from ggce.utils import physics
from ggce.utils.utils import timeit
from ggce.engine import terms as terms_module


# TODO: MSONable is broken for this class and its children
class Equation(MSONable):
    """A single equation that indexes a single Auxiliary Green's Function
    (AGF). These functions can be loosely conceptualized in terms of the number
    of phonons in each term, and can be written as

    .. math::

        f(n) \\sim \\sum_i f_i(n-1) + \\sum_j f_j(n+1) + b

    where :math:`n` is an in-exhaustive index/bad quantum number corresponding
    to the total number of phonons on those sites. The :class:`Equation` class
    contains an ``index_term``, which in this case represents :math:`f(n)`, and
    a ``terms_list`` object, which corresponds to the right hand side.

    There is also a bias term, :math:`b`, which represents possible
    inhomogeneities. Only the Green's function itself contains
    :math:`b\\neq 0`. All of these terms are implicitly functions of the
    momentum, :math:`k`, the energy/frequency, :math:`\\omega`, and the
    artificial broadening :math:`\\eta`. For the base :class:`.Equation`, the
    ``bias`` is zero.

    Formally, this class is the representation of equation 22 in
    `PRB <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.035106>`_.
    In one dimension, it is

    .. math::

        f_\\mathbf{n}(\\delta) = \\sum_{(g, \\psi, \\phi, \\xi)} g
        \\sum_{\\gamma \\in \\Gamma_L^\\xi} n^{(\\xi, \\gamma)}
        g_0(\\delta + \\gamma - \\phi + \\psi, \\tilde{\\omega})
        f_{\\mathbf{n}}^{(\\xi, \\gamma)}(\\phi - \\gamma)

    (see the manuscript for a detailed explanation of all of these terms).
    """

    @classmethod
    def from_config(cls, config, model):
        """Initializes the :class:`Equation` class from a numpy array and a
        :class:`ggce.models.Model`.

        Parameters
        ----------
        config : numpy.array
            The array representing the configuration of phonons.
        model : ggce.models.Model
            Container for the full set of parameters. Contains all Hamiltonian
            and parameter information about the terms, phonons, couplings, etc.

        Returns
        -------
        Equation
        """

        index_term = terms_module.IndexTerm(config.copy())
        return cls(index_term, model)

    @property
    def index_term(self):
        """The index term/config referencing the lhs of the equation.

        Returns
        -------
        ggce.engine.terms.IndexTerm
        """

        return self._index_term

    @property
    def model(self):
        """Container for the full set of parameters. Contains all Hamiltonian
        and parameter information about the terms, phonons, couplings, etc.

        Returns
        -------
        ggce.model.Model
            Description
        """

        return self._model

    @property
    def f_arg_terms(self):
        """A dictionary of lists of ints indexed by the config string
        representation containing lists of integers highlighting the required
        values of delta for that particular config string representation.

        Returns
        -------
        dict
        """

        return self._f_arg_terms

    def __init__(self, index_term, model, f_arg_terms=None, terms_list=None):
        logger.debug(f"Initializing {self.__class__.__name__} {index_term}")
        self._index_term = deepcopy(index_term)
        self._model = deepcopy(model)
        self._f_arg_terms = deepcopy(f_arg_terms)
        self._terms_list = deepcopy(terms_list)
        with timeit(logger.debug, "_initialize_terms"):
            self._initialize_terms()
        with timeit(logger.debug, "_populate_f_arg_terms"):
            self._populate_f_arg_terms()

    def bias(self, k, w, eta):
        """The default value for the bias is 0, except in the case of the
        Green's function."""

        return 0.0

    def visualize(self, full=True, coef=None):
        """Prints the representative strings of the terms for both the indexer
        and terms. This allows the user to at a high level, and
        without coefficients, visualize all the terms in the equation.
        If coef is not None, we assume we want to visualize the coefficient
        too, and coef is a 2-vector containing the k and w points.

        Parameters
        ----------
        full : bool, optional
            If True, prints out not only the configuration, its shape and the
            ``f_arg`` terms, but also the ``g_arg`` and ``shift`` components
            of the :class:`ggce.engine.terms.Term` object. Default is True.
        coef : None, optional
            Optional coefficients for :math:`k`, :math:`\\omega` and
            :math:`\\eta` to pass to the terms and output their coefficients.
            This is primarily used for debugging. Default is None (indicating
            no coefficient information will be printed).
        """

        id1 = self.index_term.id(full=full)
        if coef is not None:
            id1 += f" -> {self.index_term.coefficient(*coef)}"
        print(id1)
        if self._terms_list is not None:
            for term in self._terms_list:
                id1 = term.id(full=full)
                if coef is not None:
                    c = term.coefficient(*coef)
                    if c.imag < 0:
                        id1 += f" -> {c.real:.02e} - {-c.imag:.02e}i"
                    else:
                        id1 += f" -> {c.real:.02e} + {c.imag:.02e}i"
                print("\t", id1)

    def _populate_f_arg_terms(self):
        """Populates the required f_arg terms for the 'rhs' (in self._terms_list)
        that will be needed for the non-generalized equations later.
        Specifically, populates a dictionary all_delta_terms which maps the
        f_vec index string to a list of integers, which correspond to the
        needed delta values for that f_vec string. These dictionaries will be
        combined later during production of the non-generalized f-functions."""

        if self._f_arg_terms is not None:
            logger.error("f_args_terms is already initialized")
            return

        self._f_arg_terms = dict()
        for term in self._terms_list:
            n_mat_identifier = term._get_phonon_config_id()
            try:
                self._f_arg_terms[n_mat_identifier].append(term.f_arg)
            except KeyError:
                self._f_arg_terms[n_mat_identifier] = [term.f_arg]

    def _init_full(self, delta):
        """Increments every term in the term list's g-argument by the provided
        value. Also sets the f_arg to the proper value for the index term."""

        self.index_term._set_f_arg_(delta)
        for ii in range(len(self._terms_list)):
            self._terms_list[ii]._increment_g_arg_(delta)

    def _initialize_terms(self):
        """Systematically construct the generalized annihilation terms on the
        rhs of the equation."""

        if self._terms_list is not None:
            logger.error("terms_list is already initialized")
            return

        ae = self._model.phonon_absolute_extent
        n_spatial_dimensions = self._index_term.config.n_spatial_dimensions
        self._terms_list = []

        # Iterate over all possible types of the coupling operators
        for hterm in self._model.hamiltonian.terms:

            bt = hterm.phonon_index  # Phonon type
            arr = self.index_term.config.config[bt, ...]

            # We separate the two cases for creation and annihilation operators
            # on the boson operator 'b'
            if hterm.dag == "-":

                # Iterate over the site indexes for the term's alpha-index
                for loc, nval in np.ndenumerate(arr):

                    # Do not annihilate the vacuum, this term comes to 0
                    # anyway.
                    if nval == 0:
                        continue

                    t = terms_module.AnnihilationTerm(
                        self._index_term.config.config.copy(),
                        hamiltonian_term=hterm,
                        model=deepcopy(self._model),
                        constant_prefactor=hterm.coupling * nval,
                    )
                    t.step_(*loc)
                    t.check_if_green_and_simplify_()
                    t.config.validate()
                    if terms_module.config_legal(
                        t.config.config,
                        self._model.phonon_max_per_site,
                        self._model.phonon_extent,
                        allow_green=True,
                    ):
                        self._terms_list.append(t)

            # Don't want to create an equation corresponding to more than
            # the maximum number of allowed phonons.
            elif hterm.dag == "+":

                if any([xx - ae > 0 for xx in self._index_term.config.shape]):
                    logger.critical(
                        f"Absolute extent {ae} cannot be smaller than any "
                        "config spatial dimension: "
                        f"{self._index_term.config.shape[1:]}"
                    )

                # Create a dummy array to make iterating over indexes easier
                dummy = np.empty(
                    [
                        2 * ae - self._index_term.config.shape[ii + 1]
                        for ii in range(n_spatial_dimensions)
                    ]
                )

                for ii, (index, _) in enumerate(np.ndenumerate(dummy)):
                    loc = tuple(
                        [
                            index[jj]
                            - ae
                            + self._index_term.config.shape[jj + 1]
                            for jj in range(n_spatial_dimensions)
                        ]
                    )

                    t = terms_module.CreationTerm(
                        self._index_term.config.config.copy(),
                        hamiltonian_term=hterm,
                        model=deepcopy(self._model),
                        constant_prefactor=hterm.coupling,
                    )
                    t.step_(*loc)
                    t.check_if_green_and_simplify_()
                    t.config.validate()
                    if terms_module.config_legal(
                        t.config.config,
                        self._model.phonon_max_per_site,
                        self._model.phonon_extent,
                        allow_green=True,
                    ):
                        self._terms_list.append(t)


class GreenEquation(Equation):
    """Equation object corresponding to the Green's function. In one dimension,
    this corresponds directly with equation 13 in this
    `PRB <https://journals.aps.org/prb/abstract/10.1103/
    PhysRevB.104.035106>`_,

    .. math::

        f_0(0) - \\sum_{(g,\\psi,\\phi)} g e^{ik R_{\\psi - \\phi}} f_1(\\phi)
        = G_0(k, \\omega)

    The primary way in which this object differs from :class:`.Equation` is
    that it has a non-zero :class:`.GreenEquation.bias`.

    """

    def __init__(self, model):
        config = np.zeros(
            (
                model.n_phonon_types,
                *[1 for _ in range(model.hamiltonian.dimension)],
            )
        )
        index_term = terms_module.IndexTerm(config.copy())
        super().__init__(index_term=index_term, model=model)

    def bias(self, k, w, eta):
        """The bias term for the Green's function equation of motion.

        Parameters
        ----------
        k : float
            The momentum index.
        w : complex
            The complex frequency.
        eta : float
            The artificial broadening term.

        Returns
        -------
        complex
            This is :math:`G_0(k, \\omega; a, \\eta, t)`, where for the
            free particle Green's function on a lattice, :math:`G_0` is
            parameterized by the lattice constant, broadening and hopping.
        """

        return physics.G0_k_omega(
            k, w, self.model.lattice_constant, eta, self.model.hopping
        )

    def _initialize_terms(self):
        """Override for the Green's function other terms."""

        # Only instance where we actually use the base FDeltaTerm class
        self._terms_list = []

        # Iterate over all possible types of the coupling operators
        for hterm in self._model.hamiltonian.terms:

            # Only phonon creation terms contribute to the Green's function
            # EOM since annihilation terms hit the vacuum and yield zero.
            if hterm.dag == "+":
                n = self._index_term.config.config.copy()

                # This operation below is safe because the config for the
                # Green's function equation of motion will always have a shape
                # of (1, 1, ...).
                n[hterm.phonon_index, ...] = 1
                t = terms_module.EOMTerm(
                    n,
                    hamiltonian_term=hterm,
                    model=deepcopy(self._model),
                )
                self._terms_list.append(t)
