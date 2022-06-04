import cmath
import numpy as np

from monty.json import MSONable

from ggce import logger
from ggce.utils import physics
from ggce.model import SingleTerm


def _config_shape_legal(config):
    if len(config.shape) < 2:
        return False
    return True


def _config_values_legal(config):
    if np.any(config < 0):
        return False
    return True


def _config_shape_small_dimension(config):
    if len(config.shape) > 4:
        return False
    return True


def _check_config(config):
    if not _config_shape_legal(config):
        logger.critical(f"Provided config {config} has <2 dimensions")
    if not _config_values_legal(config):
        logger.critical(f"Invalid config {config}: some < 0")
    if not _config_shape_small_dimension(config):
        logger.warning(
            f"Provided config of shape {config.shape} is greater than 4, "
            "indicating more than 3 spatial dimensions. This type of case "
            "is untested and the user should proceed with caution."
        )


def _config_edges_legal(config):

    # First, sum up over all of the phonon types. This produces the
    # "anchor" config.
    _config = config.sum(axis=0)

    # For every dimension, we need to check the "edge". This amounts to
    # swapping the axes and summing
    n_dimensions = len(_config.shape)  # The number of dimensions
    for ii in range(n_dimensions):
        edge_left = _config.swapaxes(0, ii)[0, ...]
        edge_right = _config.swapaxes(0, ii)[-1, ...]
        if np.sum(edge_left) == 0 or np.sum(edge_right) == 0:
            return False
    return True


def _extent_of_1d(config1d):
    """Gets the extent of a 1d vector. TODO: this will require a more general
    solution to find the extent of 2 and 3d systems."""

    where_nonzero = np.where(config1d != 0)[0]
    L = len(where_nonzero)
    if L < 2:
        return L
    minimum_index = min(where_nonzero)
    maximum_index = max(where_nonzero)
    return maximum_index - minimum_index + 1


def config_legal(config, max_phonons_per_site=None, phonon_extent=None):
    """Helper method designed to test a standalone configuration of phonons
    for legality. A `legal` ``config`` array satisfies the following
    properties:

    - It has at least two axes (one phonon index and one spatial)
    - It has only nonnegative entries
    - Every spatial "edge" has at least one phonon of some type
    - The maximum number of phonons per site condition is satisfied
    - The phonon extent criterion is satisfied for all phonon types

    Parameters
    ----------
    config : numpy.array
        The input configuration.
    max_phonons_per_site : int, optional
        If not None, checks that the config satisfies the maximum number of
        phonons per site. Returns False if any site contains more than the
        specified number.
    phonon_extent : list, optional
        A list of int where each entry is the extent allowed for that phonon
        type. Checks this if not None, otherwise ignores.

    Returns
    -------
    bool
        Returns True if the config passes all the tests. False if it fails any
        of them.
    """

    if not _config_shape_legal(config):
        return False
    if not _config_values_legal(config):
        return False
    if not _config_edges_legal(config):
        return False
    if max_phonons_per_site is not None:
        if not np.all(config <= max_phonons_per_site):
            return False
    if phonon_extent is not None:
        if any(
            [
                phonon_extent[ii] < _extent_of_1d(c1d)
                for ii, c1d in enumerate(config)
            ]
        ):
            return False
    return True


def _validate_config_is_legal(config):
    """Helper function for exposing the :class:`Config.validate` method to
    other modules."""

    _check_config(config)

    if np.sum(config) == 0:
        return

    if not _config_edges_legal(config):
        logger.critical("Edge invalid on config axis")


class Config(MSONable):
    """A class for holding phonon occupations and defining operations on the
    cloud."""

    @property
    def config(self):
        """An array of the shape (n_phonon_types, cloud length axis 1, 2, ...).
        The array should only contain integers. Spatial information is
        contained in the indexes 1 and greater. The 0th index contains the
        information about the phonon type.

        Returns
        -------
        numpy.ndarray
        """

        return self._config

    @property
    def shape(self):
        """The shape of the config array.

        Returns
        -------
        tuple
        """

        return self.config.shape

    @property
    def n_spatial_dimensions(self):
        """The number of spatial dimensions in the config. Explicitly excludes
        counting the first dimension, which corresponds to the phonon index.

        Returns
        -------
        int
        """

        return len(self.shape) - 1

    @config.setter
    def config(self, x):
        _check_config(x)
        self._config = x

    @property
    def n_phonon_types(self):
        """The number of different types of phonons in the config. This is
        equal to the shape of the first axis of `config`.

        Returns
        -------
        int
        """

        return self._config.shape[0]

    @property
    def phonon_cloud_shape(self):
        """The shape corresponding to this particular cloud, ignoring
        information about phonon type.

        Returns
        -------
        tuple
        """

        return self._config.shape[1:]

    @property
    def total_phonons_per_type(self):
        """Gets the total number of phonons of each type. Essentially, sums
        over all axes except the first.

        Returns
        -------
        numpy.ndarray
        """

        axes = [ii + 1 for ii in range(len(self._config.shape) - 1)]
        return np.sum(self._config, axis=tuple(axes))

    @property
    def total_phonons(self):
        """Sums over the entire config. Returns the total number of phonons in
        the config.

        Returns
        -------
        int
        """

        return int(np.sum(self._config))

    def validate(self):
        """Checks the config for validity and throws a ``logger.critical``,
        which also kills the program if the validity checks fail. If any of the
        following fail, a critical is raised:

        - If any element in the config has less than 0 phonons.
        - If any of the edges of the cloud has 0 phonons total.

        .. warning::

            The Green's function, i.e. the case where there are actually 0
            phonons on the cloud, will not raise any critical.
        """

        _validate_config_is_legal(self._config)

    def __init__(self, config, max_modifications=1, modifications=0):
        """Initializes the Config class.

        Parameters
        ----------
        config : numpy.ndarray
            The configuration of phonons.
        max_modifications : int
            The maximum number of times one can modify the phonon config before
            throwing an error. This is precisely equal to the order of the
            phonon creation operators in V.
        modifications : int, optional
            The current number of modifications. Default is 0.
        """

        self._config = np.array(config).astype(int)
        self._max_modifications = max_modifications
        self._modifications = modifications
        self.validate()

    def __str__(self):
        if self.total_phonons == 0:
            return "G"
        rep = str(list(self._config.flatten()))
        shape = str(self._config.shape)
        return f"{rep}{shape}"

    def __repr__(self):
        return self.__str__()

    def id(self):
        """Returns the string representation of the config. This is constructed
        by flattening the config, converting to a list and then converting that
        list to a string. In order to ensure any possible ambiguity issues,
        the shape of the initial config is also used as an identifier.

        Returns
        -------
        str
        """

        return self.__str__()

    def _apply_phonon_reduction_rules_(self):
        """Executes the reduction rules. See remove_phonon_ for more
        details."""

        # No reduction necessary anymore, this is the Green's function
        if self.total_phonons == 0:
            return 0

        _config = self._config.sum(axis=0)

        shifts = []
        n_dimensions = len(_config.shape)  # The number of dimensions
        for ii in range(n_dimensions):
            _config = _config.swapaxes(0, ii)

            # Check where there are at least one phonon present at the 0th axis
            at_least_one_phonon_present = np.where(_config > 0)[0]

            # Find the bounds
            left = np.min(at_least_one_phonon_present).item()
            right = np.max(at_least_one_phonon_present).item()

            if right < _config.shape[0] - 1:
                _config = _config[: right + 1, ...]
                self._config = self._config.swapaxes(1, ii + 1)
                self._config = self._config[:, : right + 1, ...]
                self._config = self._config.swapaxes(1, ii + 1)
                shifts.append(0)

            elif left > 0:
                _config = _config[left:, ...]
                self._config = self._config.swapaxes(1, ii + 1)
                self._config = self._config[:, left:, ...]
                self._config = self._config.swapaxes(1, ii + 1)
                shifts.append(left)

            else:
                shifts.append(0)

            _config = _config.swapaxes(0, ii)

        return np.array(shifts)

    def remove_phonon_(self, *indexes):
        """Executes the removal of a phonon, and then all following phonon
        removal "reduction rules". I.e., removes slices of all zeros from the
        edges of the phonon clouds. Essentially Appendix A in this
        `PRB <https://journals.aps.org/prb/abstract/10.1103/
        PhysRevB.104.035106>`_. Specifically equations A1 and A3.

        Parameters
        ----------
        *indexes
            A tuple that must be of the same dimension as the config itself.
            The first value is the phonon index, and the others correspond to
            the real-space location in the phonon to remove. These indexes
            must fall within the existing config, else the program will abort.

        Returns
        -------
        numpy.ndarray
            An array of the index shifts used for modifying the coefficients
            of auxiliary Green's functions.
        """

        if self._modifications >= self._max_modifications:
            logger.critical(
                f"Max modifications {self._max_modifications} exceeded"
            )

        if len(indexes) != len(self._config.shape):
            logger.critical(
                f"Dimension mismatch between config and indexes {indexes}"
            )

        try:
            self._config[indexes] -= 1
        except IndexError:
            logger.critical(
                f"Index does not exist for {self._config} of shape "
                f"{self._config.shape} at {indexes}"
            )

        if self._config[indexes] < 0:
            logger.critical(
                f"Removal error: negative site occupancy for\n {self._config} "
                f"\nof shape {self._config.shape} at {indexes}"
            )

        # Actually modify the config object
        shift = self._apply_phonon_reduction_rules_()

        self._modifications += 1
        return shift

    def add_phonon_(self, *indexes):
        """Executes the addition of a phonon, and then all following phonon
        addition "reduction rules". Essentially Appendix A in this
        `PRB <https://journals.aps.org/prb/abstract/10.1103/
        PhysRevB.104.035106>`_. Specifically equations A2 and A4.

        Parameters
        ----------
        *indexes
            A tuple that must be of the same dimension as the config itself.
            The first value is the phonon index, and the others correspond to
            the real-space location in the phonon to remove. These indexes
            `can` fall outside of the current config, since phonons can be
            added anywhere on the chain.

        Returns
        -------
        numpy.ndarray
            An array of the index shifts used for modifying the coefficients
            of auxiliary Green's functions.
        """

        if self._modifications >= self._max_modifications:
            logger.critical(
                f"Max modifications {self._max_modifications} exceeded"
            )

        if len(indexes) != len(self._config.shape):
            logger.critical(
                f"Dimension mismatch between config and indexes {indexes}"
            )

        # Check that the phonon index is valid
        if indexes[0] < 0 or indexes[0] > self._config.shape[0] - 1:
            logger.critical(
                f"Phonon index {indexes[0]} invalid for config with "
                f"{self._config.shape[0]} unique phonon types"
            )

        spatial_indexes = np.array(indexes)[1:]
        zeros = np.array([0 for _ in range(len(spatial_indexes))])
        cloud_shape = np.array(self._config.shape)[1:]
        location_matrix = (zeros <= spatial_indexes) & (
            spatial_indexes < cloud_shape
        )

        # Easy case: the phonon type to add is in the existing cloud.
        # Here, no padding is required.
        if np.all(location_matrix):
            self._config[indexes] += 1
            self._modifications += 1
            return zeros  # Shift is all zeros in this case

        # Otherwise, we have to pad the array in various ways. First, it is
        # easy to calculate the shift. If an index is less than 0, that will
        # necessitate a shift. If an index is greater than the size of that
        # dimension, it will not necessitate a shift.
        shift = spatial_indexes.copy()
        shift[shift > 0] = 0

        # Now handle the padding. The first element is (0, 0) always, since we
        # never pad the phonon index part of the config
        pad = [(0, 0)]

        # Now, for each of the spatial dimensions we determine the padding
        pad = pad + [
            (0, 0)
            if 0 <= index < self._config.shape[ii]
            else (-index, 0)
            if index < 0
            else (0, index - self._config.shape[ii + 1] + 1)
            for ii, index in enumerate(spatial_indexes)
        ]

        # Update the config object accordingly
        self._config = np.pad(self._config, pad, "constant", constant_values=0)
        indexes = tuple([max(xx, 0) for xx in indexes])
        self._config[indexes] += 1

        self._modifications += 1
        return shift


class Term(MSONable):
    """Base class for a single term in an equation. This object contains all
    information required for defining a term in the sum of equation 22 in the
    `PRB <https://journals.aps.org/prb/abstract/10.1103/
    PhysRevB.104.035106>`_. that this code is based on."""

    @property
    def config(self):
        """The configuration of phonons. The Config class contains the
        necessary methods for adding and removing phonons of various types
        and in various locations. See :class:`.Config`.

        Returns
        -------
        ggce.engine.terms.Config
        """

        return self._config

    @property
    def shape(self):
        """The shape of the config array.

        Returns
        -------
        tuple
        """

        return self.config.shape

    @config.setter
    def config(self, x):
        if not isinstance(x, Config):
            logger.critical(f"Invalid Config term of type {type(x)}")
        self._config = x

    @property
    def hamiltonian_term(self):
        """A single term in the coupling of the Hamiltonian.

        Returns
        -------
        ggce.model.SingleTerm
        """

        return self._hamiltonian_term

    @hamiltonian_term.setter
    def hamiltonian_term(self, x):
        if x is None:
            self._hamiltonian_term = None
            return
        if not isinstance(x, SingleTerm):
            logger.critical(f"Invalid Hamiltonian term of type {type(x)}")
        self._hamiltonian_term = x

    @property
    def constant_prefactor(self):
        """A constant prefactor that multiplies the evaluated value of the
        entire term when constructing the eventual matrix to invert at the end
        of the software pipeline.

        Returns
        -------
        complex
        """

        return self._constant_prefactor

    @constant_prefactor.setter
    def constant_prefactor(self, x):
        if not isinstance(x, (int, float)):
            logger.critical(f"Invalid constant prefactor {x}")
        self._constant_prefactor = x

    @property
    def exp_shift(self):
        """Factor which multiplies the entire final prefactor term in addition
        to :class:`Term.constant_prefactor`. It is given by

        .. math::

            e^{i \\mathbf{k} \\cdot \\boldsymbol{\\delta}
            \\odot \\mathbf{a}}

        where :math:`\\delta` is this ``exp_shift``. Furthermore, this term is
        determined by the relations between the different f-functions. It is
        modified during the reduction rules as noted in
        :class:`Config.remove_phonon_` and :class:`Config.add_phonon_`.

        .. warning::

            The shape of the ``exp_shift`` should be equal to the number of
            spatial dimensions of the phonon configuration.

        Returns
        -------
        numpy.ndarray
        """

        return self._exp_shift

    @exp_shift.setter
    def exp_shift(self, x):
        if x is None:
            self._exp_shift = None
            return
        if len(x) != self.config.n_spatial_dimensions:
            logger.critical(
                f"exp_shift {x} != config spatial dimension "
                f"{self.config.n_spatial_dimensions}"
            )
        self._exp_shift = x

    @property
    def f_arg(self):
        """This array corresponds to the value of the argument of the auxiliary
        Green's function, :math:`f_n(x)`, where :math:`x` is the ``f_arg``.
        If None, this implies that the ``f_arg`` value is left general. This
        should occur if a derived class is ``Index``-like, meaning it indexes
        an entire equation. This is a term that originally appears on the
        "left hand side" of the equation of motion. During generation of the
        specific equations, this will be set to a "true delta" value and that
        equation appended to the master list.

        Returns
        -------
        numpy.ndarray
        """

        return self._f_arg

    @f_arg.setter
    def f_arg(self, x):
        if x is None:
            self._f_arg = None
            return
        if len(x) != self.config.n_spatial_dimensions:
            logger.critical(
                f"f_arg {x} != config spatial dimension "
                f"{self.config.n_spatial_dimensions}"
            )
        self._f_arg = x

    @property
    def g_arg(self):
        """Identical to :class:`Term.f_arg`, but for the argument of the
        lattice Green's function :math:`g_0(x)`, where :math:`x` is ``g_arg``.

        Returns
        -------
        numpy.ndarray
        """

        return self._g_arg

    @g_arg.setter
    def g_arg(self, x):
        if x is None:
            self._g_arg = None
            return
        if len(x) != self.config.n_spatial_dimensions:
            logger.critical(
                f"g_arg {x} != config spatial dimension "
                f"{self.config.n_spatial_dimensions}"
            )
        self._g_arg = x

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return self.__str__()

    def __init__(
        self,
        config,
        hamiltonian_term=None,
        constant_prefactor=1.0,
        exp_shift=None,
        f_arg=None,
        g_arg=None,
    ):
        self._config = Config(config)
        self._hamiltonian_term = hamiltonian_term  # None is the index term
        self._constant_prefactor = constant_prefactor

        # Determines the argument of f and prefactors
        n_dims = self.config.n_spatial_dimensions
        if exp_shift is None:
            self._exp_shift = np.array([0.0 for _ in range(n_dims)])
        else:
            self._exp_shift = exp_shift  # exp(i*k*a*exp_shift)
        self._f_arg = f_arg  # Only None for the index term
        self._g_arg = g_arg  # => g(...) = 1

    def _get_phonon_config_id(self):
        """Returns a string of the phonon config id."""

        return "{" + self.config.id() + "}"

    def _get_f_arg_id(self):
        """Returns a string of the f_arg id."""

        if self._f_arg is None:
            return "(!)"

        return str(list(self._f_arg))

    def _get_g_arg_id(self):
        """Returns a string of the f_arg id."""

        if self._g_arg is None:
            return "<!>"

        return str(list(self._g_arg))

    def _get_c_exp_id(self):
        """Returns a string of the current value of the exponential shift."""

        if self._exp_shift is None:
            return "[!]"

        return str(self._exp_shift)

    def id(self, full=False):
        """Returns a string with which one can index the term. There are two
        types of identifiers the user can request. First is the full=False
        version, which produces a string of the form ``X(f_arg)``, where
        ``X`` is the configuration and ``f_arg`` is the argument of the
        auxiliary Green's function. Second is full=True, which returns the same
        string as full=False, but with ``g_arg`` and ``c_exp`` appended to the
        end as well. ``g_arg`` is the argument of :math:`g_0` and ``c_exp`` is
        the exponential shift indexes.

        Parameters
        ----------
        full : bool
            If true, returns also the g and exp shift terms with the
            id. This is purely for visualization and not for e.g.
            running any calculations, since those terms are pooled into the
            coefficient.

        Returns
        -------
        str
        """

        t1 = self._get_phonon_config_id() + self._get_f_arg_id()
        if not full:
            return t1
        t2 = self._get_g_arg_id() + self._get_c_exp_id()
        return t1 + t2

    def update_phonon_config_(self):
        """Specific update scheme needs to be run depending on whether we add
        or remove a phonon."""

        raise NotImplementedError

    def coefficient(self):
        raise NotImplementedError

    def _modify_n_phonons_(self):
        """By default, does nothing, will only be defined for the terms
        in which we add or subtract a phonon."""

        return

    def _increment_g_arg_(self, add_to_g_arg):
        """Increments the ``g_arg`` object by the provided value.

        Parameters
        ----------
        add_to_g_arg : numpy.ndarray
            The array to add to the ``g_arg`` attribute. If the provided value
            is not of the same shape as the current ``g_arg``, an error will be
            logged, since while this will not necessarily lead to the
            termination of the program, it is likely unintended.
        """

        if add_to_g_arg.shape != self._g_arg.shape:
            logger.error(f"Shape of {add_to_g_arg} != shape of {self._g_arg}")
        self._g_arg += add_to_g_arg

    def _set_f_arg_(self, val):
        raise NotImplementedError

    def check_if_green_and_simplify_(self):
        """There are cases when the algorithm produces results that look like
        e.g. ``G(1)[0]``. This corresponds to ``f_0(1)``, which is actually
        just a Green's function times a phase factor. The precise relation is

        .. math::

            f_0(\\boldsymbol{\\delta}) =
            e^{i \\mathbf{k} \\cdot \\boldsymbol{\\delta}
            \\odot \\mathbf{a}} G(k, w)

        We need to simplify this back to a term like G(0)[0], as if we don't,
        when the specific equations are generated, it will think there are
        multiple Greens equations, of which of course there can be only one."""

        if self._get_phonon_config_id() == "G" and np.any(self._f_arg != 0):
            self._exp_shift += self._f_arg
            self._f_arg = np.zeros_like(self._f_arg)


class IndexTerm(Term):
    """A term that corresponds to an index of the equation. Critical
    differences between this and the base class include that the ``f_arg``
    object is set to None by default and the constant_prefactor is set to 1.
    """

    def __init__(self, config):
        super().__init__(
            config=config,
            hamiltonian_term=None,
            constant_prefactor=1.0,
            exp_shift=None,
            f_arg=None,
            g_arg=None,
        )

    def _set_f_arg_(self, f_arg):
        """Overrides the set value of f_arg."""

        if self._hamiltonian_term is not None:
            logger.critical("Cannot set f_arg in an IndexTerm")
        self.f_arg = f_arg

    def _increment_g_arg_(self):
        raise NotImplementedError

    def coefficient(self, k, w, eta):
        """Coefficient of the IndexTerm (which is always 1).

        Parameters
        ----------
        k : float
        w : complex
        eta : float, optional

        Returns
        -------
        float
            The ``coefficient`` method in :class:`.IndexTerm` always returns 1.
        """

        return 1.0


class EOMTerm(Term):
    """A special instance of the base class that is part of the EOM of the
    Green's function."""

    def __init__(self, config, hamiltonian_term, model):
        super().__init__(config, hamiltonian_term)
        if self.hamiltonian_term is None:
            logger.critical(
                "EOM term requires a Hamiltonian term passed in the "
                "constructor"
            )
        self._exp_shift = self.hamiltonian_term.psi - self.hamiltonian_term.phi
        self._f_arg = self.hamiltonian_term.phi
        self._constant_prefactor = self.hamiltonian_term.coupling
        self._lattice_constant = model.lattice_constant
        self._hopping = model.hopping

    def coefficient(self, k, w, eta):
        """This will set the prefactor to G0, since we assume
        that as a base class it originates from the g0 terms in the main
        EOM. Note that an index term does not have this method defined, since
        that terms prefactor should always be 1 (set by default to the
        constant prefactor), and this method will be overridden by
        AnnihilationTerm and CreationTerm classes."""

        exp_arg = 1j * np.dot(k, self._lattice_constant * self.exp_shift)

        return (
            self._constant_prefactor
            * physics.G0_k_omega(
                k, w, self._lattice_constant, eta, self._hopping
            )
            * cmath.exp(exp_arg)
        )

    def _increment_g_arg_(self, delta):
        return


class NonIndexTerm(Term):
    def __init__(self, config, hamiltonian_term, model, constant_prefactor):
        super().__init__(config, hamiltonian_term, model)
        assert self.hamiltonian_term is not None

        # This is entirely general now
        self.f_arg = self.hamiltonian_term.phi
        self.g_arg = self.hamiltonian_term.psi - self.hamiltonian_term.phi
        self.constant_prefactor = constant_prefactor
        self.freq_shift = sum(
            [
                model.hamiltonian.phonon_frequencies[ii] * bpt
                for ii, bpt in enumerate(self.config.total_phonons_per_type)
            ]
        )
        self._lattice_constant = model.lattice_constant
        self._hopping = model.hopping

    def step_(self, *location):
        """Increments or decrements the phonons on the chain depending on the
        class derived class type."""

        self.g_arg += np.array(location)
        self.f_arg -= np.array(location)
        loc = [self.hamiltonian_term.phonon_index, *location]
        self._modify_n_phonons_(*loc)

    def coefficient(self, k, w, eta):

        exp_arg = 1j * np.dot(k, self._lattice_constant * self.exp_shift)
        exp_term = cmath.exp(exp_arg)

        w_freq_shift = w - self.freq_shift

        g_contrib = physics.g0_delta_omega(
            self.g_arg,
            w_freq_shift,
            self._lattice_constant,
            eta,
            self._hopping,
        )

        return self.constant_prefactor * exp_term * g_contrib


class AnnihilationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a phonon from a specified site."""

    def _modify_n_phonons_(self, *loc):
        """Removes a phonon from the specified site corresponding to the
        correct type. Note this step is independent of the reductions of the
        f-functions to other f-functions."""

        shift = self.config.remove_phonon_(*loc)
        self.exp_shift -= shift
        self.f_arg += shift


class CreationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a phonon from a specified site."""

    def _modify_n_phonons_(self, *loc):
        """This is done for the user in update_phonon_config_. Handles
        the phonon creation cases, since these can be a bit
        confusing. Basically, we have the possibility of creating a phonon on
        a site that is outside of the range of self.config. So we need methods
        of handling the case when an IndexError would've otherwise been raised
        during an attempt to increment an index that isn't there."""

        shift = self.config.add_phonon_(*loc)
        self.exp_shift = shift
        self.f_arg -= shift
