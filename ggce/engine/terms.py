import cmath
import numpy as np

from monty.json import MSONable

from ggce import logger
from ggce.utils import physics
from ggce.model import SingleTerm


class Config(MSONable):
    """A class for holding phonon occupations and defining operations on the
    cloud."""

    @staticmethod
    def _check_config(x):
        if len(x.shape) < 2:
            logger.critical(f"Provided config {x} has <2 dimensions")
        if np.any(np.array(x.shape[1:]) == 1):
            logger.critical(f"Provided config {x} has extra dimensions")
        if len(x.shape) > 4:
            logger.warning(
                f"Provided config of shape {x.shape} is greater than 4, "
                "indicating more than 3 spatial dimensions. This type of case "
                "is untested and the user should proceed with caution."
            )

    @property
    def config(self):
        """An array of the shape (n_boson_types, cloud length axis 1, 2, ...).
        The array should only contain integers. Spatial information is
        contained in the indexes 1 and greater. The 0th index contains the
        information about the phonon type.

        Returns
        -------
        numpy.ndarray
        """

        return self._config

    @config.setter
    def config(self, x):
        Config._check_config(x)
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
        axes = [ii + 1 for ii in range(len(self._config.shape))]
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

        1. If any element in the config has less than 0 phonons.
        2. If any of the edges of the cloud has 0 phonons total.

        .. warning::

            The Green's function, i.e. the case where there are actually 0
            phonons on the cloud, will not raise any critical.
        """

        Config._check_config(self._config)

        if np.any(self._config < 0):
            logger.critical(f"Invalid config {self._config}: some < 0")

        if self.total_phonons == 0:
            return

        # First, sum up over all of the phonon types. This produces the
        # "anchor" config.
        _config = self._config.sum(axis=0)

        # For every dimension, we need to check the "edge". This amounts to
        # swapping the axes and summing
        n_dimensions = len(_config.shape)  # The number of dimensions
        for ii in range(n_dimensions):
            edge_left = _config.swapaxes(0, ii)[0, ...]
            edge_right = _config.swapaxes(0, ii)[-1, ...]
            if np.sum(edge_left) == 0 or np.sum(edge_right) == 0:
                logger.critical(f"Edge invalid on config axis {ii + 1}")

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
        return f"{rep} {shape}"

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
                f"Max modifications {self.max_modifications} exceeded"
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

        # Easy case: the boson type to add is in the existing cloud.
        # Here, no padding is required.
        if np.all(location_matrix):
            self._config[indexes] += 1
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
        necessary methods for adding and removing bosons of various types
        and in various locations. See :class:`.Config`.

        Returns
        -------
        ggce.engine.terms.Config
        """

        return self._config

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

    def _assert_shape_equal_to_config(self, x):
        if not isinstance(x, np.ndarray):
            logger.critical(f"Wrong type {x}")
        n_dims = len(self._config.config.shape) - 1
        if x.shape != n_dims:
            logger.critical(
                "Dimension mismatch between config of shape "
                f"{self.config.shape} and provided {x} of shape {x.shape}"
            )

    @exp_shift.setter
    def exp_shift(self, x):
        self._assert_shape_equal_to_config(x)
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
        self._assert_shape_equal_to_config(x)
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
        self._assert_shape_equal_to_config(x)
        self._g_arg = x

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
        if exp_shift is None:
            n_dims = len(self._config.config.shape) - 1
            self._exp_shift = np.array([0.0 for _ in range(n_dims)])
        else:
            self._assert_shape_equal_to_config(exp_shift)
            self._exp_shift = exp_shift  # exp(i*k*a*exp_shift)
        self._assert_shape_equal_to_config(f_arg)
        self._f_arg = f_arg  # Only None for the index term
        self._assert_shape_equal_to_config(g_arg)
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

        return f"[{str(self._exp_shift)}]"

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

    def update_boson_config_(self):
        """Specific update scheme needs to be run depending on whether we add
        or remove a phonon."""

        raise NotImplementedError

    def coefficient(self):
        raise NotImplementedError

    def modify_n_bosons_(self):
        """By default, does nothing, will only be defined for the terms
        in which we add or subtract a phonon."""

        return

    def increment_g_arg_(self, add_to_g_arg):
        """Increments the ``g_arg`` object by the provided value.

        Parameters
        ----------
        add_to_g_arg : numpy.ndarray
            The array to add to the ``_g_arg`` attribute. If the provided value
            is not of the same shape as the current ``_g_arg``, an error will
            be logged, since while this will not necessarily lead to the
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
        super().__init__(config=config)

    def _set_f_arg_(self, f_arg):
        """Overrides the set value of f_arg."""

        if self._hamiltonian_term is not None:
            logger.critical("Cannot set f_arg in an IndexTerm")
        self.f_arg = f_arg

    def increment_g_arg_(self):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hterm is not None
        self.exp_shift = self.hterm.x - self.hterm.y
        self.f_arg = self.hterm.y
        self.constant_prefactor = self.hterm.g

    def coefficient(self, k, w, eta):
        """This will set the prefactor to G0, since we assume
        that as a base class it originates from the g0 terms in the main
        EOM. Note that an index term does not have this method defined, since
        that terms prefactor should always be 1 (set by default to the
        constant prefactor), and this method will be overridden by
        AnnihilationTerm and CreationTerm classes."""

        return (
            self.constant_prefactor
            * physics.G0_k_omega(
                k, w, self.system_params.a, eta, self.system_params.t
            )
            * cmath.exp(1j * k * self.system_params.a * self.exp_shift)
        )

    def increment_g_arg_(self, delta):
        return


class NonIndexTerm(Term):
    def __init__(self, boson_config, hterm, system_params, constant_prefactor):
        super().__init__(boson_config, hterm, system_params)
        assert self.hterm is not None

        # This is entirely general now
        self.f_arg = self.hterm.y
        self.g_arg = self.hterm.x - self.hterm.y
        self.constant_prefactor = constant_prefactor
        self.freq_shift = sum(
            [
                self.system_params.Omega[ii] * bpt
                for ii, bpt in enumerate(self.config.total_bosons_per_type)
            ]
        )

    def step(self, location):
        """Increments or decrements the bosons on the chain depending on the
        class derived class type. The location is generally given by gamma
        in my notation."""

        self.g_arg += location
        self.f_arg -= location
        self.modify_n_bosons_(self.hterm.bt, location)

    def coefficient(self, k, w, eta):

        exp_term = cmath.exp(1j * k * self.system_params.a * self.exp_shift)

        w_freq_shift = w - self.freq_shift

        g_contrib = physics.g0_delta_omega(
            self.g_arg,
            w_freq_shift,
            self.system_params.a,
            eta,
            self.system_params.t,
        )

        return self.constant_prefactor * exp_term * g_contrib


class AnnihilationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons_(self, boson_type, location):
        """Removes a boson from the specified site corresponding to the
        correct type. Note this step is independent of the reductions of the
        f-functions to other f-functions."""

        shift = self.config.remove_phonon_(boson_type, location)
        self.exp_shift -= shift
        self.f_arg += shift


class CreationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons_(self, boson_type, location):
        """This is done for the user in update_boson_config_. Handles
        the boson creation cases, since these can be a bit
        confusing. Basically, we have the possibility of creating a boson on
        a site that is outside of the range of self.config. So we need methods
        of handling the case when an IndexError would've otherwise been raised
        during an attempt to increment an index that isn't there."""

        shift = self.config.add_boson(boson_type, location)
        self.exp_shift = shift
        self.f_arg -= shift
