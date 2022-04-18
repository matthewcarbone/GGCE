import cmath
import numpy as np

from monty.json import MSONable

from ggce import logger
from ggce.utils import physics


class Config(MSONable):
    """A class for holding phonon occupations and defining operations on the
    cloud.

    Attributes
    ----------
    config : numpy.ndarray
        An array of the shape (n_boson_types, cloud length axis 1, 2, ...).
        The array should only contain integers. spatial
    """

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
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
        self._config = x

    @property
    def n_phonon_types(self):
        """The number of different types of phonons in the config. This is
        equal to the shape of the first axis of `config`.

        Returns
        -------
        int
        """

        return self.config.shape[0]

    @property
    def phonon_cloud_shape(self):
        """The shape corresponding to this particular cloud, ignoring
        information about phonon type.

        Returns
        -------
        tuple
        """

        return self.config.shape[1:]

    @property
    def total_phonons_per_type(self):
        """Gets the total number of phonons of each type. Essentially, sums
        over all axes except the first.

        Returns
        -------
        numpy.ndarray
        """
        axes = [ii + 1 for ii in range(len(self.config.shape))]
        return np.sum(self.config, axis=tuple(axes))

    @property
    def total_phonons(self):
        """Sums over the entire config. Returns the total number of phonons in
        the config.

        Returns
        -------
        int
        """

        return int(np.sum(self.config))

    def assert_valid(self):
        """Checks the config for validity and throws a logger.critical, which
        also kills the program if the validity checks fail. If any of the
        following fail, a crtiical is raised:

        1. If any element in the config has less than 0 phonons.
        2. If any of the edges of the cloud has 0 phonons total.

        .. warning::

            The Green's function, i.e. the case where there are actually 0
            phonons on the cloud, will not raise any critical.
        """

        if np.any(self.config < 0):
            logger.critical(f"Invalid config {self.config}: some < 0")

        if self.total_phonons == 0:
            return

        # First, sum up over all of the phonon types. This produces the
        # "anchor" config.
        _config = self.config.sum(axis=0)

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

        self.config = np.array(config).astype(int)
        self._max_modifications = max_modifications
        self._modifications = modifications
        self.assert_valid()

    def __str__(self):
        if self.total_phonons == 0:
            return "G"
        rep = str(list(self.config.flatten()))
        shape = str(self.config.shape)
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
        """Executes the reduction rules. See remove_phonon_ for more details."""

        # No reduction necessary anymore, this is the Green's function
        if self.total_phonons == 0:
            return 0

        _config = self.config.sum(axis=0)

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

        if len(indexes) != len(self.config.shape):
            logger.critical(
                f"Dimension mismatch between config and indexes {indexes}"
            )

        try:
            self._config[indexes] -= 1
        except IndexError:
            logger.critical(
                f"Index does not exist for {self.config} of shape "
                f"{self.config.shape} at {indexes}"
            )

        if self._config[indexes] < 0:
            logger.critical(
                f"Removal error: negative site occupancy for\n {self.config} "
                f"\nof shape {self.config.shape} at {indexes}"
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
            _can_ fall outside of the current config, since phonons can be
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
        if indexes[0] < 0 or indexes[0] > self.config.shape[0] - 1:
            logger.critical(
                f"Phonon index {indexes[0]} invalid for config with "
                f"{self.config.shape[0]} unique phonon types"
            )

        spatial_indexes = np.array(indexes)[1:]
        zeros = np.array([0 for _ in range(len(spatial_indexes))])
        cloud_shape = np.array(self.config.shape)[1:]
        location_matrix = (zeros <= spatial_indexes) & (
            spatial_indexes < cloud_shape
        )

        # Easy case: the boson type to add is in the existing cloud.
        # Here, no padding is required.
        if np.all(location_matrix):
            self.config[indexes] += 1
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
            if 0 <= index < self.config.shape[ii]
            else (-index, 0)
            if index < 0
            else (0, index - self.config.shape[ii + 1] + 1)
            for ii, index in enumerate(spatial_indexes)
        ]

        # Update the config object accordingly
        self.config = np.pad(self.config, pad, "constant", constant_values=0)
        self.config[indexes] += 1

        self._modifications += 1
        return shift


class Term(MSONable):
    """Base class for a single term like f_n(d).

    Attributes
    ----------
    config : ggce.engine.terms.Config
        Phonon configuration object.
    constant_prefactor : float
        The constant prefactor component. Usually consists of the coupling
        strength, sign of the coupling and other prefactor terms such as
        n_gamma. Default is 1.0.
    exp_shift : int
        Factor which multiplies the entire final prefactor term. It is given
        by e^(1j*k*a*exp_shift). This term is determined by the relations
        between the different f-functions.
            1) f_{0, 0, ..., 0, n, m}(d) = f_{n, m, ...}(d + y) * e^{-ikay}
               where y is the number of indexes equal to zero before the first
               nonzero term, and is positive.
            2) f_{n, ..., m, 0, ..., 0} = f_{n, ..., m}
            3) b_gamma^dagger f_{n, m, ...}(d) =
               f_{1, 0, ..., 0, n , m}(0) * e^{ikay}; gamma < 0
               where y is equal to abs(gamma) and is positive.
            4) b_gamma^dagger f_{n, m, ...}(d) =
               f_{1, 0, ..., 0, n , m}(d - y) * e^{ikay}; gamma < 0
    f_arg : int
        The value of f_arg, where we have f_{boson_config}(f_arg). Note that if
        None, this implies that the term for f_arg is left general. This
        should only occur for an FDeltaTerm object that is an "index" for
        the equation, i.e., it is a term that originally appears on the
        lhs. During generation of the specific equations, this will be set
        to a "true delta" value and that equation appended to the master
        list. Default is None, implying the f-argument is a general delta.
    g_arg : int
        The same as for f_arg, but for the green's function delta term.
    hterm : TYPE
        Description
    system_params : TYPE
        Description

    Parameters
    ----------
    boson_config : BosonConfig
        The configuration of bosons. The BosonConfig class contains the
        necessary methods for adding and removing bosons of various types
        and in various locations.
    hterm : SingleTerm
        The parameters of the term itself. (Single Hamiltonian Term)
    """

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        if not isinstance(x, Config):
            logger.critical(f"{x} is not of type Config")
        self._config = x

    def __init__(self, config, hterm=None, system_params=None):
        self.config = Config(config)

        # If hterm is none, we know this is an index term which carries with
        # it no system-dependent prefactors or anything like that.
        self.hterm = hterm
        self.system_params = system_params
        self.constant_prefactor = 1.0

        # Determines the argument of f and prefactors
        self.exp_shift = 0.0  # exp(i*k*a*exp_shift)
        self.f_arg = None  # Only None for the index term
        self.g_arg = None  # => g(...) = 1

    def get_total_bosons(self):
        return self.config.total_bosons

    def _get_boson_config_identifier(self):
        """Returns a string of the boson_config identifier."""

        return "{" + self.config.identifier() + "}"

    def _get_f_arg_identifier(self):
        """Returns a string of the f_arg identifier."""

        return "(%.01f)" % self.f_arg if self.f_arg is not None else "(!)"

    def _get_g_arg_identifier(self):
        """Returns a string of the f_arg identifier."""

        return "<%.01f>" % self.g_arg if self.g_arg is not None else "<!>"

    def _get_c_exp_identifier(self):
        """Returns a string of the current value of the exponential shift."""

        return (
            "[%.01f]" % self.exp_shift if self.exp_shift is not None else "[!]"
        )

    def identifier(self, full=False):
        """Returns a string with which one can index the term. The string takes
        the form n0-n1-...-nL-1-(f_arg)-[exp_shift]. Also defines the
        individual identifiers for the boson_config, f_arg and c_exp shift,
        which will be utilized later.

        Parameters
        ----------
        full : bool
            If true, returns also the g and exp shift terms with the
            identifier. This is purely for visualization and not for e.g.
            running any calculations, since those terms are pooled into the
            coefficient.
        """

        t1 = self._get_boson_config_identifier() + self._get_f_arg_identifier()
        if not full:
            return t1
        t2 = self._get_g_arg_identifier() + self._get_c_exp_identifier()
        return t1 + t2

    def update_boson_config(self):
        """Specific update scheme needs to be run depending on whether we add
        or remove a boson."""

        raise NotImplementedError

    def modify_n_bosons(self):
        """By default, does nothing, will only be defined for the terms
        in which we add or subtract a boson."""

        return

    def coefficient(self, k, w, eta=None):
        raise NotImplementedError

    def increment_g_arg(self, delta):
        self.g_arg += delta

    def set_f_arg(self, val):
        raise NotImplementedError

    def check_if_green_and_simplify(self):
        """There are cases when the algorithm produces results that look like
        e.g. {G}(1)[0]. This corresponds to f_0(1), which is actually just
        a Green's function times a phase factor. The precise relation is
        f_0(delta) = e^{i * k * delta * a} G(k, w). We need to simplify this
        back to a term like {G}(0)[0], as if we don't, when the specific
        equations are generated, it will think there are multiple Greens
        equations, of which of course there can be only one."""

        if self._get_boson_config_identifier() == "{G}" and self.f_arg != 0.0:
            self.exp_shift += self.f_arg
            self.f_arg = 0.0


class IndexTerm(Term):
    """A term that corresponds to an index of the equation. Critical
    differences between this and the base class include that the f_arg object
    is set to None by default, the gamma_idx is set to None (since we will
    not be removing or adding bosons to this term) and the constant_prefactor
    is set to 1, with the lambdafy_prefactor method disabled, and the
    default, unchangeable prefactor being equal to the constant prefactor,
    independent of k and w."""

    def __init__(self, boson_config):
        super().__init__(boson_config=boson_config)

    def set_f_arg(self, val):
        """Overrides the set value of f_arg."""

        assert self.hterm is None
        self.f_arg = val

    def increment_g_arg(self, delta):
        raise NotImplementedError

    def coefficient(self, k, w, eta=None):
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

    def increment_g_arg(self, delta):
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
        self.modify_n_bosons(self.hterm.bt, location)

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

    def modify_n_bosons(self, boson_type, location):
        """Removes a boson from the specified site corresponding to the
        correct type. Note this step is independent of the reductions of the
        f-functions to other f-functions."""

        shift = self.config.remove_phonon_(boson_type, location)
        self.exp_shift -= shift
        self.f_arg += shift


class CreationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons(self, boson_type, location):
        """This is done for the user in update_boson_config. Handles
        the boson creation cases, since these can be a bit
        confusing. Basically, we have the possibility of creating a boson on
        a site that is outside of the range of self.config. So we need methods
        of handling the case when an IndexError would've otherwise been raised
        during an attempt to increment an index that isn't there."""

        shift = self.config.add_boson(boson_type, location)
        self.exp_shift = shift
        self.f_arg -= shift
