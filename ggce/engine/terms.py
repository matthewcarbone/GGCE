#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import cmath
import numpy as np

from ggce.engine import physics


class BosonConfig:
    """A class for holding boson occupations and defining operations on the
    boson cloud.

    Attributes
    ----------
    config : np.ndarray
        An array of the shape (n_boson_types, cloud_length)
    max_modifications : int
        The maximum number of times one can modify the boson config before
        throwing an error. This is precisely equal to the order of the
        boson creation operators in V.
    """

    def __init__(self, config, max_modifications=1):
        self.config = np.atleast_2d(config)
        self.n_boson_types = self.config.shape[0]
        self.cloud_length = self.config.shape[1]
        self.total_bosons_per_type = np.sum(self.config, axis=1)
        self.total_bosons = np.sum(self.total_bosons_per_type)
        self.n_modified = 0
        self.max_modifications = max_modifications
        self.assert_valid()

    def is_zero(self):
        if self.cloud_length == 1:
            if np.sum(self.config) == 0:
                return True
        return False

    def identifier(self):
        if self.is_zero():
            return "G"
        return str([list(s) for s in self.config])

    def is_legal(self):
        """Checks if the cloud is a legal configuration."""

        if self.is_zero():
            return True

        # Get the edges of the cloud
        edge_left = self.config[:, 0]
        edge_right = self.config[:, -1]
        if np.sum(edge_left) == 0 or np.sum(edge_right) == 0:
            return False
        return True

    def assert_valid(self):
        """Checks that the configuration contains entries >= 0 only. If not
        raises a RuntimeError, else silently does nothing."""

        if np.any(self.config < 0):
            raise RuntimeError(f"Invalid config: {self.config}")
        if np.sum(self.total_bosons_per_type) < 0:
            raise RuntimeError("Config has less than 0 total bosons")

    def remove_boson(self, boson_type, location):
        """Removes a boson of type boson_type from the specified cloud
        location. if removal is not possible, raises a RuntimeError. Returns
        the value to shift the exp_shift and f_arg attributes of the Terms
        class."""

        if self.n_modified >= self.max_modifications:
            raise RuntimeError("Max modifications exceeded")
        if boson_type > self.n_boson_types - 1:
            raise RuntimeError("Boson type remove error")
        if location > self.cloud_length - 1:
            raise RuntimeError("Location remove error")

        self.config[boson_type, location] -= 1

        shift = 0

        if not self.is_zero():

            at_least_one_boson_present = list(np.where(
                np.sum(self.config, axis=0) > 0
            )[0])
            left = min(at_least_one_boson_present)
            right = max(at_least_one_boson_present)

            if right < self.config.shape[1] - 1:
                self.config = self.config[:, :right + 1]

            if left > 0:
                self.config = self.config[:, left:]
                shift = left

        self.assert_valid()
        assert self.is_legal()
        self.n_modified += 1
        self.total_bosons_per_type[boson_type] -= 1
        self.total_bosons -= 1

        return shift

    def add_boson(self, boson_type, location):
        """Adds a boson of type boson_type to the specified location."""

        if self.n_modified >= self.max_modifications:
            raise RuntimeError("Max modifications exceeded")
        if boson_type > self.n_boson_types - 1:
            raise RuntimeError("Boson type add error")

        # Easy case: the boson type to add is in the existing cloud:
        if 0 <= location < self.cloud_length:
            self.config[boson_type, location] += 1

        # Adding a boson to the right or left of the existing cloud requires
        # one to append a new numpy array to the appropriate side.
        elif location < 0:
            new_array_len = -location  # Absolute value of the location
            new_arr = np.zeros((self.n_boson_types, new_array_len)).astype(int)

            # Add a boson to the first element (on the left) of the appropriate
            # boson type
            new_arr[boson_type, 0] += 1

            # Concatenate in the appropriate way
            arrs = [new_arr, self.config]
            self.config = np.concatenate(arrs, axis=1).astype(int)
            self.cloud_length = self.config.shape[1]

        # Here we're adding a boson to the right of the cloud. A similar logic
        # is adopted to the case when we add to the left of the cloud.
        else:
            new_array_len = location - self.cloud_length + 1
            new_arr = np.zeros((self.n_boson_types, new_array_len)).astype(int)
            new_arr[boson_type, -1] += 1
            arrs = [self.config, new_arr]
            self.config = np.concatenate(arrs, axis=1).astype(int)
            self.cloud_length = self.config.shape[1]

        shift = 0
        if location < 0:
            shift = -location  # shift is > 0

        self.assert_valid()
        assert self.is_legal()
        self.n_modified += 1
        self.total_bosons_per_type[boson_type] += 1
        self.total_bosons += 1

        return shift


class Term:
    """Base class for a single term like f_n(d).

    Attributes
    ----------
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
    constant_prefactor : float
        The constant prefactor component. Usually consists of the coupling
        strength, sign of the coupling and other prefactor terms such as
        n_gamma. Default is 1.0.
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

    Parameters
    ----------
    boson_config : BosonConfig
        The configuration of bosons. The BosonConfig class contains the
        necessary methods for adding and removing bosons of various types
        and in various locations.
    hterm : SingleTerm
        The parameters of the term itself. (Single Hamiltonian Term)
    """

    def __init__(self, boson_config, hterm=None, system_params=None):
        self.config = BosonConfig(boson_config)

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
        """Returns a string of the f_arg identifier. """

        return "(%.01f)" % self.f_arg if self.f_arg is not None else "(!)"

    def _get_g_arg_identifier(self):
        """Returns a string of the f_arg identifier."""

        return "<%.01f>" % self.g_arg if self.g_arg is not None else "<!>"

    def _get_c_exp_identifier(self):
        """Returns a string of the current value of the exponential shift."""

        return "[%.01f]" % self.exp_shift \
            if self.exp_shift is not None else "[!]"

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

        if self._get_boson_config_identifier() == '{G}' and self.f_arg != 0.0:
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

    def coefficient(self, k, w, eta=None):
        """This will set the prefactor to G0, since we assume
        that as a base class it originates from the g0 terms in the main
        EOM. Note that an index term does not have this method defined, since
        that terms prefactor should always be 1 (set by default to the
        constant prefactor), and this method will be overridden by
        AnnihilationTerm and CreationTerm classes."""

        if eta is None:
            eta = self.system_params.eta

        return self.constant_prefactor * physics.G0_k_omega(
            k, w, self.system_params.a, eta, self.system_params.t
        ) * cmath.exp(1j * k * self.system_params.a * self.exp_shift)

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
        self.freq_shift = sum([
            self.system_params.Omega[ii] * bpt
            for ii, bpt in enumerate(self.config.total_bosons_per_type)
        ])

    def step(self, location):
        """Increments or decrements the bosons on the chain depending on the
        class derived class type. The location is generally given by gamma
        in my notation."""

        self.g_arg += location
        self.f_arg -= location
        self.modify_n_bosons(self.hterm.bt, location)

    def coefficient(self, k, w, eta=None):

        exp_term = cmath.exp(
            1j * k * self.system_params.a * self.exp_shift
        )

        w_freq_shift = w - self.freq_shift

        if eta is None:
            eta = self.system_params.eta

        g_contrib = physics.g0_delta_omega(
            self.g_arg, w_freq_shift, self.system_params.a,
            eta, self.system_params.t
        )

        return self.constant_prefactor * exp_term * g_contrib


class AnnihilationTerm(NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons(self, boson_type, location):
        """Removes a boson from the specified site corresponding to the
        correct type. Note this step is independent of the reductions of the
        f-functions to other f-functions."""

        shift = self.config.remove_boson(boson_type, location)
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
