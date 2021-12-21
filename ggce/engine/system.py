#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import OrderedDict
import copy
import numpy as np

import time

from ggce.engine.equations import Equation, GreenEquation
from ggce.engine.physics import total_generalized_equations
from ggce.utils.logger import Logger

BYTES_TO_MB = 1048576


def config_space_gen(length, total_sum):
    """Generator for yielding all possible combinations of integers of
    length `length` that sum to total_sum. Not that cases such as
    length = 4 and total_sum = 5 like [0, 0, 2, 3] need to be screened
    out, since these do not correspond to valid f-functions.

    Source of algorithm:
    https://stackoverflow.com/questions/7748442/
    generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """

    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in config_space_gen(length - 1, total_sum - value):
                r = (value,) + permutation
                yield r


class ConfigurationSpaceGenerator:
    """Helper class for generating configuration spaces of bosons.

    Parameters
    ----------
    system_params : SystemParams
        The system parameters containing:
        * absolute_extent : int
            The maximum extent of the cloud. Note that this is NOT trivially
            the sum(M) since there are different ways in which the clouds can
            overlap. Take for instance, two boson types, with M1 = 3 and
            M2 = 2. These clouds can overlap such that absolute extent is at
            most 5, and they can also overlap such that the maximal extent is
            3.
        * M : list
            Length of the allowed cloud extent for each boson type.
        * N : list
            Total number of maximum allowed bosons for each boson type. The
            absolute maximum number of bosons is trivially sum(N).
    """

    def __init__(self, system_params):
        self.absolute_extent = system_params.absolute_extent
        self.M = system_params.M
        assert self.absolute_extent >= max(self.M)
        self.N = system_params.N
        self.N_2d = np.atleast_2d(self.N).T
        self.n_boson_types = len(self.M)
        assert len(self.N) == self.n_boson_types
        self.max_bosons_per_site = system_params.max_bosons_per_site

    def extent_of_1d(self, config1d):
        """Gets the extent of a 1d vector."""

        where_nonzero = np.where(config1d != 0)[0]
        L = len(where_nonzero)
        if L < 2:
            return L

        minimum_index = min(where_nonzero)
        maximum_index = max(where_nonzero)
        return maximum_index - minimum_index + 1

    def is_legal(self, config):
        """Checks the condition of a config. If legal, returns True, else
        returns False."""

        # First, check that the edges of the cloud have at least one boson
        # of either type on it.
        if np.sum(config[:, 0]) < 1 or np.sum(config[:, -1]) < 1:
            return False

        # Second, check that the boson numbers for each boson type are
        # satisfied
        if not np.all(config.sum(axis=1, keepdims=True) <= self.N_2d):
            return False

        # Finally, check that each boson type satisifes its extent rule
        if any([
            self.M[ii] < self.extent_of_1d(c1d)
            for ii, c1d in enumerate(config)
        ]):
            return False

        # Constraint that we have maximum N bosons per site.
        if self.max_bosons_per_site is not None:
            if not np.all(config <= self.max_bosons_per_site):
                return False

        return True

    def __call__(self):
        """Generates all possible configurations of bosons for the composite
        system. Then, reduce that space down based on the specific rules for
        each boson type."""

        nb_max = sum(self.N)
        config_dict = {N: [] for N in range(1, nb_max + 1)}
        for nb in range(1, nb_max + 1):
            for z in range(1, self.absolute_extent + 1):
                c = list(config_space_gen(z * self.n_boson_types, nb))

                # Reshape everything properly:
                c = [np.array(z).reshape(self.n_boson_types, -1) for z in c]

                # Now, we get all of the LEGAL nvecs, which are those in
                # which at least a single boson of any type is on both
                # edges of the cloud.
                tmp_legal_configs = [nn for nn in c if self.is_legal(nn)]

                # Extend the temporary config
                config_dict[nb].extend(tmp_legal_configs)

        return config_dict


class System:
    """Defines a list of Equations (a system of equations, so to speak) and
    operations on that system.

    Attributes
    ----------
    generalized_equations : List[Equation]
        A list of equations constituting the system; in general form, meaning
        all except for the Green's function are not defined for specific
        delta values.
    """

    def __init__(self, system_params, logger=Logger(dummy=True)):
        """Initializer.

        Parameters
        ----------
        system_params : SystemParameters
            Container for the full set of parameters.
        """

        self.logger = logger

        self.system_params = copy.deepcopy(system_params)

        # The number of unique boson types has already been evaluated upon
        # initializing the configuration class
        self.n_boson_types = self.system_params.n_boson_types
        self.max_bosons_per_site = self.system_params.max_bosons_per_site

        self.master_f_arg_list = None

        # The equations are organized by the number of bosons contained
        # in their configuration space.
        self.generalized_equations = dict()
        self.equations = dict()

        # Config space generator
        self.csg = ConfigurationSpaceGenerator(self.system_params)

    def _append_generalized_equation(self, n_bosons, config):

        eq = Equation(config, system_params=self.system_params)

        # Initialize the index term, basically the f_n(delta)
        eq.initialize_index_term()

        # Initialize the specific terms which constitute the f_n(delta)
        # EOM, such as f_n'(1), f_n''(0), etc.
        eq.initialize_terms(self.csg)

        # Initialize the required values for delta (arguments of f) as
        # derived from the terms themselves, used later when generating
        # the specific equations.
        eq._populate_f_arg_terms()

        # Append a master dictionary at the System object level that
        # keeps track of all the f_arg values required for each value of
        # the config.
        self._append_master_dictionary(eq)

        # Finally, append the equation to a master list of generalized
        # equations/
        try:
            self.generalized_equations[n_bosons].append(eq)
        except KeyError:
            self.generalized_equations[n_bosons] = [eq]

    def _extend_legal_configs(self, n_bosons, configs):
        """Appends a new configuration to the dictionary legal configs."""

        for config in configs:
            self._append_generalized_equation(n_bosons, config)

    # @profile
    def initialize_generalized_equations(self):
        """Starting with values for the order of the momentum approximation
        and the maximum allowed number of bosons, this method generates a list
        of config arrays, consisting of all possible legal contributions,
        meaning, all vectors [n0, n1, ..., n(ma_order - 1)] such that the
        sum of all of the entries equals n, where n = [1, ..., n_max]."""

        t0 = time.time()

        allowed_configs = self.csg()

        # Generate all possible numbers of bosons consistent with n_max.
        for n_bosons, configs in allowed_configs.items():
            self._extend_legal_configs(n_bosons, configs)

        # Manually append the Green's function (do the same as above except)
        # for this special case. Note that no matter what this is always
        # neded, but the form of the GreenEquation EOM will differ depending
        # on the system type.
        eq = GreenEquation(system_params=self.system_params)
        eq.initialize_index_term()
        eq.initialize_terms()
        eq._populate_f_arg_terms()
        self._append_master_dictionary(eq)

        # Only one Green's function, with "zero" bosons
        self.generalized_equations[0] = [eq]

        self._determine_unique_dictionary()

        dt = time.time() - t0

        L = sum([len(val) for val in self.generalized_equations.values()])
        self.logger.info(f"Generated {L} generalized equations", elapsed=dt)

        # Need to generalize this
        if self.n_boson_types == 1 and self.max_bosons_per_site is None:
            # Plus one for the Green's function
            T = 1 + total_generalized_equations(
                self.system_params.M, self.system_params.N, self.n_boson_types
            )
            self.logger.debug(
                f"Predicted {T} equations from combinatorics equations"
            )

            assert T == L, f"{T}, {L}"

        totals = self._get_total_terms()
        self.logger.debug(f"Predicting {totals} total terms")

        # Initialize the self.equations attribute's lists here since we know
        # all the keys:
        for key, _ in self.generalized_equations.items():
            self.equations[key] = []

        return L

    def _append_master_dictionary(self, eq):
        """Takes an equation and appends the master_f_arg_list dictionary."""

        if self.master_f_arg_list is None:
            self.master_f_arg_list = copy.deepcopy(eq.f_arg_terms)
            return

        # Else, we append the terms
        for n_mat_id, l_deltas in eq.f_arg_terms.items():
            for delta in l_deltas:
                try:
                    self.master_f_arg_list[n_mat_id].append(delta)
                except KeyError:
                    self.master_f_arg_list[n_mat_id] = [delta]

    def _determine_unique_dictionary(self):
        """Sorts the master delta terms for easier readability and takes
        only unique delta terms."""

        for n_mat_id, l_deltas in self.master_f_arg_list.items():
            self.master_f_arg_list[n_mat_id] = list(set(l_deltas))

    def _get_total_terms(self):
        """Predicts the total number of required specific equations needed
        to close the system."""

        cc = 0
        for n_mat_id, l_deltas in self.master_f_arg_list.items():
            cc += len(l_deltas)
        return cc

    # @profile
    def initialize_equations(self):
        """Generates the true equations on the rhs which have their explicit
        delta values provided."""

        t0 = time.time()

        for n_bosons, l_eqs in self.generalized_equations.items():
            for eq in l_eqs:
                n_mat_id = eq.index_term._get_boson_config_identifier()
                l_deltas = self.master_f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = copy.deepcopy(eq)
                    eq_copy.init_full(true_delta)
                    self.equations[n_bosons].append(eq_copy)

        dt = time.time() - t0

        L = sum([len(val) for val in self.equations.values()])
        self.logger.info(f"Generated {L} equations", elapsed=dt)

        return L

    def visualize(self, generalized=True, full=True, coef=None):
        """Allows for easy visualization of the closure. Note this isn't
        recommended when there are greater than 10 or so equations, since
        it will be very difficult to see everything.

        Parameters
        ----------
        generalized : bool
            If True, prints information on the generalized equations, else
            prints the full equations. Default is True.
        full : bool
            If True, prints information about the argument of g(...) and the
            exponential shift in addition to the n-vector and f-argument.
            Else, just prints the latter two. Default is True.
        coef : list, optional
            If not None, actually evaluates the terms at the value of the
            (k, w) coefficient. Default is None.
        """

        eqs_dict = self.generalized_equations if generalized \
            else self.equations
        od = OrderedDict(sorted(eqs_dict.items(), reverse=True))
        for n_bosons, eqs in od.items():
            print(f"{n_bosons}")
            print("-" * 60)
            for eq in eqs:
                eq.visualize(full=full, coef=coef)
            print("\n")

    def generate_unique_terms(self):
        """Constructs the basis for the problem, which is a mapping between
        the short identifiers and the index of the basis. Also, run a sanity
        check on the unique keys, which should equal the number of equations.
        """

        t0 = time.time()
        unique_short_identifiers = set()
        all_terms_rhs = set()
        for n_bosons, equations in self.equations.items():
            for eq in equations:
                unique_short_identifiers.add(eq.index_term.identifier())
                for term in eq.terms_list:
                    all_terms_rhs.add(term.identifier())
        dt = time.time() - t0

        if unique_short_identifiers == all_terms_rhs:
            self.logger.info("Closure checked and valid", elapsed=dt)
        else:
            self.logger.error("Invalid closure!")
            print(unique_short_identifiers - all_terms_rhs)
            print(all_terms_rhs - unique_short_identifiers)
            self.logger.critical("Critical error due to invalid closure.")

    def get_basis(self, full_basis=False):
        """Prepares the solver-specific information.

        Returns the non-zero elements of the matrix in the following format.
        The returned quantity is a dictionary indexed by the order of the
        hierarchy (in this case, the number of phonons contained). Each
        element of this dictionary is another dictionary, with the keys being
        the index term identifier (basically indexing the row of the matrix),
        and the values a list of tuples, where the first element of each tuple
        is the identifier (a string) and the second is a callable function of
        k and omega, representing the coefficient at that point.

        Parameters
        ----------
        full_basis : bool, optional
            If True, returns the full basis mapping. If False, returns the
            local basis mapping, which is used in the continued fraction
            solver. (The default is False).

        Returns
        -------
        dict
            The dictionary objects containing the basis.
        """

        t0 = time.time()

        # The basis object maps each unique identifier to a unique number.
        # The local_basis object maps each unique identifier to a unique
        # number within the manifold of some number of phonons.
        basis = dict()

        if full_basis:

            # Set the overall basis. Each unique identifier gets its own index.
            cc = 0
            for _, equations in self.equations.items():
                for eq in equations:
                    basis[eq.index_term.identifier()] = cc
                    cc += 1

        else:

            # Set the local basis, in which each identifier gets its own index
            # relative to the n-phonon manifold.
            for n_phonons, list_of_equations in self.equations.items():
                basis[n_phonons] = {
                    eq.index_term.identifier(): ii
                    for ii, eq in enumerate(list_of_equations)
                }

        dt = time.time() - t0
        self.logger.info("Basis has been constructed", elapsed=dt)

        return basis
