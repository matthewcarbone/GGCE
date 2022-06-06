from copy import deepcopy
from time import time

import numpy as np

from ggce import logger
from ggce.engine.terms import Config, config_legal
from ggce.engine.equations import Equation, GreenEquation
from ggce.utils.utils import timeit


# TODO: tagged for C++ acceleration
def config_space_gen(length, total_sum):
    """Generator for yielding all possible combinations of integers of
    length ``length`` that sum to ``total_sum``.

    .. warning::

        Not that cases such as ``length == 4`` and ``total_sum == 5`` such as
        ``[0, 0, 2, 3]`` still need to be screened out, since these do not
        correspond to valid f-functions.

    .. note::

        The algorithm to produce this code can be found `here <https://
        stackoverflow.com/questions/7748442/
        generate-all-possible-lists-of-length-n-that-sum-to-s-in-python>`_.

    Parameters
    ----------
    length : int
        The length of the array to produce.
    total_sum : int
        Constraints the sum over all of the elements of the array to equal this
        value.

    Yields
    ------
    tuple
    """

    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in config_space_gen(length - 1, total_sum - value):
                r = (value,) + permutation
                yield r


def generate_all_legal_configurations(model):
    """Summary

    In one dimension, this is really easy. We can simply iterate over all
    0 < n <= N and 0 < m <= M. Things get much more complicated as the
    dimensionality increases. However, it is simplified somewhat since we
    always assume that M is a "square". We can still generate configurations
    that

    Parameters
    ----------
    model : TYPE
        Description
    """

    phonon_absolute_extent = model.phonon_absolute_extent
    max_phonons_per_site = model.phonon_max_per_site
    phonon_absolute_extent = model.phonon_absolute_extent
    phonon_extent = model.phonon_extent
    phonon_number = model.phonon_number
    n_phonon_types = model.n_phonon_types

    if model.hamiltonian.dimension > 1:
        logger.critical(">1 dimensions not yet implemented")

    nb_max = sum(phonon_number)
    config_dict = {n: [] for n in range(1, nb_max + 1)}
    for nb in range(1, nb_max + 1):
        for z in range(1, phonon_absolute_extent + 1):
            c = list(config_space_gen(z * n_phonon_types, nb))

            # Reshape everything properly:
            # This will only work for 1d!!!
            c = [np.array(cc).reshape(n_phonon_types, -1) for cc in c]

            # Deal with the multiple dimensions later!
            # TODO

            # Now, we get all of the LEGAL nvecs, which are those in
            # which at least a single boson of any type is on both
            # edges of the cloud.
            tmp_legal_configs = [
                Config(cc)
                for cc in c
                if config_legal(
                    cc,
                    max_phonons_per_site=max_phonons_per_site,
                    phonon_extent=phonon_extent,
                )
            ]

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

    @property
    def generalized_equations(self):
        return self._generalized_equations

    def _append_master_dictionary(self, eq):
        """Takes an equation and appends the master_f_arg_list dictionary."""

        if self._master_f_arg_list is None:
            self._master_f_arg_list = deepcopy(eq._f_arg_terms)
            return

        # Else, we append the terms
        for n_mat_id, l_deltas in eq._f_arg_terms.items():
            for delta in l_deltas:
                try:
                    self._master_f_arg_list[n_mat_id].append(delta)
                except KeyError:
                    self._master_f_arg_list[n_mat_id] = [delta]

    def _append_generalized_equation(self, n_bosons, config):

        eq = Equation.from_config(config, model=self.model)

        # Append a master dictionary at the System object level that
        # keeps track of all the f_arg values required for each value of
        # the config.
        self._append_master_dictionary(eq)

        # Finally, append the equation to a master list of generalized
        # equations/
        self._generalized_equations[n_bosons].append(eq)

    def __init__(self, model):
        """Initializer.

        Parameters
        ----------
        model : SystemParameters
            Container for the full set of parameters.
        """

        self.model = deepcopy(model)

        # The number of unique boson types has already been evaluated upon
        # initializing the configuration class
        self.n_boson_types = self.model.n_boson_types
        self.max_bosons_per_site = self.model.max_bosons_per_site

        self._master_f_arg_list = None

        # The equations are organized by the number of bosons contained
        # in their configuration space.
        self.equations = dict()

        # Get all of the allowed configurations
        with timeit(logger.debug, "generate_all_legal_configurations"):
            allowed_configs = generate_all_legal_configurations(self.model)

        self._generalized_equations = {n: [] for n in allowed_configs.keys()}

        # Generate all possible numbers of bosons consistent with n_max.
        with timeit(logger.debug, "initialize equations"):
            for n_bosons, configs in allowed_configs.items():
                for config in configs:
                    self._append_generalized_equation(n_bosons, config)

            eq = GreenEquation(model=self.model)
            self._append_master_dictionary(eq)

            # Only one Green's function, with "zero" bosons
            self._generalized_equations[0] = [eq]

    def initialize_generalized_equations(self):
        """Starting with values for the order of the momentum approximation
        and the maximum allowed number of bosons, this method generates a
        of config arrays, consisting of all possible legal contributions,
        meaning, all vectors [n0, n1, ..., n(ma_order - 1)] such that the
        sum of all of the entries equals n, where n = [1, ..., n_max]."""

        t0 = time()



        self._determine_unique_dictionary()

        dt = time() - t0

        L = sum([len(val) for val in self._generalized_equations.values()])
        logger.info(f"Generated {L} generalized equations")

        # Need to generalize this
        if self.n_boson_types == 1 and self.max_bosons_per_site is None:
            # Plus one for the Green's function
            T = 1 + total_generalized_equations(
                self.model.M, self.model.N, self.n_boson_types
            )
            self.logger.debug(
                f"Predicted {T} equations from combinatorics equations"
            )

            assert T == L, f"{T}, {L}"

        totals = self._get_total_terms()
        self.logger.debug(f"Predicting {totals} total terms")

        # Initialize the self.equations attribute's lists here since we know
        # all the keys:
        for key, _ in self._generalized_equations.items():
            self.equations[key] = []

        return L

    def _determine_unique_dictionary(self):
        """Sorts the master delta terms for easier readability and takes
        only unique delta terms."""

        for n_mat_id, l_deltas in self._master_f_arg_list.items():
            self._master_f_arg_list[n_mat_id] = list(set(l_deltas))

    def _get_total_terms(self):
        """Predicts the total number of required specific equations needed
        to close the system."""

        cc = 0
        for n_mat_id, l_deltas in self._master_f_arg_list.items():
            cc += len(l_deltas)
        return cc

    # @profile
    def initialize_equations(self):
        """Generates the true equations on the rhs which have their explicit
        delta values provided."""

        t0 = time()

        for n_bosons, l_eqs in self._generalized_equations.items():
            for eq in l_eqs:
                n_mat_id = eq.index_term._get_boson_config_identifier()
                l_deltas = self._master_f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = copy.deepcopy(eq)
                    eq_copy.init_full(true_delta)
                    self.equations[n_bosons].append(eq_copy)

        dt = time() - t0

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

        eqs_dict = self._generalized_equations if generalized \
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

        t0 = time()
        unique_short_identifiers = set()
        all_terms_rhs = set()
        for n_bosons, equations in self.equations.items():
            for eq in equations:
                unique_short_identifiers.add(eq.index_term.identifier())
                for term in eq.terms_list:
                    all_terms_rhs.add(term.identifier())
        dt = time() - t0

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
        and the values a list of tuples, where the first element of each
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

        t0 = time()

        # The basis object maps each unique identifier to a unique number.
        # The local_basis object maps each unique identifier to a unique
        # number within the manifold of some number of phonons.
        basis = dict()

        if full_basis:

            # Set the overall basis. Each unique identifier gets its own .
            cc = 0
            for _, equations in self.equations.items():
                for eq in equations:
                    basis[eq.index_term.identifier()] = cc
                    cc += 1

        else:

            # Set the local basis, in which each identifier gets its own
            # relative to the n-phonon manifold.
            for n_phonons, list_of_equations in self.equations.items():
                basis[n_phonons] = {
                    eq.index_term.identifier(): ii
                    for ii, eq in enumerate(list_of_equations)
                }

        dt = time() - t0
        self.logger.info("Basis has been constructed", elapsed=dt)

        return basis
