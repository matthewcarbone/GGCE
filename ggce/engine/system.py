from collections import OrderedDict
import copy
from pathlib import Path
import time

from ggce.engine.equations import Equation, GreenEquation
from ggce.utils.combinatorics import ConfigurationSpaceGenerator, \
    total_generalized_equations
from ggce.utils.logger import Logger
from ggce.utils.utils import Metadata


class System:
    """Defines a list of Equations (a system of equations, so to speak) and
    operations on that system."""

    def __init__(
        self, model, logger=Logger(dummy=True), ggce_config_storage=None
    ):
        """Initializer.
        
        Parameters
        ----------
        model : ggce.model.Model
            Container for the full set of parameters.
        logger : ggce.utils.logger.Logger, optional
            The logger to use.
        ggce_config_storage : str, optional
            The path to the directory where to save the basis/equations.
        """

        self._logger = logger

        self._model = copy.deepcopy(model)

        # Config space generator
        self._csg = ConfigurationSpaceGenerator(self._model)

        # Load in the other objects if they exist
        if ggce_config_storage is None:
            self._logger.warning("Storage not specified, recalculating basis")
            self._generalized_equations = Metadata.load(None)
            self._master_f_arg_list = Metadata.load(None)
            self._equations = Metadata.load(None)
        else:
            self._logger.info(f"Storage specified at {ggce_config_storage}")
            self._logger.info("Will reload from disk if basis exists")
            Path(ggce_config_storage).mkdir(exist_ok=True, parents=True)
            name = self._model.name
            tmp = f"{name}_g_eq.pkl"
            ge_path = Path(ggce_config_storage) / Path(tmp)
            self._generalized_equations = Metadata.load(ge_path)
            tmp = f"{name}_f_args.pkl"
            f_path = Path(ggce_config_storage) / Path(tmp)
            self._master_f_arg_list = Metadata.load(f_path)
            tmp = f"{name}_eq.pkl"
            e_path = Path(ggce_config_storage) / Path(tmp)
            self._equations = Metadata.load(e_path)

    def _append_generalized_equation(self, n_bosons, config):

        eq = Equation(config, model=self._model)

        # Initialize the index term, basically the f_n(delta)
        eq.initialize_index_term()

        # Initialize the specific terms which constitute the f_n(delta)
        # EOM, such as f_n'(1), f_n''(0), etc.
        eq.initialize_terms(self._csg)

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
            self._generalized_equations[n_bosons].append(eq)
        except KeyError:
            self._generalized_equations[n_bosons] = [eq]

    def _extend_legal_configs(self, n_bosons, configs):
        """Appends a new configuration to the dictionary legal configs."""

        for config in configs:
            self._append_generalized_equation(n_bosons, config)

    def initialize_generalized_equations(self):
        """Starting with values for the order of the momentum approximation
        and the maximum allowed number of bosons, this method generates a list
        of config arrays, consisting of all possible legal contributions,
        meaning, all vectors [n0, n1, ..., n(ma_order - 1)] such that the
        sum of all of the entries equals n, where n = [1, ..., n_max]."""

        t0 = time.time()

        allowed_configs = self._csg()

        # Generate all possible numbers of bosons consistent with n_max.
        for n_bosons, configs in allowed_configs.items():
            self._extend_legal_configs(n_bosons, configs)

        # Manually append the Green's function (do the same as above except)
        # for this special case. Note that no matter what this is always
        # neded, but the form of the GreenEquation EOM will differ depending
        # on the system type.
        eq = GreenEquation(model=self._model)
        eq.initialize_index_term()
        eq.initialize_terms()
        eq._populate_f_arg_terms()
        self._append_master_dictionary(eq)

        # Only one Green's function, with "zero" bosons
        self._generalized_equations[0] = [eq]

        self._determine_unique_dictionary()

        dt = time.time() - t0

        L = sum([len(val) for val in self._generalized_equations.values()])
        self._logger.info(f"Generated {L} generalized equations", elapsed=dt)

        # Need to generalize this
        n_boson_types = self._model._n_boson_types
        max_bosons_per_site = self._model._max_bosons_per_site
        if n_boson_types == 1 and max_bosons_per_site is None:
            assert len(self._model._M) == 1
            assert len(self._model._N) == 1

            # Plus one for the Green's function
            T = 1 + total_generalized_equations(
                self._model._M[0], self._model._N[0]
            )
            self._logger.debug(
                f"Predicted {T} equations from combinatorics equations"
            )

            assert T == L, f"{T}, {L}"

        totals = self._get_total_terms()
        self._logger.debug(f"Predicting {totals} total terms")

        # Initialize the self._equations attribute's lists here since we know
        # all the keys:
        for key, _ in self._generalized_equations.items():
            self._equations[key] = []

        self._generalized_equations.save()
        self._master_f_arg_list.save()

        return L

    def _append_master_dictionary(self, eq):
        """Takes an equation and appends the master_f_arg_list dictionary."""

        if len(self._master_f_arg_list) == 0:
            for key, value in eq.f_arg_terms.items():
                self._master_f_arg_list[key] = value
            return

        # Else, we append the terms
        for n_mat_id, l_deltas in eq.f_arg_terms.items():
            for delta in l_deltas:
                try:
                    self._master_f_arg_list[n_mat_id].append(delta)
                except KeyError:
                    self._master_f_arg_list[n_mat_id] = [delta]

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

    def initialize_equations(self):
        """Generates the true equations on the rhs which have their explicit
        delta values provided."""

        t0 = time.time()

        for n_bosons, l_eqs in self._generalized_equations.items():
            for eq in l_eqs:
                n_mat_id = eq.index_term._get_boson_config_identifier()
                l_deltas = self._master_f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = copy.deepcopy(eq)
                    eq_copy.init_full(true_delta)
                    self._equations[n_bosons].append(eq_copy)

        dt = time.time() - t0

        L = sum([len(val) for val in self._equations.values()])
        self._logger.info(f"Generated {L} equations", elapsed=dt)

        self._equations.save()

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
            else self._equations
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
        for n_bosons, equations in self._equations.items():
            for eq in equations:
                unique_short_identifiers.add(eq.index_term.identifier())
                for term in eq.terms_list:
                    all_terms_rhs.add(term.identifier())
        dt = time.time() - t0

        if unique_short_identifiers == all_terms_rhs:
            self._logger.info("Closure checked and valid", elapsed=dt)
        else:
            self._logger.error("Invalid closure!")
            print(unique_short_identifiers - all_terms_rhs)
            print(all_terms_rhs - unique_short_identifiers)
            self._logger.critical("Critical error due to invalid closure.")

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
            for _, equations in self._equations.items():
                for eq in equations:
                    basis[eq.index_term.identifier()] = cc
                    cc += 1

        else:

            # Set the local basis, in which each identifier gets its own index
            # relative to the n-phonon manifold.
            for n_phonons, list_of_equations in self._equations.items():
                basis[n_phonons] = {
                    eq.index_term.identifier(): ii
                    for ii, eq in enumerate(list_of_equations)
                }

        dt = time.time() - t0
        self._logger.info("Basis has been constructed", elapsed=dt)

        return basis
