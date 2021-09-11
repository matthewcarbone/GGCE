from collections import OrderedDict
import copy
import os
from pathlib import Path
import pickle
import time

from ggce.engine.equations import Equation, GreenEquation
from ggce.utils.logger import Logger
from ggce.utils.combinatorics import ConfigurationSpaceGenerator, \
    total_generalized_equations


def get_GGCE_CONFIG_STORAGE(default_name=".GGCE/GGCE_config_storage"):
    """Returns the user-set value for the location to store the basis
    functions. If  is set as an environment variable,
    that value is returned. Else, the default of /default_name is
    returned.
    
    Parameters
    ----------
    default_name : str, optional
        The name of the directory [structure], relative to $HOME, where the
        basis functions should be stored.

    Returns
    -------
    Posix.Path
    """

    path = os.environ.get("GGCE_CONFIG_STORAGE", None)
    if path is None:
        path = Path.home() / Path(default_name)
    return path


class GeneralizedEquations(dict):
    """Summary
    """
    
    @staticmethod
    def _get_name(model):
        ae = model.absolute_extent
        M = model.M
        N = model.N
        max_bosons_per_site = model.max_bosons_per_site
        return f"{ae}_{M}_{N}_{max_bosons_per_site}_GeneralizedEquations.pkl"

    @classmethod
    def load(cls, model):
        name = Path(GeneralizedEquations._get_name(model))
        name = get_GGCE_CONFIG_STORAGE() / name
        if name.exists():
            d = pickle.load(open(name, "rb"))
            return cls(d, model, recompute_required=False)
        else:
            return cls(dict(), model)

    def save(self):
        root = get_GGCE_CONFIG_STORAGE() 
        name = root / self._name
        if not root.exists():
            root.mkdir(exist_ok=False, parents=True)

        # Convert the class into a proper dictionary before saving
        pickle.dump(dict(self), open(name, "wb"), protocol=4)

    def __init__(self, x, model, recompute_required=True):
        """Summary
        
        Parameters
        ----------
        x : TYPE
            Description
        recompute_required : bool, optional
            Description
        """

        super().__init__(x)
        self._name = self.__class__._get_name(model)
        self._recompute_required = recompute_required


class Metadata(dict):

    def load(cls, name):
        if name.exists():
            d = pickle.load(open(name, "rb"))
            return cls(d, name=name, recompute_required=False)
        else:
            return cls(dict(), name=name, recompute_required=True)

    def save(self):
        self._name.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(dict(self), open(self._name, "wb"), protocol=4)

    def __init__(self, x, name=None, recompute_required=True):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description
        recompute_required : bool, optional
            Description
        """

        super().__init__(x)
        self._name = Path(name)
        self._recompute_required = recompute_required


class System:
    """Defines a list of Equations (a system of equations, so to speak) and
    operations on that system.
    
    Attributes
    ----------
    csg : ggce.utils.combinatorics.ConfigSpaceGenerator
        The class used to generate the generalized equations basis.
    equations : TYPE
        Description
    generalized_equations : List[Equation]
        A list of equations constituting the system; in general form, meaning
        all except for the Green's function are not defined for specific
        delta values.
    """

    def __init__(self, model, logger=Logger(dummy=True), check_load=True):
        """Initializer.
        
        Parameters
        ----------
        model : ggce.model.Model
            Container for the full set of parameters.
        logger : ggce.utils.logger.Logger, optional
            A user-provided logger. Default is a dummy logger which does
            nothing.
        check_load : bool, optional
            If True, checks the disk for the existence of the basis that was
            constructed already. This can save a huge amount of compute time,
            since the basis would otherwise be constructed on every MPI rank.
            In fact, it is recommended that the user preconstruct the basis
            anyway.
        """

        self._logger = logger

        self._model = copy.deepcopy(model)

        # The number of unique boson types has already been evaluated upon
        # initializing the configuration class
        self._n_boson_types = self._model.n_boson_types
        self._max_bosons_per_site = self._model.max_bosons_per_site

        self._master_f_arg_list = None

        # The equations are organized by the number of bosons contained
        # in their configuration space.
        self._initialize_generalized_equations_object(check_load)

        self.equations = dict()

        # Config space generator
        self.csg = ConfigurationSpaceGenerator(self._model)
        self._check_load = check_load

    @property
    def _generalized_equation_name(self):
        ae = self._model.absolute_extent
        M = self._model.M
        N = self._model.N
        max_bosons_per_site = self._model.max_bosons_per_site
        return f"{ae}_{M}_{N}_{max_bosons_per_site}_GeneralizedEquations.pkl"

    @property
    def _equation_name(self):
        return self.__equation_name

    def _initialize_generalized_equations_object(self, check_load):

        name = self._generalized_equation_name

        if check_load:
            self.generalized_equations = Metadata.load(name)
            if self.generalized_equations._recompute_required:
                self._logger.info(
                    "Generalized equations not saved; recalculating"
                )
            else:
                self._logger.info(
                    "Generalized equations loaded"
                )
        else:
            self.generalized_equations = Metadata(dict(), name)
            self._logger.warning(
                "check_load is False, recalculation of the bases will be "
                "forced"
            )

    def _append_generalized_equation(self, n_bosons, config):

        eq = Equation(config, system_params=self._model)

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

    def initialize_generalized_equations(self, check_load=True):
        """Starting with values for the order of the momentum approximation
        and the maximum allowed number of bosons, this method generates a list
        of config arrays, consisting of all possible legal contributions,
        meaning, all vectors [n0, n1, ..., n(ma_order - 1)] such that the
        sum of all of the entries equals n, where n = [1, ..., n_max].
        
        Parameters
        ----------
        check_load : bool, optional
        
        Returns
        -------
        int
            The number of generalized equations.
        """

        t0 = time.time()

        if not self.generalized_equations._recompute_required:
            self._logger.debug("generalized equations read from disk")

        else:
            allowed_configs = self.csg()

            # Generate all possible numbers of bosons consistent with n_max.
            for n_bosons, configs in allowed_configs.items():
                self._extend_legal_configs(n_bosons, configs)

            # Manually append the Green's function (do the same as above
            # except) for this special case. Note that no matter what this is
            # always neded, but the form of the GreenEquation EOM will differ
            # depending on the system type.
            eq = GreenEquation(system_params=self._model)
            eq.initialize_index_term()
            eq.initialize_terms()
            eq._populate_f_arg_terms()
            self._append_master_dictionary(eq)

            # Only one Green's function, with "zero" bosons
            self.generalized_equations[0] = [eq]

            # Save this configuration to disk for the next time
            self.generalized_equations.save()

        self._determine_unique_dictionary()

        dt = time.time() - t0

        L = sum([len(val) for val in self.generalized_equations.values()])
        self._logger.info(f"Generated {L} generalized equations", elapsed=dt)

        # Need to generalize this
        if self._n_boson_types == 1 and self._max_bosons_per_site is None:
            assert len(self._model.M) == 1
            assert len(self._model.N) == 1

            # Plus one for the Green's function
            T = 1 + total_generalized_equations(
                self._model.M[0], self._model.N[0]
            )
            self._logger.debug(
                f"Predicted {T} generalized equations from combinatorics"
            )

            assert T == L, f"{T}, {L}"

        totals = self._get_total_terms()
        self._logger.debug(f"Predicting {totals} total terms")

        # Initialize the self.equations attribute's lists here since we know
        # all the keys:
        for key, _ in self.generalized_equations.items():
            self.equations[key] = []

        return L

    def _append_master_dictionary(self, eq):
        """Takes an equation and appends the master_f_arg_list dictionary."""

        if self._master_f_arg_list is None:
            self._master_f_arg_list = copy.deepcopy(eq.f_arg_terms)
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

        for n_bosons, l_eqs in self.generalized_equations.items():
            for eq in l_eqs:
                n_mat_id = eq.index_term._get_boson_config_identifier()
                l_deltas = self._master_f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = copy.deepcopy(eq)
                    eq_copy.init_full(true_delta)
                    self.equations[n_bosons].append(eq_copy)

        dt = time.time() - t0

        L = sum([len(val) for val in self.equations.values()])
        self._logger.info(f"Generated {L} equations", elapsed=dt)

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
        self._logger.info("Basis has been constructed", elapsed=dt)

        return basis
