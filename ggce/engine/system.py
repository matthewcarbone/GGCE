# TODO: entire module should be accelerated via C++

from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import json
import numpy as np
import pickle

from ggce.logger import logger
from ggce.model import Model
from ggce.engine.terms import Config, config_legal
from ggce.engine.equations import Equation, GreenEquation
from ggce.utils.utils import timeit
from ggce.utils.combinatorics import total_generalized_equations


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
            # which at least a single phonon of any type is on both
            # edges of the cloud.
            tmp_legal_configs = [
                Config(cc)
                for cc in c
                if config_legal(
                    cc,
                    max_phonons_per_site=max_phonons_per_site,
                    phonon_extent=phonon_extent,
                    phonon_number=phonon_number,
                )
            ]

            # Extend the temporary config
            config_dict[nb].extend(tmp_legal_configs)

    return config_dict


class System:
    """Defines a list of Equations (a system of equations, so to speak) and
    operations on that system."""

    @property
    def model(self):
        return self._model

    @property
    def generalized_equations(self):
        """A list of equations constituting the system; in general form,
        meaning all except for the Green's function are not defined for
        specific delta values.

        Returns
        -------
        list
            A list of :class:`Equation` objects.
        """

        return self._generalized_equations

    @property
    def equations(self):
        return self._equations

    def _append_master_dictionary(self, eq):
        """Takes an equation and appends the master_f_arg_list dictionary."""

        if self._f_arg_list is None:
            self._f_arg_list = deepcopy(eq._f_arg_terms)
            return

        # Else, we append the terms
        for n_mat_id, l_deltas in eq._f_arg_terms.items():
            for delta in l_deltas:
                try:
                    self._f_arg_list[n_mat_id].append(delta)
                except KeyError:
                    self._f_arg_list[n_mat_id] = [delta]

    def _append_generalized_equation(self, n_phonons, config):
        eq = Equation.from_config(config, model=self._model)

        # Append a master dictionary at the System object level that
        # keeps track of all the f_arg values required for each value of
        # the config.
        self._append_master_dictionary(eq)

        # Finally, append the equation to a master list of generalized
        # equations/
        self._generalized_equations[n_phonons].append(eq)

    def _determine_unique_dictionary(self):
        """Sorts the master delta terms for easier readability and takes
        only unique delta terms."""

        for n_mat_id, l_deltas in self._f_arg_list.items():
            new_list = [
                np.array(xx)
                for xx in set([tuple(yy.tolist()) for yy in l_deltas])
            ]
            self._f_arg_list[n_mat_id] = new_list

    def _get_total_terms(self):
        """Predicts the total number of required specific equations needed
        to close the system."""

        return sum([len(ll) for ll in self._f_arg_list.values()])

    def _predict_total_terms(self):
        L = sum([len(val) for val in self._generalized_equations.values()])

        # Need to generalize this
        phonon_max_per_site = self._model.phonon_max_per_site
        n_phonon_types = self._model.n_phonon_types
        if n_phonon_types == 1 and phonon_max_per_site is None:
            # Plus one for the Green's function

            T = 1 + total_generalized_equations(
                self._model.phonon_extent,
                self._model.phonon_number,
                n_phonon_types,
            )

            if L == T:
                logger.info(
                    f"Predicted {L} generalized equations (agrees with "
                    "analytic formula)"
                )
            else:
                logger.error(
                    f"Predicted {T} generalized equations from analytic "
                    f"equation but {L} were generated. This will likely "
                    "in a critical error!"
                )
        else:
            logger.info(f"Predicted {L} generalized equations")

    def _initialize_generalized_equations(self, allowed_configs):
        self._generalized_equations = {n: [] for n in allowed_configs.keys()}

        # Generate all possible numbers of phonons consistent with n_max.
        for n_phonons, configs in allowed_configs.items():
            for config in configs:
                self._append_generalized_equation(n_phonons, config.config)

        eq = GreenEquation(model=self._model)
        self._append_master_dictionary(eq)

        # Only one Green's function, with "zero" phonons
        self._generalized_equations[0] = [eq]

        self._determine_unique_dictionary()
        self._predict_total_terms()

    def _initialize_equations(self):
        totals = self._get_total_terms()

        # Initialize the self._equations attribute's lists here since we know
        # all the keys:
        self._equations = {
            key: [] for key in self._generalized_equations.keys()
        }

        # Initialize the full set of equations
        for n_phonons, l_eqs in self._generalized_equations.items():
            for eq in l_eqs:
                n_mat_id = eq.index_term._get_phonon_config_id()
                l_deltas = self._f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = deepcopy(eq)
                    eq_copy._init_full(true_delta)
                    self._equations[n_phonons].append(eq_copy)

        L = sum([len(val) for val in self._equations.values()])

        if L == totals:
            logger.info(f"Generated {L} total equations")
        else:
            logger.error(
                f"Predicted {totals} equations from generalized form but {L} "
                f"were generated. This is likely a bug in the code that will "
                "result in a critical error!"
            )

    def _final_checks(self):
        """Runs a sanity check on the unique keys, which should equal the
        number of equations.
        """

        unique_short_identifiers = set()
        all_terms_rhs = set()
        for n_phonons, equations in self._equations.items():
            for eq in equations:
                unique_short_identifiers.add(eq.index_term.id())
                for term in eq._terms_list:
                    all_terms_rhs.add(term.id())

        if unique_short_identifiers == all_terms_rhs:
            logger.info("Closure checked and valid")
        else:
            # logger.error("Invalid closure!")
            # logger.error(unique_short_identifiers - all_terms_rhs)
            # logger.error(all_terms_rhs - unique_short_identifiers)
            logger.critical("Critical error due to invalid closure.")

    def _save_model(self):
        """Saves the model attribute to disk. Will not overwrite an existing
        file."""

        if self._root is None:
            return

        path = Path(self._root) / Path("model.json")
        m = json.loads(self._model.to_json())
        with open(path, "w") as f:
            json.dump(m, f, indent=4, sort_keys=False)

    def checkpoint(self):
        """Runs the checkpoint protocol to attempt and save the current
        state of the system. This method will only run if a root directory is
        defined at class instantiation. Note that existing checkpoints will
        never be overwritten.

        If the number of equations is larger than a pre-specified cutoff
        (by default 1,000,000), then checkpointing raises a warning and
        does not work.
        """

        if self._root is None:
            logger.warning(
                "root not provided to System - System checkpointing disabled"
            )
            return

        # check how many equations there are -- if too many, checkpointing
        # will be disabled
        try:
            L = sum([len(val) for val in self._equations.values()])
            if L > self._chkpt_lim:
                logger.warning(
                    f"\nMatrices have size {L}, which is larger than the"
                    f" checkpoint limit {self._chkpt_lim:.0f}.\nPickle "
                    f"performance expected to be slow for such large"
                    f" files. System checkpointing is disabled.\n"
                    f"To override the preset, pass a value for chkpt_lim"
                    f" to System object."
                )
                return
        except AttributeError:  # just means no equations generated yet
            pass

        for attr in ["generalized_equations", "f_arg_list", "equations"]:
            obj = eval(f"self._{attr}")
            path = Path(self._root) / Path(f"{attr}.pkl")
            if obj is not None and not path.exists():
                pickle.dump(
                    obj,
                    open(path, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                logger.info(f"Checkpoint saved: {attr}")

    @classmethod
    def from_checkpoint(cls, root):
        """Reloads the state of the System from the saved pickle file
        checkpoint.

        Parameters
        ----------
        root : os.PathLike
            Checkpoint directory
        """

        path = Path(root) / Path("model.json")
        if not path.exists():
            logger.critical(f"Model checkpoint file {path} does not exist")

        with open(path, "r") as f:
            model = Model.from_dict(json.load(f))

        generalized_equations = None
        path = Path(root) / Path("generalized_equations.pkl")
        if path.exists():
            generalized_equations = pickle.load(open(path, "rb"))

        f_arg_list = None
        path = Path(root) / Path("f_arg_list.pkl")
        if path.exists():
            f_arg_list = pickle.load(open(path, "rb"))

        equations = None
        path = Path(root) / Path("equations.pkl")
        if path.exists():
            equations = pickle.load(open(path, "rb"))

        return cls(
            model=model,
            generalized_equations=generalized_equations,
            f_arg_list=f_arg_list,
            equations=equations,
            root=str(root),
        )

    def __init__(
        self,
        model=None,
        generalized_equations=None,
        f_arg_list=None,  # TODO: pool f_arg_list into generalized_equations
        equations=None,
        root=None,
        mpi_comm=None,
        chkpt_lim=1e6,
        autoprime=True,
    ):
        self._root = root
        if self._root is not None:
            Path(root).mkdir(exist_ok=True, parents=True)

        self._model = deepcopy(model)
        self._save_model()

        self._generalized_equations = generalized_equations
        self._f_arg_list = f_arg_list
        self._equations = equations
        self._chkpt_lim = chkpt_lim

        if autoprime:
            # Get all of the allowed configurations - additionally checkpoint
            # the generalized configurations in a root directory if its
            # provided
            self._prime_system()
        else:
            logger.warning(
                "System has not been primed. Load from disk "
                "or prime explicitly with _prime_system."
            )

    def _prime_system(self):
        """Helper function that facilitates the construction of configs,
        generalized and total equations. By default it is called during
        init, but it can be disabled by passing autoprime=False, in case a
        user wants to initialize the system from disk or save initialization
        for later."""
        if self._generalized_equations is None:
            with timeit(logger.debug, "Legal configurations generated"):
                confs = generate_all_legal_configurations(self._model)
            with timeit(logger.debug, "Generalized equations initialized"):
                self._initialize_generalized_equations(confs)
        else:
            logger.info("Generalized equations reloaded from state")
        self.checkpoint()

        if self._equations is None:
            with timeit(logger.debug, "Equations initialized"):
                self._initialize_equations()
        else:
            logger.info("Equations reloaded from state")
        self.checkpoint()

        with timeit(logger.debug, "Final checks"):
            self._final_checks()

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

        eqs_dict = (
            self._generalized_equations if generalized else self._equations
        )
        od = OrderedDict(sorted(eqs_dict.items(), reverse=True))
        for n_phonons, eqs in od.items():
            print(f"{n_phonons}")
            print("-" * 60)
            for eq in eqs:
                eq.visualize(full=full, coef=coef)
            print("\n")

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

        # The basis object maps each unique identifier to a unique number.
        # The local_basis object maps each unique identifier to a unique
        # number within the manifold of some number of phonons.
        basis = dict()

        if full_basis:
            # Set the overall basis. Each unique identifier gets its own .
            cc = 0
            for _, equations in self._equations.items():
                for eq in equations:
                    basis[eq.index_term.id()] = cc
                    cc += 1

        else:
            # Set the local basis, in which each identifier gets its own
            # relative to the n-phonon manifold.
            for n_phonons, list_of_equations in self._equations.items():
                basis[n_phonons] = {
                    eq.index_term.id(): ii
                    for ii, eq in enumerate(list_of_equations)
                }

        return basis
