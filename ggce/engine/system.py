#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import OrderedDict
import copy
import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import time

from ggce.utils.logger import default_logger as dlog
from ggce.engine.equations import Equation, GreenEquation
from ggce.engine.physics import total_generalized_equations

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

    def __init__(self, system_params):
        """Initializer.

        Parameters
        ----------
        system_params : SystemParameters
            Container for the full set of parameters.
        """

        self.system_params = system_params

        # The number of unique boson types has already been evaluated upon
        # initializing the configuration class
        self.n_boson_types = self.system_params.n_boson_types
        self.max_bosons_per_site = self.system_params.max_bosons_per_site

        self.master_f_arg_list = None

        # The equations are organized by the number of bosons contained
        # in their configuration space.
        self.generalized_equations = dict()
        self.equations = dict()
        self.recursion_solver_basis = dict()
        self.basis = dict()
        self.unique_short_identifiers = set()

        # Total number of bosons allowed is the sum of the number of bosons
        # allowed for each boson type
        self.N_total_max = sum(self.system_params.N)

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
        dlog.info(f"({dt:.02f}s) Generated {L} generalized equations")

        # Need to generalize this
        if self.n_boson_types == 1 and self.max_bosons_per_site is None:
            # Plus one for the Green's function
            T = 1 + total_generalized_equations(
                self.system_params.M, self.system_params.N, self.n_boson_types
            )
            dlog.debug(f"Predicted {T} equations from combinatorics equations")

            assert T == L, f"{T}, {L}"

        totals = self._get_total_terms()
        dlog.debug(f"Predicting {totals} total terms")

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
        dlog.info(f"({dt:.02f}s) Generated {L} equations")

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
        all_terms_rhs = set()
        for n_bosons, equations in self.equations.items():
            for eq in equations:
                self.unique_short_identifiers.add(eq.index_term.identifier())
                for term in eq.terms_list:
                    all_terms_rhs.add(term.identifier())
        dt = time.time() - t0

        if self.unique_short_identifiers == all_terms_rhs:
            dlog.info(f"({dt:.02f}s) Closure is valid")
        else:
            dlog.critical("Invalid closure!")
            print(self.unique_short_identifiers - all_terms_rhs)
            print(all_terms_rhs - self.unique_short_identifiers)
            raise RuntimeError("Invalid closure!")

    def prime_solver(self):
        """Prepares the solver-specific information.

        Recursive method:
            We need a mapping, for a given n_bosons, the identifier and the
        basis index. Note that the basis index resets every time we move to
        a new number of total bosons. This is why organizing the equations
        hierarchically by total bosons was a sensible earlier choice.
        """

        t0 = time.time()
        for n_bosons, equations in self.equations.items():
            self.recursion_solver_basis[n_bosons] = {
                eq.index_term.identifier(): ii
                for ii, eq in enumerate(equations)
            }
        cc = 0
        for _, equations in self.equations.items():
            for eq in equations:
                self.basis[eq.index_term.identifier()] = cc
                cc += 1
        dt = time.time() - t0
        dlog.info(f"({dt:.02f}s) Solvers primed")

    def one_shot_sparse_solve(self, k, w):
        """Executes a oneshot sparse solver. Each row/column corresponds to a
        different f_n(delta) function."""

        meta = {
            'alphas': [],
            'betas': [],
            'inv': [],
            'time': []
        }

        t0_all = time.time()

        # Initialize the sparse matrix to solve
        row_ind = []
        col_ind = []
        dat = []
        total_bosons = np.sum(self.system_params.N)
        for n_bosons in range(total_bosons + 1):
            for eq in self.equations[n_bosons]:
                row_dict = dict()
                index_term_id = eq.index_term.identifier()
                ii_basis = self.basis[index_term_id]
                for term in eq.terms_list + [eq.index_term]:
                    jj = self.basis[term.identifier()]
                    try:
                        row_dict[jj] += term.coefficient(k, w)
                    except KeyError:
                        row_dict[jj] = term.coefficient(k, w)

                row_ind.extend([ii_basis for _ in range(len(row_dict))])
                col_ind.extend([key for key, _ in row_dict.items()])
                dat.extend([value for _, value in row_dict.items()])

        X = coo_matrix((
            np.array(dat, dtype=np.complex64),
            (np.array(row_ind), np.array(col_ind))
        )).tocsr()

        size = (X.data.size + X.indptr.size + X.indices.size) / BYTES_TO_MB

        dlog.debug(f"\tMemory usage of sparse X is {size:.02f} MB")

        # Initialize the corresponding sparse vector
        # {G}(0)
        row_ind = np.array([self.basis['{G}(0.0)']])
        col_ind = np.array([0])
        v = coo_matrix((
            np.array([self.equations[0][0].bias(k, w)], dtype=np.complex64),
            (row_ind, col_ind)
        )).tocsr()

        res = spsolve(X, v)
        G = res[self.basis['{G}(0.0)']]

        if -G.imag / np.pi < 0.0:
            dlog.error(
                f"Negative A({k:.02f}, {w:.02f}): {(-G.imag / np.pi):.02f}"
            )

        dt = time.time() - t0_all
        dlog.debug(f"Sparse matrices solved in {dt:.02f} s")

        meta['time'] = [dt]
        meta['inv'] = [len(self.basis)]

        return G, meta

    def _compute_alpha_beta_(self, n_bosons, n_shift, k, w, mat):
        """Computes the auxiliary matrices alpha_n (n_shift = -1) and beta_n
        (n_shift = 1). Modifies the matrix mat in place. Note that mat should
        be a complex matrix of zeros before being passed to this method."""

        n_bosons_shift = n_bosons + n_shift

        equations_n = self.equations[n_bosons]
        for ii, eq in enumerate(equations_n):
            index_term_id = eq.index_term.identifier()
            ii_basis = self.recursion_solver_basis[n_bosons][index_term_id]

            for term in eq.terms_list:
                if term.get_total_bosons() != n_bosons_shift:
                    continue
                t_id = term.identifier()
                jj_basis = self.recursion_solver_basis[n_bosons_shift][t_id]
                mat[ii_basis, jj_basis] += term.coefficient(k, w)

    def _compute_mat_to_invert(self, n_bosons, k, w, beta_n, A):

        # Fill beta
        t0 = time.time()
        self._compute_alpha_beta_(n_bosons, 1, k, w, beta_n)
        dt = time.time() - t0
        dlog.debug(f"({dt:.02f}s) Filled beta {beta_n.shape}")

        identity = np.eye(beta_n.shape[0], A.shape[1])

        t0 = time.time()
        initial_A_shape = A.shape

        return identity - beta_n @ A, initial_A_shape

    def _log_solve_info(self):
        """Pipes some of the current solving information to the outstream."""

        d = copy.deepcopy(vars(self.system_params))
        d['terms'] = len(d['terms'])
        dlog.debug(f"Solving recursively: {d}")

    def continued_fraction_dense_solve(self, k, w):
        """Executes the solution for some given k, w. Also returns the shapes
        of all computed matrices."""

        t0_all = time.time()

        self._log_solve_info()

        meta = {
            'alphas': [],
            'betas': [],
            'inv': [],
            'time': []
        }

        total_bosons = np.sum(self.system_params.N)

        for n_bosons in range(total_bosons, 0, -1):

            t0 = time.time()
            d_n = len(self.recursion_solver_basis[n_bosons])
            d_n_m_1 = len(self.recursion_solver_basis[n_bosons - 1])

            if n_bosons == total_bosons:
                A = np.zeros((d_n, d_n_m_1), dtype=np.complex64)
                self._compute_alpha_beta_(n_bosons, -1, k, w, A)
                continue

            d_n_p_1 = len(self.recursion_solver_basis[n_bosons + 1])
            alpha_n = np.zeros((d_n, d_n_m_1), dtype=np.complex64)
            meta['alphas'].append((d_n, d_n_m_1))

            beta_n = np.zeros((d_n, d_n_p_1), dtype=np.complex64)
            meta['betas'].append((d_n, d_n_p_1))
            to_inv, initial_A_shape = \
                self._compute_mat_to_invert(n_bosons, k, w, beta_n, A)

            # Fill alpha
            t0 = time.time()
            self._compute_alpha_beta_(n_bosons, -1, k, w, alpha_n)
            dt = time.time() - t0
            dlog.debug(f"({dt:.02f}s) Filled alpha {alpha_n.shape}")

            # This is the rate-limiting step ##################################
            t0 = time.time()
            A = linalg.solve(to_inv, alpha_n)
            dt = time.time() - t0
            ###################################################################

            dlog.debug(
                f"({dt:.02f}s inv) A2 {initial_A_shape} -> A1 {A.shape}"
            )
            meta['inv'].append(to_inv.shape[0])
            meta['time'].append(dt)

        # The final answer is actually A_1. It is related to G via the vector
        # equation V_1 = A_1 G, where G is a scalar. It turns out that the
        # sum over the terms in this final A are actually -Sigma, where
        # Sigma is the self-energy!
        self_energy_times_G0 = 0.0
        A = np.atleast_1d(A.squeeze())
        for term in self.equations[0][0].terms_list:
            basis_idx = self.recursion_solver_basis[1][term.identifier()]
            self_energy_times_G0 += A[basis_idx] * term.coefficient(k, w)

        # The Green's function is of course given by Dyson's equation.
        eom = self.equations[0]
        if len(eom) != 1:
            dlog.critical("More than one EOM!")
            raise RuntimeError("More than one EOM!")
        G0 = eom[0].bias(k, w)  # Convenient way to access G0...
        G = G0 / (1.0 - self_energy_times_G0)

        if -G.imag / np.pi < 0.0:
            dlog.error(
                f"Negative A({k:.02f}, {w:.02f}): {(-G.imag / np.pi):.02f}"
            )

        dt_all = time.time() - t0_all
        meta['time'].append(dt_all)  # Last entry is the total

        dlog.debug(f"({dt_all:.02f}s) Done: G({k:.02f}, {w:.02f})")

        return G, meta

    def solve(self, k, w, solver):
        if solver == 0:
            return self.continued_fraction_dense_solve(k, w)
        elif solver == 1:
            return self.one_shot_sparse_solve(k, w)
        else:
            raise RuntimeError(f"Unknown solver type {solver}")
