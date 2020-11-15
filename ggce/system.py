#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import copy
import numpy as np
from scipy import linalg
import time

from ggce.utils import utils
from ggce.utils.logger import default_logger as dlog
from ggce.equations import Equation, GreenEquation
from ggce.physics import total_generalized_equations


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

    def _log_configuration(self):
        """Pipes configuration information to the outstream in debug mode."""

        dlog.debug("Initializing System class")
        for key, value in vars(self.config).items():
            if key == 'terms':
                dlog.debug(f"n terms = {len(value)}")
            else:
                dlog.debug(f"{key}: {value}")

    def __init__(self, config):
        """Initializer.

        Parameters
        ----------
        config : InputParameters
            Container for the full set of parameters.
        """

        self.config = config
        filter_type = None if self.config.config_filter == 'no_filter' \
            else self.config.config_filter
        self.config_filter = utils.ConfigFilter(filter_type=filter_type)
        self._log_configuration()

        # The number of unique boson types has already been evaluated upon
        # initializing the configuration class
        self.n_boson_types = 1  # config.n_unique_bosons

        self.master_f_arg_list = None

        # The equations are organized by the number of bosons contained
        # in their configuration space.
        self.generalized_equations = dict()
        self.equations = dict()
        self.recursion_solver_basis = dict()
        self.basis = dict()
        self.unique_short_identifiers = set()

    def _ma_sanity_check(self, n):
        return int(7 * n + 1 + 5 * n * (n - 1) / 2)

    def _append_generalized_equation(self, n_bosons, n_mat):

        eq = Equation(
            n_mat, config=self.config, config_filter=self.config_filter
        )

        # Initialize the index term, basically the f_n(delta)
        eq.initialize_index_term()

        # Initialize the specific terms which constitute the f_n(delta)
        # EOM, such as f_n'(1), f_n''(0), etc.
        eq.initialize_terms()

        # Initialize the required values for delta (arguments of f) as
        # derived from the terms themselves, used later when generating
        # the specific equations.
        eq._populate_f_arg_terms()

        # Append a master dictionary at the System object level that
        # keeps track of all the f_arg values required for each value of
        # n_mat.
        self._append_master_dictionary(eq)

        # Finally, append the equation to a master list of generalized
        # equations/
        try:
            self.generalized_equations[n_bosons].append(eq)
        except KeyError:
            self.generalized_equations[n_bosons] = [eq]

    def _extend_legal_configs(self, n_bosons, n_mats):
        """Appends a new configuration to the dictionary legal_n_mats."""

        for n_mat in n_mats:
            self._append_generalized_equation(n_bosons, n_mat)

    def initialize_generalized_equations(self):
        """Starting with values for the order of the momentum approximation
        and the maximum allowed number of bosons, this method generates a list
        of n_mat lists, consisting of all possible legal contributions,
        meaning, all vectors [n0, n1, ..., n(ma_order - 1)] such that the
        sum of all of the entries equals n, where n = [1, ..., n_max]."""

        t0 = time.time()

        # Special case: we can count the total terms and compare it to the
        # Berciu paper to be sure we have the correct n_mat terms calculated
        berciu_condition = self.config.M == 3 and 'EFB' in self.config.model \
            and self.n_boson_types == 1

        if berciu_condition:
            _totals = 0
            _b_totals = 0

        # Generate all possible numbers of bosons consistent with n_max.
        for n_bosons in range(1, self.config.N + 1):

            # Iterate over all possible lengths consistent with the MA. For
            # each boson type in the range, construct the allowed permutations.
            for z in range(1, self.config.M + 1):

                z_n_configuration_space = \
                    list(utils.configuration_space_generator(z, n_bosons))

                # Now, we get all of the LEGAL nvecs, which are those in
                # which at least a single boson of any type is on both
                # edges of the cloud.
                tmp_legal_configs = [
                    nn for nn in z_n_configuration_space
                    if self.config_filter(nn)
                ]

                # Berciu special case, track based on the analysis in her
                # paper
                if berciu_condition:
                    if z == 1:
                        _totals += len(tmp_legal_configs) * 8
                    elif z == 2:
                        _totals += len(tmp_legal_configs) * 7
                    elif z == 3:
                        _totals += len(tmp_legal_configs) * 5

                self._extend_legal_configs(n_bosons, tmp_legal_configs)

            if self.config.M == 3 and 'EFB' in self.config.model:
                _b_totals += self._ma_sanity_check(n_bosons)

        if berciu_condition:
            dlog.debug(
                f"Special case: MA(3) Berciu terms {_totals} vs {_b_totals} "
                "(predicted vs truth)"
            )

        # Manually append the Green's function (do the same as above except)
        # for this special case. Note that no matter what this is always
        # neded, but the form of the GreenEquation EOM will differ depending
        # on the system type.
        eq = GreenEquation(config=self.config)
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

        # Plus one for the Green's function
        T = int(total_generalized_equations(self.config.M, self.config.N) + 1)
        dlog.debug(f"Predicted {T} equations from combinatorics equations")

        assert T == L

        totals = self._get_total_terms()
        dlog.debug(f"Predicting {totals} total terms")

        # Initialize the self.equations attribute's lists here since we know
        # all the keys:
        for key, _ in self.generalized_equations.items():
            self.equations[key] = []

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
                n_mat_id = eq.index_term._get_n_mat_identifier()
                l_deltas = self.master_f_arg_list[n_mat_id]
                for true_delta in l_deltas:
                    eq_copy = copy.deepcopy(eq)
                    eq_copy.init_full(true_delta)
                    self.equations[n_bosons].append(eq_copy)

        dt = time.time() - t0

        L = sum([len(val) for val in self.equations.values()])
        dlog.info(f"({dt:.02f}s) Generated {L} equations")

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
        for n_bosons, eqs in eqs_dict.items():
            print(f"{n_bosons}")
            print("-" * 60)
            for eq in eqs:
                eq.visualize(full=full, coef=coef)
            print("\n")

    def visualize_filter(self, M, N):
        self.config_filter.visualize(M, N)

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
        dt = time.time() - t0
        dlog.info(f"({dt:.02f}s) Recursive solver primed")

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
                if term.n_bosons != n_bosons_shift:
                    continue
                t_id = term.identifier()
                jj_basis = self.recursion_solver_basis[n_bosons_shift][t_id]
                mat[ii_basis, jj_basis] += term.coefficient(k, w)

    def _log_solve_info(self):
        """Pipes some of the current solving information to the outstream."""

        d = copy.deepcopy(vars(self.config))
        d['terms'] = len(d['terms'])
        dlog.debug(f"Solving recursively: {d}")

    def solve(self, k, w):
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

        for n_bosons in range(self.config.N, 0, -1):

            t0 = time.time()
            d_n = len(self.recursion_solver_basis[n_bosons])
            d_n_m_1 = len(self.recursion_solver_basis[n_bosons - 1])

            if n_bosons == self.config.N:
                A = np.zeros((d_n, d_n_m_1), dtype=np.complex64)
                self._compute_alpha_beta_(n_bosons, -1, k, w, A)
                continue

            d_n_p_1 = len(self.recursion_solver_basis[n_bosons + 1])
            alpha_n = np.zeros((d_n, d_n_m_1), dtype=np.complex64)
            meta['alphas'].append((d_n, d_n_m_1))

            # Fill alpha
            t0 = time.time()
            self._compute_alpha_beta_(n_bosons, -1, k, w, alpha_n)
            dt = time.time() - t0
            dlog.debug(f"({dt:.02f}s) Filled alpha {alpha_n.shape}")

            beta_n = np.zeros((d_n, d_n_p_1), dtype=np.complex64)
            meta['betas'].append((d_n, d_n_p_1))

            # Fill beta
            t0 = time.time()
            self._compute_alpha_beta_(n_bosons, 1, k, w, beta_n)
            dt = time.time() - t0
            dlog.debug(f"({dt:.02f}s) Filled beta {beta_n.shape}")

            # Expensive...
            identity = np.eye(beta_n.shape[0], A.shape[1])

            t0 = time.time()
            initial_A_shape = A.shape
            A = linalg.inv(identity - beta_n @ A) @ alpha_n
            dt = time.time() - t0
            dlog.debug(f"({dt:.02f}s) A2 {initial_A_shape} -> A1 {A.shape}")
            meta['inv'].append(identity.shape[0])
            meta['time'].append(time.time() - t0)

        # The final answer is actually A_1. It is related to G via the vector
        # equation V_1 = A_1 G, where G is a scalar. It turns out that the
        # sum over the terms in this final A are actually -Sigma, where
        # Sigma is the self-energy!
        self_energy_times_G0 = 0.0
        for term in self.equations[0][0].terms_list:
            basis_idx = self.recursion_solver_basis[1][term.identifier()]
            self_energy_times_G0 += \
                A.squeeze()[basis_idx] * term.coefficient(k, w)

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

        dlog.debug(f"({dt_all:.02f}s) Done: G({k:.02f}, {w:.02f})")

        return G, meta
