#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import copy

from ggce.engine import physics
from ggce.engine import terms as terms_module


class Equation:
    """A single equation that describes f_n in terms of other f-functions,
    plus a constant. Note that by definition f_0(0) is the full Green's
    function within the MA.
        The core attributes are the terms and the bias, such that the sum of
    terms equals the bias, which is a simple constant number (really a function
    depending on k and w). The terms will constitute the basis of populating
    the matrix equation to solve, and the entries of that matrix will be the
    callable prefactors, functions of k and w as well.

    Attributes
    ----------
    bias : Callable
        Function of k and w, default is 0. Will be G0(k, w) in the case of the
        free Green's function.
    index_term : FDeltaTerm
        A "pointer", in a sense, to the equation. Allows us to label the
        equation. It contributes additively to the rest of the terms.
    terms : List[FDeltaTerm]
        A list of the remainder of the FDeltaTerms.
    f_arg_terms : Dict[List[int]]
        A dictionary indexed by the _n_mat_identifier containing lists of
        integers highlighting the required values of delta for that
        particular _n_mat_identifier.
    """

    def __init__(self, n_mat_index, config, config_filter=None):
        """Initializer.

        Parameters
        ----------
        n_mat_index : list
            The list of integers representing the f-subscript. In order to be
            a valid f-function, the first and last indexes must be at least
            one. These properties are asserted in the setter for this
            parameter. This means that only allowed f-function subscripts can
            be passed to this function.
        config : InputParameters
            Container for the full set of parameters.
        """

        self.n_mat_index = n_mat_index
        self.f_arg_terms = None
        self.config = config
        self.config_filter = config_filter

    def bias(self, k, w):
        """The default value for the bias is 0, except in the case of the
        Green's function."""

        return 0.0

    def initialize_index_term(self):
        """Initializes the index which is used as a reference to the equation.
        """

        if len(self.n_mat_index) == 1 and self.n_mat_index[0] == 0:
            pass
        else:
            assert self.n_mat_index[0] > 0
            assert self.n_mat_index[-1] > 0

        self.index_term = terms_module.IndexTerm(copy.copy(self.n_mat_index))

    def visualize(self, full=True, coef=None):
        """Prints the representative strings of the terms for both the indexer
        and terms. This allows the user to at a high level, and
        without coefficients, visualize all the terms in the equation.
        If coef is not None, we assume we want to visualize the coefficient
        too, and coef is a 2-vector containing the k and w points."""

        id1 = self.index_term.identifier(full=full)
        if coef is not None:
            id1 += f" -> {self.index_term.coefficient(*coef)}"
        print(id1)
        for term in self.terms_list:
            id1 = term.identifier(full=full)
            if coef is not None:
                id1 += f" -> {term.coefficient(*coef)}"
            print("\t", id1)

    def _populate_f_arg_terms(self):
        """Populates the required f_arg terms for the 'rhs' (in self.terms_list)
        that will be needed for the non-generalized equations later.
        Specifically, populates a dictionary all_delta_terms which maps the
        f_vec index string to a list of integers, which correspond to the
        needed delta values for that f_vec string. These dictionaries will be
        combined later during production of the non-generalized f-functions."""

        self.f_arg_terms = dict()
        for ii, term in enumerate(self.terms_list):
            n_mat_identifier = term._get_n_mat_identifier()
            try:
                self.f_arg_terms[n_mat_identifier].append(term.f_arg)
            except KeyError:
                self.f_arg_terms[n_mat_identifier] = [term.f_arg]

    def init_full(self, delta):
        """Increments every term in the term list's g-argument by the provided
        value. Also sets the f_arg to the proper value for the index term."""

        self.index_term.set_f_arg(delta)
        for ii in range(len(self.terms_list)):
            self.terms_list[ii].increment_g_arg(delta)

    def initialize_terms(self):
        """Systematically construct the generalized annihilation terms on the
        rhs of the equation."""

        self.terms_list = []
        assert all([n >= 0 for n in self.n_mat_index])
        N = sum(self.n_mat_index)

        # Iterate over all possible types of the coupling operators
        for v_term in self.config.terms:

            # We separate the two cases for creation and annihilation operators
            # on the boson operator 'b'
            if v_term.dagger == '-':

                # Iterate over the site indexes for the term's alpha-index
                for gamma_idx, nval in enumerate(self.n_mat_index):

                    # Do not annihilate the vacuum, this term comes to 0
                    # anyway.
                    if nval == 0:
                        continue

                    constant_prefactor = v_term.sign * self.config.g * nval

                    t = terms_module.AnnihilationTerm(
                        copy.copy(self.n_mat_index), hterm=v_term,
                        constant_prefactor=constant_prefactor
                    )
                    t.step(gamma_idx)

                    # Check the filter for the new configuration, which may not
                    # be legal, since we might have added/removed a boson in an
                    # illegal place based on the filter parameters.
                    if not self.config_filter(t.n_mat):
                        continue

                    t.check_if_green_and_simplify()
                    self.terms_list.append(t)

            # Don't want to create an equation corresponding to more than
            # the maximum number of allowed bosons.
            elif v_term.dagger == '+' and N + 1 <= self.config.N:
                for gamma_idx in range(
                    len(self.n_mat_index) - self.config.M, self.config.M
                ):

                    constant_prefactor = v_term.sign * self.config.g
                    t = terms_module.CreationTerm(
                        copy.copy(self.n_mat_index), hterm=v_term,
                        constant_prefactor=constant_prefactor
                    )
                    t.step(gamma_idx)

                    if not self.config_filter(t.n_mat):
                        continue

                    t.check_if_green_and_simplify()

                    self.terms_list.append(t)


class GreenEquation(Equation):
    """Specific equation corresponding to the Green's function."""

    def __init__(self, config):
        super().__init__(n_mat_index=[0], config=config)

    def bias(self, k, w):
        """Initializes the bias for the Green's function."""

        return physics.G0_k_omega(
            k, w, self.config.a, self.config.eta, self.config.t
        )

    def initialize_terms(self):
        """Override for the Green's function other terms."""

        # Only instance where we actually use the base FDeltaTerm class
        self.terms_list = []

        # Iterate over all possible types of the coupling operators
        for v_term in self.config.terms:

            # Only boson creation terms contribute to the Green's function
            # EOM since annihilation terms hit the vacuum and yield zero.
            if v_term.dagger == '+':
                n = [1]
                t = terms_module.EOMTerm(n_mat=n, hterm=v_term)
                self.terms_list.append(t)
