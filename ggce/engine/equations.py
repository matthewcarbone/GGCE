import copy
import numpy as np

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

    def __init__(self, config_index, system_params):
        """Initializer.

        Parameters
        ----------
        config_index : np.ndarray
            The list of integers representing the f-subscript. In order to be
            a valid f-function, the first and last indexes must be at least
            one. These properties are asserted in the setter for this
            parameter. This means that only allowed f-function subscripts can
            be passed to this function.
        system_params : SystemParams
            Container for the full set of parameters.
        """

        self.config_index = config_index
        self.f_arg_terms = None
        self.system_params = system_params

    def bias(self, k, w, eta=None):
        """The default value for the bias is 0, except in the case of the
        Green's function."""

        return 0.0

    def initialize_index_term(self):
        """Initializes the index which is used as a reference to the equation.
        """

        # Pass over the Green's function
        if self.config_index.shape[1] == 1 and np.sum(self.config_index) == 0:
            pass
        else:
            assert np.any(self.config_index[:, 0]) > 0
            assert np.any(self.config_index[:, -1]) > 0

        self.index_term = terms_module.IndexTerm(copy.copy(self.config_index))

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
                c = term.coefficient(*coef)
                if c.imag < 0:
                    id1 += f" -> {c.real:.02e} - {-c.imag:.02e}i"
                else:
                    id1 += f" -> {c.real:.02e} + {c.imag:.02e}i"
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
            n_mat_identifier = term._get_boson_config_identifier()
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

    def initialize_terms(self, config_space_generator):
        """Systematically construct the generalized annihilation terms on the
        rhs of the equation."""

        ae = self.system_params.absolute_extent

        self.terms_list = []
        assert np.all(self.config_index >= 0)

        # Iterate over all possible types of the coupling operators
        for hterm in self.system_params.terms:

            # Boson type
            bt = hterm.bt

            # We separate the two cases for creation and annihilation operators
            # on the boson operator 'b'
            if hterm.d == '-':

                # Iterate over the site indexes for the term's alpha-index
                for loc, nval in enumerate(self.config_index[bt, :]):

                    # Do not annihilate the vacuum, this term comes to 0
                    # anyway.
                    if nval == 0:
                        continue

                    t = terms_module.AnnihilationTerm(
                        copy.copy(self.config_index), hterm=hterm,
                        system_params=self.system_params.get_fFunctionInfo(),
                        constant_prefactor=hterm.g * nval
                    )
                    t.step(loc)
                    t.check_if_green_and_simplify()
                    if t.config.is_zero():
                        self.terms_list.append(t)
                    elif config_space_generator.is_legal(t.config.config):
                        self.terms_list.append(t)

            # Don't want to create an equation corresponding to more than
            # the maximum number of allowed bosons.
            elif hterm.d == '+':
                for loc in range(self.config_index.shape[1] - ae, ae):

                    t = terms_module.CreationTerm(
                        copy.copy(self.config_index), hterm=hterm,
                        system_params=self.system_params.get_fFunctionInfo(),
                        constant_prefactor=hterm.g
                    )
                    t.step(loc)
                    t.check_if_green_and_simplify()

                    if t.config.is_zero():
                        self.terms_list.append(t)
                    elif config_space_generator.is_legal(t.config.config):
                        self.terms_list.append(t)


class GreenEquation(Equation):
    """Specific equation corresponding to the Green's function."""

    def __init__(self, system_params):
        config = np.zeros((system_params.n_boson_types, 1))
        super().__init__(config_index=config, system_params=system_params)

    def bias(self, k, w, eta=None):
        """Initializes the bias for the Green's function."""

        if eta is None:
            eta = self.system_params.eta

        return physics.G0_k_omega(
            k, w, self.system_params.a, eta, self.system_params.t
        )

    def initialize_terms(self):
        """Override for the Green's function other terms."""

        # Only instance where we actually use the base FDeltaTerm class
        self.terms_list = []

        # Iterate over all possible types of the coupling operators
        for hterm in self.system_params.terms:

            # Only boson creation terms contribute to the Green's function
            # EOM since annihilation terms hit the vacuum and yield zero.
            if hterm.d == '+':
                n = np.zeros((self.system_params.n_boson_types, 1)).astype(int)
                n[hterm.bt, :] = 1
                t = terms_module.EOMTerm(
                    boson_config=n, hterm=hterm,
                    system_params=self.system_params.get_fFunctionInfo()
                )
                self.terms_list.append(t)
