from copy import deepcopy, copy
import numpy as np

from ggce import logger
from ggce.utils import physics
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
    f_arg_terms : TYPE
        Description
    terms_list : list
        Description

    Deleted Attributes
    ------------------
    bias : Callable
        Function of k and w, default is 0. Will be G0(k, w) in the case of the
        free Green's function.
    terms : List[FDeltaTerm]
        A list of the remainder of the FDeltaTerms.
    config_index : TYPE
        Description

    index_term : FDeltaTerm
        A "pointer", in a sense, to the equation. Allows us to label the
        equation. It contributes additively to the rest of the terms.
    system_params : TYPE
        Description
    """

    @classmethod
    def from_config(cls, config, model):
        """Initializes the :class:`Equation` class from a numpy array and a
        :class:`ggce.models.Model`.

        Parameters
        ----------
        config : numpy.array
            The array representing the configuration of phonons.
        model : ggce.models.Model
            Container for the full set of parameters. Contains all Hamiltonian
            and parameter information about the terms, phonons, couplings, etc.

        Returns
        -------
        Equation
        """

        index_term = terms_module.IndexTerm(config.copy())
        return cls(index_term, model)

    @property
    def index_term(self):
        """The index term/config referencing the lhs of the equation.

        Returns
        -------
        ggce.engine.terms.IndexTerm
        """

        return self._index_term

    @property
    def model(self):
        """Container for the full set of parameters. Contains all Hamiltonian
        and parameter information about the terms, phonons, couplings, etc.

        Returns
        -------
        ggce.model.Model
            Description
        """

        return self._model

    @property
    def f_arg_terms(self):
        """A dictionary of lists of ints indexed by the config string
        representation containing lists of integers highlighting the required
        values of delta for that particular config string representation.

        Returns
        -------
        dict
        """

        return self._f_arg_terms

    def __init__(self, index_term, model, f_arg_terms=None, terms_list=None):
        self._index_term = deepcopy(index_term)
        self._model = deepcopy(model)
        self._f_arg_terms = deepcopy(f_arg_terms)
        self._terms_list = deepcopy(terms_list)

    def bias(self, k, w, eta):
        """The default value for the bias is 0, except in the case of the
        Green's function."""

        return 0.0

    def visualize(self, full=True, coef=None):
        """Prints the representative strings of the terms for both the indexer
        and terms. This allows the user to at a high level, and
        without coefficients, visualize all the terms in the equation.
        If coef is not None, we assume we want to visualize the coefficient
        too, and coef is a 2-vector containing the k and w points."""

        id1 = self.index_term.id(full=full)
        if coef is not None:
            id1 += f" -> {self.index_term.coefficient(*coef)}"
        print(id1)
        if self._terms_list is not None:
            for term in self._terms_list:
                id1 = term.id(full=full)
                if coef is not None:
                    c = term.coefficient(*coef)
                    if c.imag < 0:
                        id1 += f" -> {c.real:.02e} - {-c.imag:.02e}i"
                    else:
                        id1 += f" -> {c.real:.02e} + {c.imag:.02e}i"
                print("\t", id1)

    def _populate_f_arg_terms(self):
        """Populates the required f_arg terms for the 'rhs' (in self._terms_list)
        that will be needed for the non-generalized equations later.
        Specifically, populates a dictionary all_delta_terms which maps the
        f_vec index string to a list of integers, which correspond to the
        needed delta values for that f_vec string. These dictionaries will be
        combined later during production of the non-generalized f-functions."""

        if self._f_arg_terms is not None:
            logger.error("f_args_terms is already initialized")
            return

        self._f_arg_terms = dict()
        for ii, term in enumerate(self._terms_list):
            n_mat_identifier = term._get_boson_config_identifier()
            try:
                self._f_arg_terms[n_mat_identifier].append(term.f_arg)
            except KeyError:
                self._f_arg_terms[n_mat_identifier] = [term.f_arg]

    def init_full(self, delta):
        """Increments every term in the term list's g-argument by the provided
        value. Also sets the f_arg to the proper value for the index term."""

        self.index_term.set_f_arg(delta)
        for ii in range(len(self._terms_list)):
            self._terms_list[ii].increment_g_arg(delta)

    def initialize_terms(self):
        """Systematically construct the generalized annihilation terms on the
        rhs of the equation."""

        if self._terms_list is not None:
            logger.error("terms_list is already initialized")
            return

        ae = self._model.phonon_absolute_extent
        n_spatial_dimensions = self._index_term.config.n_spatial_dimensions
        self._terms_list = []

        # Iterate over all possible types of the coupling operators
        for hterm in self._model.hamiltonian.terms:

            bt = hterm.phonon_index  # Phonon type
            arr = self.index_term.config.config[bt, ...]

            # We separate the two cases for creation and annihilation operators
            # on the boson operator 'b'
            if hterm.dag == "-":

                # Iterate over the site indexes for the term's alpha-index
                for loc, nval in np.ndenumerate(arr):

                    # Do not annihilate the vacuum, this term comes to 0
                    # anyway.
                    if nval == 0:
                        continue

                    t = terms_module.AnnihilationTerm(
                        copy(self._index_term.config.config),
                        hamiltonian_term=hterm,
                        model=deepcopy(self._model),
                        constant_prefactor=hterm.coupling * nval,
                    )
                    t.step_(*loc)
                    t.check_if_green_and_simplify_()
                    t.config.validate()
                    self._terms_list.append(t)

            # Don't want to create an equation corresponding to more than
            # the maximum number of allowed phonons.
            elif hterm.dag == "+":

                if any([xx - ae > 0 for xx in self._index_term.config.shape]):
                    logger.critical(
                        f"Absolute extent {ae} cannot be smaller than any "
                        "config spatial dimension: "
                        f"{self._index_term.config.shape[1:]}"
                    )

                # Create a dummy array to make iterating over indexes easier
                dummy = np.empty(
                    [
                        2 * ae - self._index_term.config.shape[ii + 1]
                        for ii in range(n_spatial_dimensions)
                    ]
                )

                for ii, (index, _) in enumerate(np.ndenumerate(dummy)):
                    loc = tuple(
                        [
                            index[jj]
                            - ae
                            + self._index_term.config.shape[jj + 1]
                            for jj in range(n_spatial_dimensions)
                        ]
                    )

                    t = terms_module.CreationTerm(
                        copy(self._index_term.config.config),
                        hamiltonian_term=hterm,
                        model=deepcopy(self._model),
                        constant_prefactor=hterm.coupling,
                    )
                    t.step_(*loc)
                    t.check_if_green_and_simplify_()
                    t.config.validate()
                    self._terms_list.append(t)


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
        self._terms_list = []

        # Iterate over all possible types of the coupling operators
        for hterm in self.system_params.terms:

            # Only boson creation terms contribute to the Green's function
            # EOM since annihilation terms hit the vacuum and yield zero.
            if hterm.d == "+":
                n = np.zeros((self.system_params.n_boson_types, 1)).astype(int)
                n[hterm.bt, :] = 1
                t = terms_module.EOMTerm(
                    boson_config=n,
                    hterm=hterm,
                    system_params=self.system_params.get_fFunctionInfo(),
                )
                self._terms_list.append(t)
