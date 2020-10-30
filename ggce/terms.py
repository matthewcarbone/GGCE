#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import cmath

from ggce import physics


class Term:
    """Base class for a single term like f_n(d).

    Attributes
    ----------
    exp_shift : int
        Factor which multiplies the entire final prefactor term. It is given
        by e^(1j*k*a*exp_shift). This term is determined by the relations
        between the different f-functions.
            1) f_{0, 0, ..., 0, n, m}(d) = f_{n, m, ...}(d + y) * e^{-ikay}
               where y is the number of indexes equal to zero before the first
               nonzero term, and is positive.
            2) f_{n, ..., m, 0, ..., 0} = f_{n, ..., m}
            3) b_gamma^dagger f_{n, m, ...}(d) =
               f_{1, 0, ..., 0, n , m}(0) * e^{ikay}; gamma < 0
               where y is equal to abs(gamma) and is positive.
            4) b_gamma^dagger f_{n, m, ...}(d) =
               f_{1, 0, ..., 0, n , m}(d - y) * e^{ikay}; gamma < 0
    constant_prefactor : float
        The constant prefactor component. Usually consists of the coupling
        strength, sign of the coupling and other prefactor terms such as
        n_gamma. Default is 1.0.
    f_arg : int
        The value of f_arg, where we have f_{n_mat}(f_arg). Note that if
        None, this implies that the term for f_arg is left general. This
        should only occur for an FDeltaTerm object that is an "index" for
        the equation, i.e., it is a term that originally appears on the
        lhs. During generation of the specific equations, this will be set
        to a "true delta" value and that equation appended to the master
        list. Default is None, implying the f-argument is a general delta.
    g_arg : int
        The same as for f_arg, but for the green's function delta term.

    Parameters
    ----------
    n_mat : list
        The n-"matrix". This vector of vectors is indexed in the following
        way. For example, n[ii][jj] means that on boson of type ii, and
        site jj there are n[ii][jj] bosons.
    hterm : SHT
        The parameters of the term itself. (Single Hamiltonian Term)
    """

    def __init__(self, n_mat, hterm=None):
        self.n_mat = list(n_mat)
        self.n_bosons = sum(self.n_mat)

        # If hterm is none, we know this is an index term which carries with
        # it no system-dependent prefactors or anything like that.
        self.hterm = hterm
        self.constant_prefactor = 1.0

        # If True, will prevent any accidental modification of the n_mat. This
        # flag is set to True after incrementing/decrementing a position on the
        # boson chain.
        self.lock_n_mat = False

        # Determines the argument of f and prefactors
        self.exp_shift = 0  # exp(i*k*a*exp_shift)
        self.f_arg = None  # Only None for the index term
        self.g_arg = None  # => g(...) = 1

    def _get_n_mat_identifier(self):
        """Returns a string of the n_mat identifier."""

        if len(self.n_mat) == 1 and self.n_mat[0] == 0:
            return "{G}"
        return "{" + str(self.n_mat)[1:-1] + "}"

    def _get_f_arg_identifier(self):
        """Returns a string of the f_arg identifier."""

        return "(%i)" % self.f_arg if self.f_arg is not None else "(!)"

    def _get_g_arg_identifier(self):
        """Returns a string of the f_arg identifier."""

        return "<%i>" % self.g_arg if self.g_arg is not None else "<!>"

    def _get_c_exp_identifier(self):
        """Returns a string of the current value of the exponential shift."""

        return "[%i]" % self.exp_shift if self.exp_shift is not None else "[!]"

    def identifier(self, full=False):
        """Returns a string with which one can index the term. The string takes
        the form n0-n1-...-nL-1-(f_arg)-[exp_shift]. Also defines the
        individual identifiers for the n_mat, f_arg and c_exp shift, which will
        be utilized later.

        Parameters
        ----------
        full : bool
            If true, returns also the g and exp shift terms with the
            identifier. This is purely for visualization and not for e.g.
            running any calculations, since those terms are pooled into the
            coefficient.
        """

        t1 = self._get_n_mat_identifier() + self._get_f_arg_identifier()
        if not full:
            return t1
        t2 = self._get_g_arg_identifier() + self._get_c_exp_identifier()
        return t1 + t2

    def update_n_mat(self):
        """Specific update scheme needs to be run depending on whether we add
        or remove a boson."""

        raise NotImplementedError

    def modify_n_bosons(self):
        """By default, does nothing, will only be defined for the terms
        in which we add or subtract a boson."""

        return

    def coefficient(self, k, w):
        raise NotImplementedError

    def increment_g_arg(self, delta):
        self.g_arg += delta

    def set_f_arg(self, val):
        raise NotImplementedError

    def check_if_green_and_simplify(self):
        """There are cases when the algorithm produces results that look like
        e.g. {G}(1)[0]. This corresponds to f_0(1), which is actually just
        a Green's function times a phase factor. The precise relation is
        f_0(delta) = e^{i * k * delta * a} G(k, w). We need to simplify this
        back to a term like {G}(0)[0], as if we don't, when the specific
        equations are generated, it will think there are multiple Greens
        equations, of which of course there can be only one."""

        if self._get_n_mat_identifier() == '{G}' and self.f_arg != 0:
            self.exp_shift += self.f_arg
            self.f_arg = 0


class IndexTerm(Term):
    """A term that corresponds to an index of the equation. Critical
    differences between this and the base class include that the f_arg object
    is set to None by default, the gamma_idx is set to None (since we will
    not be removing or adding bosons to this term) and the constant_prefactor
    is set to 1, with the lambdafy_prefactor method disabled, and the
    default, unchangeable prefactor being equal to the constant prefactor,
    independent of k and w."""

    def __init__(self, n_mat):
        super().__init__(n_mat=n_mat)
        self.lock_n_mat = True

    def set_f_arg(self, val):
        """Overrides the set value of f_arg and re-initializes the
        identifier."""

        assert self.hterm is None
        self.f_arg = val

    def increment_g_arg(self, delta):
        raise NotImplementedError

    def coefficient(self, k, w):
        return 1.0


class EOMTerm(Term):
    """A special instance of the base class that is part of the EOM of the
    Green's function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hterm is not None
        self.lock_n_mat = True
        self.exp_shift = self.hterm.x - self.hterm.y
        self.f_arg = self.hterm.y
        self.constant_prefactor = self.hterm.sign * self.hterm.model_params.g

    def coefficient(self, k, w):
        """This will set the prefactor to G0, since we assume
        that as a base class it originates from the g0 terms in the main
        EOM. Note that an index term does not have this method defined, since
        that terms prefactor should always be 1 (set by default to the
        constant prefactor), and this method will be overridden by
        AnnihilationTerm and CreationTerm classes."""

        return self.constant_prefactor * physics.G0_k_omega(
            k, w, self.hterm.model_params.a, self.hterm.model_params.eta,
            self.hterm.model_params.t
        ) * cmath.exp(1j * k * self.hterm.model_params.a * self.exp_shift)

    def increment_g_arg(self, delta):
        return


class _NonIndexTerm(Term):

    def __init__(self, n_mat, hterm, constant_prefactor):
        super().__init__(n_mat=n_mat, hterm=hterm)
        assert self.hterm is not None

        # This is entirely general now
        self.f_arg = self.hterm.y
        self.g_arg = self.hterm.x - self.hterm.y
        self.initial_bosons = sum(self.n_mat)
        self.constant_prefactor = constant_prefactor

    def step(self, gamma_idx):
        """Increments or decrements the bosons on the chain depending on the
        class derived class type."""

        assert not self.lock_n_mat
        self.g_arg += gamma_idx
        self.f_arg -= gamma_idx
        self.modify_n_bosons(gamma_idx)
        self.update_n_mat(gamma_idx)
        self.lock_n_mat = True

    def coefficient(self, k, w):

        freq_shift = self.hterm.model_params.Omega * self.initial_bosons

        exp_term = cmath.exp(
            1j * k * self.hterm.model_params.a * self.exp_shift
        )

        w_freq_shift = w - freq_shift

        g_contrib = physics.g0_delta_omega(
            self.g_arg, w_freq_shift,
            self.hterm.model_params.a,
            self.hterm.model_params.eta,
            self.hterm.model_params.t
        )

        return self.constant_prefactor * exp_term * g_contrib


class AnnihilationTerm(_NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons(self, gamma_idx):
        """Removes a boson from the specified site corresponding to the
        correct type. Note this step is independent of the reductions of the
        f-functions to other f-functions"""

        self.n_mat[gamma_idx] -= 1
        assert all([n >= 0 for n in self.n_mat])

    def update_n_mat(self, gamma_idx):
        """Specific updater for the boson annihilation terms."""

        if len(self.n_mat) == 1 and self.n_mat[0] == 0:  # Green's function
            pass
        else:
            locs_where_neq_zero = [
                ii for ii, n in enumerate(self.n_mat) if n > 0
            ]
            right = max(locs_where_neq_zero)
            left = min(locs_where_neq_zero)
            if right < len(self.n_mat) - 1:
                self.n_mat = self.n_mat[:right + 1]
            if left > 0:
                self.n_mat = self.n_mat[left:]
                self.exp_shift -= left
                self.f_arg += left

        self.n_bosons = self.initial_bosons - 1


class CreationTerm(_NonIndexTerm):
    """Specific object for containing f-terms in which we must subtract
    a boson from a specified site."""

    def modify_n_bosons(self, gamma_idx):
        """This is done for the user in update_n_mat."""

        if gamma_idx > len(self.n_mat) - 1:
            shape1 = gamma_idx + 1 - len(self.n_mat)

            # Create the array to append to the right of the n-matrix
            to_concat = [0 for _ in range(shape1)]

            # The last entry (column-wise) at the alpha'th index (row-wise)
            # gets the boson
            to_concat[-1] = 1
            self.n_mat = self.n_mat + to_concat

        elif gamma_idx < 0:
            shape1 = abs(gamma_idx)
            to_concat = [0 for _ in range(shape1)]
            to_concat[0] = 1
            self.n_mat = to_concat + self.n_mat

        else:
            self.n_mat[gamma_idx] += 1

    def update_n_mat(self, gamma_idx):
        """Handles the boson creation cases, since these can be a bit
        confusing. Basically, we have the possibility of creating a boson on
        a site that is outside of the range of self.n_mat. So we need methods
        of handling the case when an IndexError would've otherwise been raised
        during an attempt to increment an index that isn't there."""

        if gamma_idx < 0:
            self.exp_shift = abs(gamma_idx)
            self.f_arg -= abs(gamma_idx)

        self.n_bosons = self.initial_bosons + 1
