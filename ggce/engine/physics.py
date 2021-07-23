__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
from scipy.special import comb


def g0_delta_omega(delta, omega, a, eta, tf, sgn=1.0, e0=0.0):
    """Free Green's function in position/frequency space. Note that the
    frequency is generally a complex variable, corresponding to w + i * eta."""

    if tf == 0.0:
        if delta != 0:
            return 0.0
        return 1.0 / (omega + eta * 1j)

    omega = omega + eta * 1j
    B = 2.0 * np.abs(tf)
    x = (omega - e0) / B
    prefactor = -sgn * 1j / np.sqrt(B**2 - (omega - e0)**2)
    t1 = (-x + sgn * 1j * np.sqrt(1.0 - x**2))**np.abs(delta * a)
    return t1 * prefactor


def G0_k_omega(k, omega, a, eta, tf):
    return 1.0 / (omega + 1j * eta + 2.0 * tf * np.cos(k * a))


def generalized_equations_combinatorics_term(m, n):
    """The total number of generalized equations is given by the exact
    relation

    sum_{m, n = 1}^{M, N} c_{m, n}

    where c_{m, n} is given by this equation, and is equal to

    * 1 if m = 1 or n = 2
    * (m + n - 3) choose (n - 2) otherwise
    """

    if m == 1 or n == 2:
        return 1

    return comb(m + n - 3, n - 2, exact=True)


def total_generalized_equations(M, N, nbt):
    """Gets the total number of generalized equations as predicted by the
    combinatorics equation described in
    generalized_equations_combinatorics_term.
    """

    bosons = [sum([
        sum([
            generalized_equations_combinatorics_term(m, n)
            for n in range(1, N[bt] + 1)
        ]) for m in range(1, M[bt] + 1)
    ]) for bt in range(nbt)]
    return int(np.prod(bosons))
