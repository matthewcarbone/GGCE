"""Utilities corresponding to physical quantities."""

import numpy as np
from scipy.special import comb


def g0_delta_omega(
    delta, omega, lattice_constant, broadening, hopping, sgn=1.0, e0=0.0
):
    """Free Green's function in position/frequency space. Note that the
    frequency is a complex variable, corresponding to
    :math:`\\omega + i \\eta`. This function is equation 21 in this
    `PRB <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.035106>`_.
    The equation is

    .. math::

        g_0(\\delta, \\omega) = -i \\frac{\\left[-\\omega_\\eta / 2t + i
        \\sqrt{1 - (\\omega_\\eta/2t)^2}\\right]^{|\\delta|}}{\\sqrt{4t^2 -
        \\omega_\\eta^2}}

    where :math:`\\omega_\\eta = \\omega + i \\eta`. The broadening is
    :math:`\\eta`, the hopping :math:`t`, and the lattice constant :math:`a`.

    .. warning::
        The parameters `sgn` and `e0` are not used in the GGCE formalism, but
        are explained in E. N. Economou, Green’s Functions in Quantum Physics
        (Springer-Verlag, Berlin, 1983).

    Parameters
    ----------
    delta : float
        The site index variable.
    omega : float
        The complex frequency variable
    lattice_constant : float
        The lattice constant: distance between neighboring sites.
    broadening : float
        The artificial broadening term.
    hopping : float
        The hopping strength.
    sgn : float, optional
        Optional sign parameter. It is recommended that this remain unchanged
        from the default. Default is 1.
    e0 : float, optional
        Optional E0 parameter. It is recommended that this remain unchanged
        from the default. Default is 1.

    Returns
    -------
    float
        Complex value of :math:`g_0`.
    """

    a = lattice_constant

    if hopping == 0.0:
        if delta != 0:
            return 0.0
        return 1.0 / (omega + broadening * 1j)

    omega = omega + broadening * 1j
    B = 2.0 * np.abs(hopping)
    x = (omega - e0) / B
    prefactor = -sgn * 1j / np.sqrt(B**2 - (omega - e0) ** 2)
    t1 = (-x + sgn * 1j * np.sqrt(1.0 - x**2)) ** np.abs(delta * a)
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

    bosons = [
        sum(
            [
                sum(
                    [
                        generalized_equations_combinatorics_term(m, n)
                        for n in range(1, N[bt] + 1)
                    ]
                )
                for m in range(1, M[bt] + 1)
            ]
        )
        for bt in range(nbt)
    ]
    return int(np.prod(bosons))

    # total = 0
    # for nn in range(1, N + 1):
    #     for mm in range(1, M + 1):
    #         if nn == 1:
    #             if mm == 1:
    #                 total += 1
    #         elif nn == 2:
    #             total += 1
    #         elif nn >= 2 and mm == 1:
    #             total += 1
    #         else:
    #             total += binom(mm + nn - 3, nn - 2)
    # return total


def mgf_sum_rule(w, s, order):
    return np.sum(s[1:] * w[1:] ** order * np.diff(w))


def holstein_sum_rule_check(w, s, config):
    """Imports the wgrid (w), spectrum (s) and config and produces a summary
    of the sum rules."""

    ek = -2.0 * config.t * np.cos(config.k * config.a)
    g = config.g

    print("Sum rules ratios: (computed / analytic)")

    # First sum rule, area under curve is 1:
    s0 = mgf_sum_rule(w, s, 0)
    print(f"\t#0: {s0:.04f}")

    s1 = mgf_sum_rule(w, s, 1) / ek
    print(f"\t#1: {s1:.04f}")

    s2_ana = ek**2 + g**2
    s2 = mgf_sum_rule(w, s, 2) / s2_ana
    print(f"\t#2: {s2:.04f}")

    s3_ana = ek**3 + 2.0 * g**2 * ek + g**2 * config.Omega
    s3 = mgf_sum_rule(w, s, 3) / s3_ana
    print(f"\t#3: {s3:.04f}")
