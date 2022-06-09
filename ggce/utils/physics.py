"""Utilities corresponding to physical quantities."""

import numpy as np


def g0_delta_omega(
    delta, omega, lattice_constant, broadening, hopping, sgn=1.0, e0=0.0
):
    """The free particle Green's function in position/frequency space. Note
    that the frequency is a complex variable, corresponding to
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

        The parameters ``sgn`` and ``e0`` are not used in the GGCE formalism,
        but are explained in `E. N. Economou, Green’s Functions in Quantum
        Physics (Springer-Verlag, Berlin, 1983)`.

    Additionally, when :math:`t=0`, there are two cases. If
    :math:`\\delta = 0`, then the result is analytically 0. Otherwise, the
    result ends up being :math:`(\\omega + i\\eta)^{-1}`.

    Parameters
    ----------
    delta : float
        The site index variable
    omega : complex
        The complex frequency variable
    lattice_constant : float
        The lattice constant: distance between neighboring sites
    broadening : float
        The artificial broadening term
    hopping : float
        The hopping strength
    sgn : float, optional
        Optional sign parameter. It is recommended that this remain unchanged
        from the default. Default is 1.
    e0 : float, optional
        Optional E0 parameter. It is recommended that this remain unchanged
        from the default. Default is 1.

    Returns
    -------
    complex
        Value of :math:`g_0`.
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


def G0_k_omega(k, omega, lattice_constant, eta, hopping):
    """The free particle Green's function on a 1D lattice in standard
    momentum-frequency space. The equation is

    .. math::

        G_0(k, \\omega) = \\frac{1}{\\omega + i\\eta - \\varepsilon_k}

    where :math:`\\varepsilon_k = -2t \\cos(ka)`.

    Parameters
    ----------
    k : float
        The momentum variable
    omega : complex
        The complex frequency variable.
    lattice_constant : float
        The lattice constant: distance between neighboring sites
    broadening : float
        The artificial broadening term
    hopping : float
        The hopping strength

    Returns
    -------
    complex
        Value of :math:`G_0`.
    """

    return 1.0 / (
        omega + 1j * eta + 2.0 * hopping * np.cos(k * lattice_constant)
    )
