"""All functions corresponding to physical quantities are located here."""

import numpy as np


def g0_delta_omega(delta, omega, a, eta, tf, sgn=1.0, e0=0.0):
    """Free Green's function in position/frequency space. Note that the
    frequency is generally a complex variable, corresponding to w + i * eta.

    Parameters
    ----------
    delta : int
        The site-index (FT of the momentum space vector).
    omega : float
        Real frequency.
    a : float
        Lattice constant.
    eta : float
        Broadening.
    tf : float
        Hopping.
    sgn, e0 : float
        Extra terms to allow the overall equation to be more general (do not
        change these!).

    Returns
    -------
    float
        The (complex) value of g0(delta, omega).
    """

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
    """The free particle Green's function in k, omega space for the free
    particle on a lattice.

    Parameters
    ----------
    k : float
        The momentum quantum number.
    omega : float
        Real frequency.
    a : float
        Lattice constant.
    eta : float
        Broadening.
    tf : float
        Hopping.

    Returns
    -------
    float
        The (complex) value of G0(k, omega).
    """

    return 1.0 / (omega + 1j * eta + 2.0 * tf * np.cos(k * a))
