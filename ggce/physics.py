#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np


def g0_delta_omega(delta, omega, a, eta, tf, sgn=1.0, e0=0.0):
    """Free Green's function in position/frequency space. Note that the
    frequency is generally a complex variable, corresponding to w + i * eta."""

    omega = omega + eta * 1j
    B = 2.0 * np.abs(tf)
    x = (omega - e0) / B
    prefactor = -sgn * 1j / np.sqrt(B**2 - (omega - e0)**2)
    t1 = (-x + sgn * 1j * np.sqrt(1.0 - x**2))**np.abs(delta * a)
    return t1 * prefactor


def G0_k_omega(k, omega, a, eta, tf):
    return 1.0 / (omega + 1j * eta + 2.0 * tf * np.cos(k * a))
