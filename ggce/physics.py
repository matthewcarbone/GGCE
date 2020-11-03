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


def mgf_sum_rule(w, s, order):
    return np.sum(s[1:] * w[1:]**order * np.diff(w))


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
