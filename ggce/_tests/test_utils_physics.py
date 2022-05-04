import pytest

import numpy as np

from ggce.utils import physics


def g0_delta_omega(delta, omega, a, eta, tf, sgn=1.0, e0=0.0):
    """This is the empirically correct g0_delta_omega. Any changes to the
    function (for e.g. speed improvements) defined in physics should return
    the same result as this function."""

    if tf == 0.0:
        if delta != 0:
            return 0.0
        return 1.0 / (omega + eta * 1j)

    omega = omega + eta * 1j
    B = 2.0 * np.abs(tf)
    x = (omega - e0) / B
    prefactor = -sgn * 1j / np.sqrt(B**2 - (omega - e0) ** 2)
    t1 = (-x + sgn * 1j * np.sqrt(1.0 - x**2)) ** np.abs(delta * a)
    return t1 * prefactor


@pytest.mark.parametrize("d", [-1, 100])
@pytest.mark.parametrize("o", [-0.5, -50.0, 50.0])
@pytest.mark.parametrize("e", [0.1, 1e-8])
@pytest.mark.parametrize("t", [0.0, 50.0])
def test_g0_delta_omega(d, o, e, t):
    assert g0_delta_omega(d, o, 1.0, e, t) == physics.g0_delta_omega(
        d, o, 1.0, e, t
    )


def G0_k_omega(k, omega, a, eta, tf):
    """This is the empirically correct G0_k_omega. Any changes to the
    function (for e.g. speed improvements) defined in physics should return
    the same result as this function."""

    epsilon_k = -2.0 * tf * np.cos(k * a)
    return 1.0 / (omega + 1j * eta - epsilon_k)


@pytest.mark.parametrize("k", [0.0, 0.5 * np.pi, np.pi])
@pytest.mark.parametrize("o", [-0.5, -50.0, 50.0])
@pytest.mark.parametrize("e", [0.1, 1e-8])
@pytest.mark.parametrize("t", [0.0, 50.0])
def test_G0_k_omega(k, o, e, t):
    assert G0_k_omega(k, o, 1.0, e, t) == physics.G0_k_omega(k, o, 1.0, e, t)
