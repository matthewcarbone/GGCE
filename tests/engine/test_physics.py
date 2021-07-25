import numpy as np
import pytest

from ggce.engine import physics


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
    prefactor = -sgn * 1j / np.sqrt(B**2 - (omega - e0)**2)
    t1 = (-x + sgn * 1j * np.sqrt(1.0 - x**2))**np.abs(delta * a)
    return t1 * prefactor


@pytest.mark.parametrize("d", [-1, -2, 0, -10, 100])
@pytest.mark.parametrize("o", [-0.5, 0.5, 0.0, -50.0, 50.0])
@pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("e", [0.1, 0.01, 1e-5, 1e-8])
@pytest.mark.parametrize("t", [0.0, 1.0, 5.0, 50.0])
def test_g0_delta_omega(d, o, a, e, t):
    assert g0_delta_omega(d, o, a, e, t) \
        == physics.g0_delta_omega(d, o, a, e, t)


def G0_k_omega(k, omega, a, eta, tf):
    """This is the empirically correct G0_k_omega. Any changes to the
    function (for e.g. speed improvements) defined in physics should return
    the same result as this function."""

    return 1.0 / (omega + 1j * eta + 2.0 * tf * np.cos(k * a))


@pytest.mark.parametrize("k", [0.0, 0.5 * np.pi, np.pi])
@pytest.mark.parametrize("o", [-0.5, 0.5, 0.0, -50.0, 50.0])
@pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("e", [0.1, 0.01, 1e-8])
@pytest.mark.parametrize("t", [0.0, 1.0, 50.0])
def test_G0_k_omega(k, o, a, e, t):
    assert G0_k_omega(k, o, a, e, t) == physics.G0_k_omega(k, o, a, e, t)
