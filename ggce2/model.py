from collections import namedtuple
import numpy as np
import math

from ggce.utils.logger import Logger


def model_coupling_map(coupling_type, t, Omega, lam):
    """Returns the value for g, the scalar that multiplies the coupling in the
    Hamiltonian. Converts the user-input lambda value to this g. Uses
    pre-defined values for the dimensionless coupling to get g for a variety
    of pre-defined default models.

    Parameters
    ----------
    coupling_type : str
        The desired coupling type. Can be Holstein, Peierls, BondPeierls, or
        EdwardsFermionBoson
    t : float
        The hopping strength.
    Omega : float
        The (Einsten) boson frequency.
    lam : float
        The dimensionless coupling.

    Returns
    -------
    float
        The value for the coupling (g).

    Raises
    ------
    RuntimeError
        If an unknown coupling type is provided.
    """

    if coupling_type == 'Holstein':
        return math.sqrt(2.0 * t * Omega * lam)
    elif coupling_type == 'EdwardsFermionBoson':
        return lam
    elif coupling_type == 'Peierls':
        return math.sqrt(t * Omega * lam / 2.0)
    elif coupling_type == 'BondPeierls':
        return math.sqrt(t * Omega * lam)
    else:
        raise RuntimeError(f"Unknown coupling_type type {coupling_type}")
