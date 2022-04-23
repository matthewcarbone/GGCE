import pytest

import numpy as np

from ..model import Hamiltonian
from ..model import Model as _Model


@pytest.fixture
def ZeroTemperatureModel():
    return _Model(
        hopping=1.0,
        lattice_constant=1.0,
        temperature=0.0,
        hamiltonian=Hamiltonian(),
        phonon_max_per_site=None,
        phonon_extent=[],
        phonon_number=[],
        phonon_absolute_extent=None,
        n_phonon_types=0,
    )


@pytest.fixture
def FiniteTemperatureModel():
    return _Model(
        hopping=1.0,
        lattice_constant=1.0,
        temperature=0.5,
        hamiltonian=Hamiltonian(),
        phonon_max_per_site=None,
        phonon_extent=[],
        phonon_number=[],
        phonon_absolute_extent=None,
        n_phonon_types=0,
    )


@pytest.fixture
def Random1dPhononArray():
    ptype1 = (np.random.random(size=(4,)) + 1.5).astype(int)
    ptype2 = (np.random.random(size=(4,)) + 2.5).astype(int)
    ptype3 = (np.random.random(size=(4,)) + 3.5).astype(int)
    return np.array([ptype1, ptype2, ptype3])


@pytest.fixture
def Random2dPhononArray():
    ptype1 = (np.random.random(size=(4, 5)) + 1.5).astype(int)
    ptype2 = (np.random.random(size=(4, 5)) + 2.5).astype(int)
    ptype3 = (np.random.random(size=(4, 5)) + 3.5).astype(int)
    return np.array([ptype1, ptype2, ptype3])


@pytest.fixture
def Random3dPhononArray():
    ptype1 = (np.random.random(size=(4, 5, 6)) + 1.5).astype(int)
    ptype2 = (np.random.random(size=(4, 5, 6)) + 2.5).astype(int)
    ptype3 = (np.random.random(size=(4, 5, 6)) + 3.5).astype(int)
    return np.array([ptype1, ptype2, ptype3])
