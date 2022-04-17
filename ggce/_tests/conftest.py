import pytest

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
