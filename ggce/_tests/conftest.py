import pytest

import numpy as np

from ..model import Model as _Model


@pytest.fixture
def ZeroTemperatureModel():
    return _Model.from_parameters(
        hopping=1.0,
        lattice_constant=1.0,
        temperature=0.0,
        phonon_max_per_site=None,
        dimension=1,
    )


@pytest.fixture
def FiniteTemperatureModel():
    return _Model.from_parameters(
        hopping=1.0,
        lattice_constant=1.0,
        temperature=0.5,
        phonon_max_per_site=None,
        dimension=1,
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

## add this so slow tests are skipped by default
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
