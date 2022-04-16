import math
import pytest

import numpy as np

from ..model import model_coupling_map, SingleTerm


@pytest.mark.parametrize(
    "coupling_type,t,Omega,lam",
    [
        ("Holstein", 1.2345, 4.5678, 5.678),
        ("Holstein", 1.2345, -4.5678, 5.8),
        ("EdwardsFermionBoson", 1.2345, 4.5678, 5.678),
        ("Peierls", 1.2345, 4.5678, 5.678),
        ("BondPeierls", 1.2345, 4.5678, 5.678),
    ],
)
def test_model_coupling_map(coupling_type, t, Omega, lam):

    # This ground truth value is just copied from a version of the
    # model_coupling_map that is known to work
    def ground_truth(coupling_type, t, Omega, lam):
        if coupling_type == "Holstein":
            return math.sqrt(2.0 * t * np.abs(Omega) * lam)
        elif coupling_type == "EdwardsFermionBoson":
            return lam
        elif coupling_type == "Peierls":
            return math.sqrt(t * np.abs(Omega) * lam / 2.0)
        elif coupling_type == "BondPeierls":
            return math.sqrt(t * np.abs(Omega) * lam)
        else:
            raise RuntimeError

    assert model_coupling_map(coupling_type, t, Omega, lam) == ground_truth(
        coupling_type, t, Omega, lam
    )


class TestSingleTerm:
    @staticmethod
    @pytest.mark.parametrize(
        "coupling_type,psi,phi,dag,coupling,phonon_index,phonon_frequency",
        [
            ("Holstein", 1, -1, "+", 1.23, 0, 4.56),
            ("Holstein", 1, -1, "+", 1.23, 1, -4.56),
            ("Holstein", 1, -1, "-", 1.23, 2, 4.56),
            ("Holstein", 1, -1, "-", 1.23, 3, -4.56),
            ("Peierls", 1, -1, "+", 1.23, 0, 4.56),
            ("BondPeierls", 1, -1, "+", 1.23, 1, -4.56),
            ("EdwardsFermionBoson", 1, -1, "-", 1.23, 2, 4.56),
        ],
    )
    def test_initializer(
        coupling_type, psi, phi, dag, coupling, phonon_index, phonon_frequency
    ):
        SingleTerm(*[value for value in locals().values()])

    @staticmethod
    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("psi", "hi"),
            ("phi", "hi"),
            ("dag", "not + or -"),
            ("dag", 1.0),
            ("dag", -2),
            ("coupling", "hi"),
            ("coupling", [1.0, -2.0, 3.0]),
            ("phonon_index", "hi"),
            ("phonon_index", -1),
            ("phonon_frequency", "hi"),
        ],
    )
    def test_setters_raises(attribute, value):
        st = SingleTerm("Holstein", 1, -1, "+", 1.23, 0, 4.56)
        with pytest.raises(AssertionError):
            setattr(st, attribute, value)
