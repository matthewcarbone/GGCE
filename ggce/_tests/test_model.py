from copy import deepcopy
import math
import pytest

import numpy as np

from ggce.model import model_coupling_map, SingleTerm, Hamiltonian


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
            ("Holstein", np.array([1]), np.array([-1]), "+", 1.23, 0, 4.56),
            ("Holstein", np.array([1]), np.array([-1]), "+", 1.23, 1, -4.56),
            ("Holstein", np.array([1]), np.array([-1]), "-", 1.23, 2, 4.56),
            ("Holstein", np.array([1]), np.array([-1]), "-", 1.23, 3, -4.56),
            ("Peierls", np.array([1]), np.array([-1]), "+", 1.23, 0, 4.56),
            (
                "BondPeierls",
                np.array([1]),
                np.array([-1]),
                "+",
                1.23,
                1,
                -4.56,
            ),
            (
                "EdwardsFermionBoson",
                np.array([1]),
                np.array([-1]),
                "-",
                1.23,
                2,
                4.56,
            ),
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
        st = SingleTerm(
            "Holstein", np.array([1]), np.array([-1]), "+", 1.23, 0, 4.56
        )
        with pytest.raises(SystemExit):
            setattr(st, attribute, value)


class TestHamiltonian:
    @staticmethod
    def test_initializer():
        Hamiltonian()

    @staticmethod
    @pytest.mark.parametrize(
        "coupling_type",
        ["Holstein", "Peierls", "EdwardsFermionBoson", "BondPeierls"],
    )
    @pytest.mark.parametrize("phonon_index", [0, 1, 2, -1])
    def test_add_(coupling_type, phonon_index):
        h = deepcopy(Hamiltonian())
        if phonon_index < 0:
            with pytest.raises(SystemExit):
                h.add_(coupling_type, phonon_index, 1.0, 1.0, None, 1.0)
        else:
            h.add_(coupling_type, phonon_index, 1.0, 1.0, None, 1.0)

            if coupling_type == "Holstein":
                assert len(h.terms) == 2
            elif coupling_type == "EdwardsFermionBoson":
                assert len(h.terms) == 4
            elif coupling_type == "BondPeierls":
                assert len(h.terms) == 4
            elif coupling_type == "Peierls":
                assert len(h.terms) == 8

    @staticmethod
    def test_invalid_coupling_add():
        h = deepcopy(Hamiltonian())
        h.add_("InvalidCoupling", 0, 2.0, 1.0, None, 1.0)
        assert len(h.terms) == 0

    @staticmethod
    @pytest.mark.parametrize(
        "coupling,dimensionless_coupling",
        [(1.0, 1.0), (None, None)],
    )
    def test_XOR_add_(coupling, dimensionless_coupling):
        h = deepcopy(Hamiltonian())
        with pytest.raises(ValueError):
            h.add_(
                coupling_type="Holstein",
                phonon_index=0,
                phonon_frequency=1.0,
                coupling_strength=coupling,
                dimensionless_coupling_strength=dimensionless_coupling,
                coupling_multiplier=1.0,
            )


class TestModel:
    @staticmethod
    def test_visualize(ZeroTemperatureModel):
        ZeroTemperatureModel.visualize()

    @staticmethod
    def test_visualize_after_add(ZeroTemperatureModel):
        ZeroTemperatureModel.add_("Holstein", 1.0, 2, 3)
        ZeroTemperatureModel.visualize()

    @staticmethod
    def test_finite_T_no_M_no_N_1(FiniteTemperatureModel):
        FiniteTemperatureModel.add_(
            coupling_type="Holstein",
            phonon_frequency=2.5,
            phonon_extent=3,
            phonon_number=2,
            phonon_extent_tfd=2,
            phonon_number_tfd=None,
            coupling_strength=1.0,
            dimensionless_coupling_strength=None,
            phonon_index_override=None,
        )
        assert len(FiniteTemperatureModel.phonon_number) == 0
        assert len(FiniteTemperatureModel.phonon_extent) == 0
        assert len(FiniteTemperatureModel.hamiltonian.terms) == 0

    @staticmethod
    def test_finite_T_no_M_no_N_2(FiniteTemperatureModel):
        FiniteTemperatureModel.add_(
            coupling_type="Holstein",
            phonon_frequency=2.5,
            phonon_extent=3,
            phonon_number=2,
            phonon_extent_tfd=None,
            phonon_number_tfd=3,
            coupling_strength=1.0,
            dimensionless_coupling_strength=None,
            phonon_index_override=None,
        )
        assert len(FiniteTemperatureModel.phonon_number) == 0
        assert len(FiniteTemperatureModel.phonon_extent) == 0
        assert len(FiniteTemperatureModel.hamiltonian.terms) == 0

    @staticmethod
    def test_finite_T_no_M_no_N_3(FiniteTemperatureModel):
        FiniteTemperatureModel.add_(
            coupling_type="Holstein",
            phonon_frequency=2.5,
            phonon_extent=3,
            phonon_number=2,
            phonon_extent_tfd=None,
            phonon_number_tfd=None,
            coupling_strength=1.0,
            dimensionless_coupling_strength=None,
            phonon_index_override=None,
        )
        assert len(FiniteTemperatureModel.phonon_number) == 0
        assert len(FiniteTemperatureModel.phonon_extent) == 0
        assert len(FiniteTemperatureModel.hamiltonian.terms) == 0
