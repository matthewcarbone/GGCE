import pytest
import numpy as np

from ggce import Model, System, MassSolverMUMPS
from ggce.utils.utils import process_dispersion

mpi4py_imported = False
try:
    from mpi4py import MPI

    mpi4py_imported = True
except ImportError:
    pass

petsc_imported = False
try:
    from ggce.executors.petsc4py.solvers import MassSolverMUMPS

    petsc_imported = True
except ImportError:
    pass

ATOL = 1.0e-4

H_disp = {
    "disp": np.array(
        [
            -2.13357591,
            -2.11002857,
            -2.04152528,
            -1.93577551,
            -1.81115636,
            -1.70275774,
            -1.63772221,
            -1.60489404,
            -1.58675713,
            -1.57505631,
            -1.56689429,
            -1.56031997,
            -1.55514015,
            -1.55080771,
            -1.54719531,
            -1.54424225,
            -1.54177714,
            -1.54015774,
            -1.53918069,
            -1.53886704
         ],
    ),
    "lftm": np.array(
        [
            0.05002659,
            0.05002811,
            0.05004279,
            0.05009543,
            0.05026857,
            0.05079536,
            0.05166395,
            0.05223035,
            0.0526713 ,
            0.05322507,
            0.05320764,
            0.05365926,
            0.05371972,
            0.0538172 ,
            0.05391855,
            0.05401722,
            0.05443246,
            0.05444023,
            0.05444205,
            0.05441458
        ],
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=0.5,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.2,
    ),
    "w0": -2.5,
    "eta": 0.05,
    "kgrid": np.linspace(0, np.pi, 20),
}


Ht_disp = {
    "disp": np.array(
        [
        -2.27032010,
        -2.24710519,
        -2.17884402,
        -2.07004391,
        -1.92889763,
        -1.76884135,
        -1.60981514,
        -1.47473396,
        -1.37678952,
        -1.31205636,
        -1.26949333,
        -1.24038372,
        -1.21935644,
        -1.20362145,
        -1.19155228,
        -1.18237100,
        -1.17545341,
        -1.17069893,
        -1.16788926,
        -1.16691047  
        ],
    ),
    "lftm": np.array(
        [
            0.05003133,
            0.05003679,
            0.05006176,
            0.05001756,
            0.05003599,
            0.0500623 ,
            0.05011613,
            0.05022575,
            0.0503448 ,
            0.05048927,
            0.05063826,
            0.05064263,
            0.0507317 ,
            0.0507624 ,
            0.05082355,
            0.05077456,
            0.05085719,
            0.05081329,
            0.05077591,
            0.05085204
        ],
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.1
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.0,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.3,
    ),
    "w0": -2.5,
    "eta": 0.05,
    "kgrid": np.linspace(0, np.pi, 20),
}

P_disp = {
    "disp": np.array(
        [
            -2.05221829,
            -2.02962257,
            -1.96366601,
            -1.86098486,
            -1.73851190,
            -1.63596423,
            -1.58668088,
            -1.56751988,
            -1.55763702,
            -1.55055595,
            -1.54450427,
            -1.53879446,
            -1.53337647,
            -1.52834307,
            -1.52374767,
            -1.51939479,
            -1.51433297,
            -1.12053083,
            -1.12487661,
            -1.12842012
        ],
    ),
    "lftm": np.array(
        [
            0.05000364,
            0.05000494,
            0.05001221,
            0.05003807,
            0.05021303,
            0.05112328,
            0.05284144,
            0.05389852,
            0.05450736,
            0.05511493,
            0.05560135,
            0.05633197,
            0.05737958,
            0.05891222,
            0.06126965,
            0.06522925,
            0.07291950,
            0.20392106,
            0.18979730,
            0.17680745
        ],
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=0.5,
        phonon_extent=4,
        phonon_number=3,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.2,
    ),
    "w0": -2.5,
    "eta": 0.05,
    "kgrid": np.linspace(0, np.pi, 20),
}

Pt_disp = {
    "disp": np.array(
        [
            -2.03860534,
            -2.01424822,
            -1.94216566,
            -1.82544492,
            -1.66981736,
            -1.48539322,
            -1.29253027,
            -1.13655662,
            -1.06103682,
            -1.03309347,
            -1.02002084,
            -1.01205506,
            -1.00665565,
            -1.00202791,
            -0.99794541,
            -0.99394416,
            -0.98742742,
            -0.08718754,
            -0.08316406,
            -0.08190966
        ],
    ),
    "lftm": np.array(
        [
            0.05000570,
            0.05000590,
            0.05000530,
            0.05000756,
            0.05001251,
            0.05002866,
            0.05011567,
            0.05076352,
            0.05263697,
            0.05473189,
            0.05657045,
            0.05836157,
            0.05977918,
            0.06213563,
            0.06537793,
            0.07031304,
            0.08309610,
            0.39414916,
            0.39360121,
            0.38676333
        ],
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.1
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=1.0,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.1,
    ),
    "w0": -2.5,
    "eta": 0.05,
    "kgrid": np.linspace(0, np.pi, 20),
}


@pytest.mark.skipif(not petsc_imported, reason="PETSc not installed")
@pytest.mark.skipif(not mpi4py_imported, reason="mpi4py not installed")
@pytest.mark.parametrize(
    "p",
    [H_disp, P_disp, 
     Ht_disp, Pt_disp],
)
def test_dispersion_finiteT(p):
    disp = np.array(p["disp"])
    lftm = np.array(p["lftm"])
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])

    kgrid = p["kgrid"]
    w0 = p["w0"]
    eta = p["eta"]

    COMM = MPI.COMM_WORLD
    executor_petsc = MassSolverMUMPS(System(model), mpi_comm=COMM)
    results = executor_petsc.dispersion(kgrid, w0, eta=eta, nmax=1000, \
            eta_step_div=10, peak_routine="scipy", next_k_offset_factor=2)
    dispersion, lifetimes = process_dispersion(results)

    assert np.allclose(disp, dispersion, atol=ATOL)

    assert np.allclose(lftm, lifetimes, atol=ATOL)

