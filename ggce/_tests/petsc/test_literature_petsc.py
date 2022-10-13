import pytest
import importlib

import numpy as np

from ggce import Model, System

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

EFB_Figure5_k0_params = {
    "gt": np.array(
        [
            [-5.50000000e00, 1.43501419e-03],
            [-5.37975952e00, 3.19572879e-03],
            [-5.25951904e00, 1.41728939e-02],
            [-5.13927856e00, 5.28686138e-01],
            [-5.01903808e00, 8.66237856e-03],
            [-4.89879760e00, 4.80114543e-03],
            [-4.77855711e00, 5.76243081e-03],
            [-4.65831663e00, 1.44579253e-03],
            [-4.53807615e00, 1.24225633e-03],
            [-4.41783567e00, 1.24347394e-03],
            [-4.29759519e00, 1.68881078e-03],
            [-4.17735471e00, 3.54070940e-03],
            [-4.05711423e00, 1.42006278e-02],
            [-3.93687375e00, 4.70721754e00],
            [-3.81663327e00, 1.30010459e-02],
            [-3.69639279e00, 4.29116160e-02],
            [-3.57615230e00, 2.59127687e-02],
            [-3.45591182e00, 8.53183064e-03],
            [-3.33567134e00, 5.75017293e-03],
            [-3.21543086e00, 1.25839520e00],
            [-3.09519038e00, 4.64001429e-03],
            [-2.97494990e00, 2.01461344e-03],
            [-2.85470942e00, 2.05228957e-03],
            [-2.73446894e00, 1.64700276e-03],
            [-2.61422846e00, 1.01781738e-02],
        ]
    ),
    "model_params": dict(
        hopping=0.1, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=1.25,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=2.5,
    ),
    "k": 0.0,
    "eta": 0.005,
}

EFB_Figure6_k0_params = {
    "gt": np.array(
        [
            [-7.91666666e00, 3.96981633e-03],
            [-7.83316633e00, 6.25192528e-03],
            [-7.74966599e00, 1.18474177e-02],
            [-7.66616566e00, 3.38340922e-02],
            [-7.58266532e00, 5.13783654e-01],
            [-7.49916499e00, 1.27178629e-01],
            [-7.41566466e00, 2.54007223e-02],
            [-7.33216432e00, 1.73770867e-02],
            [-7.24866399e00, 2.77627942e-02],
            [-7.16516366e00, 1.60349562e-01],
            [-7.08166332e00, 3.61023120e-01],
            [-6.99816299e00, 3.44109821e-02],
            [-6.91466265e00, 1.82075772e-02],
            [-6.83116232e00, 3.34215377e-02],
            [-6.74766199e00, 1.73988473e-01],
            [-6.66416165e00, 6.86677493e-02],
            [-6.58066132e00, 1.61842703e00],
            [-6.49716098e00, 3.84301974e-02],
            [-6.41366065e00, 2.36407511e-01],
            [-6.33016032e00, 3.28368812e-01],
            [-6.24665998e00, 3.48905558e-01],
            [-6.16315965e00, 2.82901774e-01],
            [-6.07965931e00, 5.00377475e-01],
            [-5.99615898e00, 4.72728395e-01],
            [-5.91265865e00, 8.19103149e-02],
        ]
    ),
    "model_params": dict(
        hopping=3.333333333, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=0.41666666,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=0.83333333,
    ),
    "k": 0.0,
    "eta": 0.005,
}

EFB_Figure5_k1_params = {
    "gt": np.array(
        [
            [-5.50000000e00, 5.70009767e-04],
            [-5.37975952e00, 8.46262556e-04],
            [-5.25951904e00, 1.46302959e-03],
            [-5.13927856e00, 3.46046303e-03],
            [-5.01903808e00, 2.23078513e-02],
            [-4.89879760e00, 4.64096303e-02],
            [-4.77855711e00, 1.20128768e00],
            [-4.65831663e00, 5.14812003e-03],
            [-4.53807615e00, 4.91922895e-02],
            [-4.41783567e00, 2.08925512e-03],
            [-4.29759519e00, 1.61751407e-03],
            [-4.17735471e00, 2.23838506e-03],
            [-4.05711423e00, 3.09670706e-03],
            [-3.93687375e00, 7.18639055e-03],
            [-3.81663327e00, 4.26293073e-02],
            [-3.69639279e00, 5.42547982e-01],
            [-3.57615230e00, 1.28572682e-02],
            [-3.45591182e00, 2.27431234e-02],
            [-3.33567134e00, 9.11500901e-03],
            [-3.21543086e00, 4.00282833e-03],
            [-3.09519038e00, 2.88827163e-03],
            [-2.97494990e00, 3.50607146e-03],
            [-2.85470942e00, 2.76394326e-03],
            [-2.73446894e00, 2.33004830e-03],
            [-2.61422846e00, 4.91466132e-02],
        ]
    ),
    "model_params": dict(
        hopping=0.1, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=1.25,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=2.5,
    ),
    "k": 0.5 * np.pi,
    "eta": 0.005,
}

EFB_Figure6_k1_params = {
    "gt": np.array(
        [
            [-7.91666666e00, 2.32625925e-03],
            [-7.83316633e00, 3.32827825e-03],
            [-7.74966599e00, 5.38888772e-03],
            [-7.66616566e00, 1.10987315e-02],
            [-7.58266532e00, 4.27098467e-02],
            [-7.49916499e00, 5.11242824e00],
            [-7.41566466e00, 3.58186694e-02],
            [-7.33216432e00, 1.42537547e-02],
            [-7.24866399e00, 1.48681498e-02],
            [-7.16516366e00, 3.49786993e-02],
            [-7.08166332e00, 6.92494221e-01],
            [-6.99816299e00, 9.47702144e-02],
            [-6.91466265e00, 2.31839083e-02],
            [-6.83116232e00, 2.76177695e-02],
            [-6.74766199e00, 1.28050998e-01],
            [-6.66416165e00, 4.65735123e-02],
            [-6.58066132e00, 2.54056049e-01],
            [-6.49716098e00, 1.54417793e-01],
            [-6.41366065e00, 2.14314658e-01],
            [-6.33016032e00, 1.01377365e00],
            [-6.24665998e00, 3.29395107e-01],
            [-6.16315965e00, 1.82301008e-01],
            [-6.07965931e00, 3.77814154e-01],
            [-5.99615898e00, 1.21252615e00],
            [-5.91265865e00, 2.16788154e-01],
        ]
    ),
    "model_params": dict(
        hopping=3.333333333, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=0.41666666,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=0.83333333,
    ),
    "k": 0.08333333 * np.pi,
    "eta": 0.005,
}

EFB_Figure5_k2_params = {
    "gt": np.array(
        [
            [-5.50000000e00, 6.56198651e-04],
            [-5.37975952e00, 1.05423340e-03],
            [-5.25951904e00, 2.09222564e-03],
            [-5.13927856e00, 6.80256912e-03],
            [-5.01903808e00, 4.10703772e-01],
            [-4.89879760e00, 1.51019463e-02],
            [-4.77855711e00, 3.47672117e-02],
            [-4.65831663e00, 2.69908752e-03],
            [-4.53807615e00, 1.54922356e-03],
            [-4.41783567e00, 1.13498177e-03],
            [-4.29759519e00, 1.18545604e-03],
            [-4.17735471e00, 2.15536581e-03],
            [-4.05711423e00, 6.08022403e-03],
            [-3.93687375e00, 8.82318049e-02],
            [-3.81663327e00, 5.33174310e-03],
            [-3.69639279e00, 1.47541011e-02],
            [-3.57615230e00, 1.16532005e00],
            [-3.45591182e00, 1.79040206e-02],
            [-3.33567134e00, 6.21414078e-03],
            [-3.21543086e00, 5.07344742e-03],
            [-3.09519038e00, 3.33886584e-01],
            [-2.97494990e00, 5.98140209e-03],
            [-2.85470942e00, 2.86979889e-03],
            [-2.73446894e00, 1.52474837e-03],
            [-2.61422846e00, 1.64323866e-02],
        ]
    ),
    "model_params": dict(
        hopping=0.1, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=1.25,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=2.5,
    ),
    "k": np.pi,
    "eta": 0.005,
}

EFB_Figure6_k2_params = {
    "gt": np.array(
        [
            [-7.91666666e00, 8.08789021e-04],
            [-7.83316633e00, 1.01990773e-03],
            [-7.74966599e00, 1.37558681e-03],
            [-7.66616566e00, 2.08990473e-03],
            [-7.58266532e00, 4.05915173e-03],
            [-7.49916499e00, 1.55149926e-02],
            [-7.41566466e00, 6.93816212e-01],
            [-7.33216432e00, 1.09160246e-02],
            [-7.24866399e00, 5.05862053e-03],
            [-7.16516366e00, 5.41734419e-03],
            [-7.08166332e00, 1.07634751e-02],
            [-6.99816299e00, 7.71070336e-02],
            [-6.91466265e00, 8.73142489e-02],
            [-6.83116232e00, 2.18240309e-02],
            [-6.74766199e00, 2.54721787e-02],
            [-6.66416165e00, 1.07387097e-02],
            [-6.58066132e00, 9.78324311e-02],
            [-6.49716098e00, 4.31866001e-02],
            [-6.41366065e00, 1.50430316e00],
            [-6.33016032e00, 5.20143638e-01],
            [-6.24665998e00, 1.86242364e-01],
            [-6.16315965e00, 9.04862044e-02],
            [-6.07965931e00, 2.73056119e-01],
            [-5.99615898e00, 2.11288425e-01],
            [-5.91265865e00, 1.79665744e-01],
        ]
    ),
    "model_params": dict(
        hopping=3.333333333, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="EdwardsFermionBoson",
        phonon_frequency=0.41666666,
        phonon_extent=3,
        phonon_number=9,
        dimensionless_coupling_strength=0.83333333,
    ),
    "k": 0.16666667 * np.pi,
    "eta": 0.005,
}


@pytest.mark.skipif(not petsc_imported, reason="PETSc not installed")
@pytest.mark.skipif(not mpi4py_imported, reason="mpi4py not installed")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "p",
    [
        EFB_Figure5_k0_params,
        EFB_Figure6_k0_params,
        EFB_Figure5_k1_params,
        EFB_Figure6_k1_params,
        EFB_Figure5_k2_params,
        EFB_Figure6_k2_params,
    ],
)
def test_prb_82_085116_2010(p):
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    gt = p["gt"]
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])

    from ggce.executors.petsc4py.solvers import MassSolverMUMPS

    executor_petsc = MassSolverMUMPS(system=System(model), mpi_comm=COMM)
    w_grid = gt[:, 0]
    A_gt = gt[:, 1]

    results_petsc = executor_petsc.spectrum(p["k"], w_grid, eta=p["eta"])
    results_petsc = (-results_petsc.imag / np.pi).squeeze()
    assert np.allclose(results_petsc, A_gt, atol=ATOL)
