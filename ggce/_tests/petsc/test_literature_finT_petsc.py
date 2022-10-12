import pytest

import numpy as np

from ggce import Model, System, SparseSolver

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

H_Figure8a = {
    "gt": np.array(
        [
            [-3.991268516958683, -0.0019082489287027859] ,
            [-3.6356109263386025, 0.004375013917941085] ,
            [-3.2799533357185218, 0.2041568027331613] ,
            [-2.924295745098441, 0.7979252669965866] ,
            [-2.5686381544783607, 0.08094285084119753] ,
            [-2.2129805638582796, 0.11582835458847689] ,
            [-1.857322973238199, 0.6442572879291804] ,
            [-1.5016653826181185, 0.019190868908443987] ,
            [-1.1460077919980376, 0.041721965182285194] ,
            [-0.7903502013779571, 0.23429605331517328] ,
            [-0.43469261075787635, 0.04039872660798197] ,
            [-0.0790350201377957, 0.02220700466294396]
        ]
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.1
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.,
        phonon_extent=3,
        phonon_number=9,
        phonon_extent_tfd=3,
        phonon_number_tfd=5,
        dimensionless_coupling_strength=1.0,
    ),
    "k": 0.,
    "eta": 0.05,
}

H_Figure8c = {
    "gt": [
            [-3.991268516958683, 0.04551739888277512] ,
            [-3.6356109263386025, 0.14035961840052702] ,
            [-3.2799533357185218, 0.13006413366252909] ,
            [-2.924295745098441, 0.845818941694827] ,
            [-2.5686381544783607, 0.12995200068544874] ,
            [-2.2129805638582796, 0.07733165000816758] ,
            [-1.857322973238199, 0.6764986583525406] ,
            [-1.5016653826181185, 0.05985881017017673] ,
            [-1.1460077919980376, 0.05528152999788378] ,
            [-0.7903502013779571, 0.2546393784843073] ,
            [-0.43469261075787635, 0.03650288205185187] ,
            [-0.0790350201377957, 0.023496315901370697]
          ],
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.4
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.,
        phonon_extent=3,
        phonon_number=9,
        phonon_extent_tfd=3,
        phonon_number_tfd=5,
        dimensionless_coupling_strength=1.0,
    ),
    "k": 0.,
    "eta": 0.05,
}

H_Figure8a_q = {
    "gt": np.array(
        [
            [-3.991268516958683, 0.009262084029614925],
            [-3.6356109263386025, 0.02098609320819378],
            [-3.2799533357185218, 0.09703467041254044],
            [-2.924295745098441, 1.0559293031692505],
            [-2.5686381544783607, 0.05456865206360817],
            [-2.2129805638582796, 0.06743030250072479],
            [-1.857322973238199, 0.685399055480957],
            [-1.5016653826181185, 0.03507662191987038],
            [-1.1460077919980376, 0.03647913411259651],
            [-0.7903502013779571, 0.2587514817714691],
            [-0.43469261075787635, 0.046595487743616104],
            [-0.0790350201377957, 0.015414925292134285]
        ],
    ),
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.1
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.,
        phonon_extent=2,
        phonon_number=9,
        phonon_extent_tfd=2,
        phonon_number_tfd=5,
        dimensionless_coupling_strength=1.0,
    ),
    "k": 0.,
    "eta": 0.05,
}

H_Figure8c_q = {
    "gt": [
            [-3.991268516958683, 0.036382149904966354],
            [-3.6356109263386025, 0.04817003011703491],
            [-3.2799533357185218, 0.10950811952352524],
            [-2.924295745098441, 1.5255520343780518],
            [-2.5686381544783607, 0.07808094471693039],
            [-2.2129805638582796, 0.07547001540660858],
            [-1.857322973238199, 0.7808491587638855],
            [-1.5016653826181185, 0.06408513337373734],
            [-1.1460077919980376, 0.05582793802022934],
            [-0.7903502013779571, 0.2490590661764145],
            [-0.43469261075787635, 0.05023759976029396],
            [-0.0790350201377957, 0.027970779687166214]
          ],
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.4
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.,
        phonon_extent=2,
        phonon_number=9,
        phonon_extent_tfd=2,
        phonon_number_tfd=5,
        dimensionless_coupling_strength=1.0,
    ),
    "k": 0.,
    "eta": 0.05,
}

## only run this "precise" test if you have time
## it can take several hours
@pytest.mark.slow
@pytest.mark.skipif(not petsc_imported, reason="PETSc not installed")
@pytest.mark.skipif(not mpi4py_imported, reason="mpi4py not installed")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "p",
    [
        H_Figure8a,
        H_Figure8c
    ],
)
def test_prb_102_165155_2020(p):
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    gt = np.array(p["gt"])
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])

    executor_sparse = MassSolverMUMPS(system=System(model),mpi_comm=COMM)
    w_grid = gt[:, 0]
    A_gt = gt[:, 1]

    results_sparse = executor_sparse.spectrum(p["k"], w_grid, eta=p["eta"], pbar=True)
    results_sparse = (-results_sparse.imag / np.pi).squeeze()

    assert np.allclose(results_sparse, A_gt, atol=ATOL)

## this quick test benchmarks the code against a version with known
## accurate performance (as tested against literature data above)
## but in a small cloud regime -- so the result does not agree with the
## literature exactly, but it would if it was run at higher theory levels
@pytest.mark.skipif(not petsc_imported, reason="PETSc not installed")
@pytest.mark.skipif(not mpi4py_imported, reason="mpi4py not installed")
@pytest.mark.parametrize(
    "p",
    [
        H_Figure8a_q,
        H_Figure8c_q
    ],
)
def test_prb_102_165155_2020_small_cloud(p):
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    gt = np.array(p["gt"])
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])

    executor_sparse = MassSolverMUMPS(system=System(model),mpi_comm=COMM)
    w_grid = gt[:, 0]
    A_gt = gt[:, 1]

    results_sparse = executor_sparse.spectrum(p["k"], w_grid, eta=p["eta"], pbar=True)
    results_sparse = (-results_sparse.imag / np.pi).squeeze()
    assert np.allclose(results_sparse, A_gt, atol=ATOL)
