from ggce.model import Model
from ggce.executors.serial import SerialDenseExecutor


def test_basis_save_load_state():

    model = Model()
    model.set_parameters(hopping=0.1)
    model.add_coupling(
        "Holstein", Omega=1.25, M=3, N=9,
        dimensionless_coupling=2.5
    )
    executor_sparse = SerialDenseExecutor(model, "debug")
    executor_sparse.prime()
