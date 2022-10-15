import pytest

import numpy as np

from ggce import Model, System, DenseSolver

ATOL = 1.0e-6
TEPS = 1.0e-6

model_h = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=2,
        dimensionless_coupling_strength=0.6,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}

model_p = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=2.35,
        phonon_extent=2,
        phonon_number=2,
        dimensionless_coupling_strength=1.3,
    ),
    "k": 1.46,
    "eta": 0.5,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}

model_hp = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=2,
        dimensionless_coupling_strength=0.6,
    ),
    "model_add_params2": dict(
        coupling_type="Peierls",
        phonon_frequency=0.69,
        phonon_extent=2,
        phonon_number=2,
        dimensionless_coupling_strength=2.3,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}

model_ht = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=TEPS
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.6,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}

model_pt = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=TEPS
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=2.35,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=1.3,
    ),
    "k": 1.46,
    "eta": 0.5,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}

model_hpt = {
    "model_params": dict(
        hopping=1.0, phonon_max_per_site=None, temperature=TEPS
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=0.6,
    ),
    "model_add_params2": dict(
        coupling_type="Peierls",
        phonon_frequency=0.69,
        phonon_extent=2,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=2,
        dimensionless_coupling_strength=2.3,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt",
}


@pytest.mark.parametrize(
    "p",
    [
        [model_h, model_ht],
        [model_p, model_pt],
        [model_hp, model_hpt],
    ],
)
def test_zero_vs_tiny_T(p):

    k = p[0]["k"]
    w = np.linspace(-2, 0, 10)

    # Check the true T=0 case
    model = Model.from_parameters(**p[0]["model_params"])

    model.add_(**p[0]["model_add_params"])
    # accommodate double-model case
    try:
        model.add_(**p[0]["model_add_params2"])
    except KeyError:
        pass

    solver = DenseSolver(System(model))
    results = solver.greens_function(k, w, eta=p[0]["eta"]).squeeze()

    # Check the true T=epsilon case
    model = Model.from_parameters(**p[1]["model_params"])

    model.add_(**p[1]["model_add_params"])
    # accommodate double-model case
    try:
        model.add_(**p[1]["model_add_params2"])
    except KeyError:
        pass

    solver = DenseSolver(System(model))
    results_t = solver.greens_function(k, w, eta=p[1]["eta"]).squeeze()

    assert np.allclose(results, results_t, atol=ATOL)
