import pytest

import numpy as np
import shutil, os, json
from pathlib import Path
import pickle

from ggce.logger import _testing_mode
from ggce import Model, System, SparseSolver

model_h = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=4,
        phonon_number=7,
        dimensionless_coupling_strength=0.6,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt"
}

model_p = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=2.35,
        phonon_extent=3,
        phonon_number=4,
        dimensionless_coupling_strength=1.3,
    ),
    "k": 1.46,
    "eta": 0.5,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt"
}

model_hp = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=0.0
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=4,
        dimensionless_coupling_strength=0.6,
    ),
    "model_add_params2": dict(
        coupling_type="Peierls",
        phonon_frequency=0.69,
        phonon_extent=2,
        phonon_number=3,
        dimensionless_coupling_strength=2.3,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt"
}

model_ht = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=0.6
    ),
    "model_add_params": dict(
        coupling_type="Holstein",
        phonon_frequency=1.23,
        phonon_extent=2,
        phonon_number=3,
        phonon_extent_tfd=2,
        phonon_number_tfd=4,
        dimensionless_coupling_strength=0.6,
    ),
    "k": 0.46,
    "eta": 0.005,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt"
}

model_pt = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=1.3
    ),
    "model_add_params": dict(
        coupling_type="Peierls",
        phonon_frequency=2.35,
        phonon_extent=3,
        phonon_number=2,
        phonon_extent_tfd=2,
        phonon_number_tfd=4,
        dimensionless_coupling_strength=1.3,
    ),
    "k": 1.46,
    "eta": 0.5,
    "root_sys": "sys_chkpt",
    "root_res": "res_chkpt"
}

model_hpt = {
    "model_params": dict(
        hopping=1., phonon_max_per_site=None, temperature=2.1
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
    "root_res": "res_chkpt"
}

@pytest.mark.parametrize(
    "p",
    [
        model_h,
        model_p,
        model_hp,
        model_ht,
        model_pt,
        model_hpt,
    ],
)
def test_sys_chkpt(p):
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])
    # accommodate double-parameter case
    try:
        model.add_(**p["model_add_params2"])
    except KeyError:
        pass

    root = p["root_sys"]
    sys = System(model=model,root=root)
    sys.checkpoint()

    sys_disk = System.from_checkpoint(root)

    assert sys._f_arg_list == sys_disk._f_arg_list

    # iterate through the tree of equations to extract unique identifiers
    # compare identifiers between the two Systems
    unique_short_identifiers = set()
    all_terms_rhs = set()
    for n_phonons, equations in sys._equations.items():
        for eq in equations:
            unique_short_identifiers.add(eq.index_term.id())
            for term in eq._terms_list:
                all_terms_rhs.add(term.id())

    unique_short_identifiers_disk = set()
    all_terms_rhs_disk = set()
    for n_phonons, equations in sys_disk._equations.items():
        for eq in equations:
            unique_short_identifiers_disk.add(eq.index_term.id())
            for term in eq._terms_list:
                all_terms_rhs_disk.add(term.id())

    assert unique_short_identifiers == all_terms_rhs_disk

    shutil.rmtree(root)

@pytest.mark.parametrize(
    "p",
    [
        model_h,
        model_p,
        model_hp,
        model_ht,
        model_pt,
        model_hpt,
    ],
)
def test_model_chkpt(p):
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])
    # accommodate double-parameter case
    try:
        model.add_(**p["model_add_params2"])
    except KeyError:
        pass
    root = p["root_sys"]
    sys = System(model=model,root=root,autoprime=False)

    with open( os.path.join(root, "model.json"), "rb") as f:
        model_disk = json.load(f)

    model = json.loads(model.to_json()) # convert to str for comparison

    assert model == model_disk

    shutil.rmtree(root)

@pytest.mark.parametrize(
    "p",
    [
        model_h,
        model_p,
        model_hp,
        model_ht,
        model_pt,
        model_hpt,
    ],
)
def test_res_chkpt(p):
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])
    # accommodate double-parameter case
    try:
        model.add_(**p["model_add_params2"])
    except KeyError:
        pass

    root = p["root_res"]
    executor = SparseSolver(System(model),root=root)
    w_grid = np.linspace(-3,-2,10)

    results = executor.spectrum(p["k"], w_grid, eta=p["eta"]).squeeze()

    # load results from disk and confirm
    results_disk = []
    for w in w_grid:
        pathstr = executor._k_omega_eta_to_str(p["k"], w, p["eta"])
        ckpt_path = f"{pathstr}.pkl"
        path = executor._results_directory / Path(ckpt_path)
        results_disk.append(np.array(pickle.load(open(path, "rb"))))

    assert (results == results_disk).all()

    shutil.rmtree(root)
