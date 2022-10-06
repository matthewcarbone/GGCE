import pytest

import numpy as np
import shutil

from ggce.logger import _testing_mode
from ggce import Model, System

model_test_params = {
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


@pytest.mark.parametrize("p",[model_test_params])
def test_sys_chkpt(p):
    model = Model.from_parameters(**p["model_params"])
    model.add_(**p["model_add_params"])
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

def test_model_chkpt():
    #TODO test the checkpointing of results
    pass

def test_res_chkpt():
    #TODO test the checkpointing of results
    pass
