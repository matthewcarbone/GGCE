#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import numpy as np
import pytest
import yaml


from ggce.engine import structures, system
from ggce.utils import utils


@pytest.fixture
def efb_figure6_trials():

    # Load in the parameters
    d = yaml.safe_load(open("inp/benchmarks/EFB_Figure5.yaml", 'r'))
    model = d['model']
    info = d['info']
    params = d['model_parameters']

    # Create the ModelParams object
    mp = structures.ModelParams(model, info, params)

    # Get all of the sub-dictionaries containing the data- it's just one
    # for this particular data
    return [xx for xx in mp][0]


class TestLiterature:

    def test_EFB_figure6_k0(self, efb_figure6_trials):
        trial = efb_figure6_trials

        # Set the "ground truth" data, calculated using this code, compared
        # to the true correct answer in Berciu & Fehske, PRB 82, 085116 (2010)
        # after normalizing to height 1, and randomly selected.

        gt = np.array([
            [-5.50000000e+00,  1.43501419e-03],
            [-5.37975952e+00,  3.19572879e-03],
            [-5.25951904e+00,  1.41728939e-02],
            [-5.13927856e+00,  5.28686138e-01],
            [-5.01903808e+00,  8.66237856e-03],
            [-4.89879760e+00,  4.80114543e-03],
            [-4.77855711e+00,  5.76243081e-03],
            [-4.65831663e+00,  1.44579253e-03],
            [-4.53807615e+00,  1.24225633e-03],
            [-4.41783567e+00,  1.24347394e-03],
            [-4.29759519e+00,  1.68881078e-03],
            [-4.17735471e+00,  3.54070940e-03],
            [-4.05711423e+00,  1.42006278e-02],
            [-3.93687375e+00,  4.70721754e+00],
            [-3.81663327e+00,  1.30010459e-02],
            [-3.69639279e+00,  4.29116160e-02],
            [-3.57615230e+00,  2.59127687e-02],
            [-3.45591182e+00,  8.53183064e-03],
            [-3.33567134e+00,  5.75017293e-03],
            [-3.21543086e+00,  1.25839520e+00],
            [-3.09519038e+00,  4.64001429e-03],
            [-2.97494990e+00,  2.01461344e-03],
            [-2.85470942e+00,  2.05228957e-03],
            [-2.73446894e+00,  1.64700276e-03],
            [-2.61422846e+00,  1.01781738e-02]
        ])

        w_grid = gt[:, 0]
        A_gt = gt[:, 1]

        sp = structures.SystemParams(trial)
        sp.prime()

        with utils.DisableLogger():
            sy = system.System(sp)
            sy.initialize_generalized_equations()
            sy.initialize_equations()
            sy.generate_unique_terms()
            sy.prime_solver()

            sparse = []
            dense = []
            for w in w_grid:
                G, meta = sy.one_shot_sparse_solve(0.0, w)
                sparse.append(-G.imag / np.pi)
                G, meta = sy.continued_fraction_dense_solve(0.0, w)
                dense.append(-G.imag / np.pi)

            sparse = np.array(sparse)
            dense = np.array(dense)

            assert np.allclose(sparse, dense, atol=1e-4)
            assert np.allclose(dense, A_gt, atol=1e-4)
