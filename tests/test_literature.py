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
def efb_figure5_trials():

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

    def test_EFB_figure5_k0(self, efb_figure5_trials):
        trial = efb_figure5_trials

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

    def test_EFB_figure5_k1(self, efb_figure5_trials):
        trial = efb_figure5_trials

        gt = np.array([
             [-5.50000000e+00,  5.70009767e-04],
             [-5.37975952e+00,  8.46262556e-04],
             [-5.25951904e+00,  1.46302959e-03],
             [-5.13927856e+00,  3.46046303e-03],
             [-5.01903808e+00,  2.23078513e-02],
             [-4.89879760e+00,  4.64096303e-02],
             [-4.77855711e+00,  1.20128768e+00],
             [-4.65831663e+00,  5.14812003e-03],
             [-4.53807615e+00,  4.91922895e-02],
             [-4.41783567e+00,  2.08925512e-03],
             [-4.29759519e+00,  1.61751407e-03],
             [-4.17735471e+00,  2.23838506e-03],
             [-4.05711423e+00,  3.09670706e-03],
             [-3.93687375e+00,  7.18639055e-03],
             [-3.81663327e+00,  4.26293073e-02],
             [-3.69639279e+00,  5.42547982e-01],
             [-3.57615230e+00,  1.28572682e-02],
             [-3.45591182e+00,  2.27431234e-02],
             [-3.33567134e+00,  9.11500901e-03],
             [-3.21543086e+00,  4.00282833e-03],
             [-3.09519038e+00,  2.88827163e-03],
             [-2.97494990e+00,  3.50607146e-03],
             [-2.85470942e+00,  2.76394326e-03],
             [-2.73446894e+00,  2.33004830e-03],
             [-2.61422846e+00,  4.91466132e-02]
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
                G, meta = sy.one_shot_sparse_solve(0.5 * np.pi, w)
                sparse.append(-G.imag / np.pi)
                G, meta = sy.continued_fraction_dense_solve(0.5 * np.pi, w)
                dense.append(-G.imag / np.pi)

            sparse = np.array(sparse)
            dense = np.array(dense)

            assert np.allclose(sparse, dense, atol=1e-4)
            assert np.allclose(dense, A_gt, atol=1e-4)

    def test_EFB_figure5_k2(self, efb_figure5_trials):
        trial = efb_figure5_trials

        gt = np.array([
             [-5.50000000e+00,  6.56198651e-04],
             [-5.37975952e+00,  1.05423340e-03],
             [-5.25951904e+00,  2.09222564e-03],
             [-5.13927856e+00,  6.80256912e-03],
             [-5.01903808e+00,  4.10703772e-01],
             [-4.89879760e+00,  1.51019463e-02],
             [-4.77855711e+00,  3.47672117e-02],
             [-4.65831663e+00,  2.69908752e-03],
             [-4.53807615e+00,  1.54922356e-03],
             [-4.41783567e+00,  1.13498177e-03],
             [-4.29759519e+00,  1.18545604e-03],
             [-4.17735471e+00,  2.15536581e-03],
             [-4.05711423e+00,  6.08022403e-03],
             [-3.93687375e+00,  8.82318049e-02],
             [-3.81663327e+00,  5.33174310e-03],
             [-3.69639279e+00,  1.47541011e-02],
             [-3.57615230e+00,  1.16532005e+00],
             [-3.45591182e+00,  1.79040206e-02],
             [-3.33567134e+00,  6.21414078e-03],
             [-3.21543086e+00,  5.07344742e-03],
             [-3.09519038e+00,  3.33886584e-01],
             [-2.97494990e+00,  5.98140209e-03],
             [-2.85470942e+00,  2.86979889e-03],
             [-2.73446894e+00,  1.52474837e-03],
             [-2.61422846e+00,  1.64323866e-02]
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
                G, meta = sy.one_shot_sparse_solve(np.pi, w)
                sparse.append(-G.imag / np.pi)
                G, meta = sy.continued_fraction_dense_solve(np.pi, w)
                dense.append(-G.imag / np.pi)

            sparse = np.array(sparse)
            dense = np.array(dense)

            assert np.allclose(sparse, dense, atol=1e-4)
            assert np.allclose(dense, A_gt, atol=1e-4)
