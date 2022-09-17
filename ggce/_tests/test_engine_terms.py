import pytest

import numpy as np
import warnings

from ggce.logger import _testing_mode
from ggce.engine.terms import Config


class TestConfig:
    @staticmethod
    def test_MSONable():
        config = Config(np.array([[1, 2, 3, 4]]))
        d = config.as_dict()
        config2 = Config.from_dict(d)
        assert np.all(config.config == config2.config)

    @staticmethod
    def test_id_Green(
        Random1dPhononArray, Random2dPhononArray, Random3dPhononArray
    ):
        assert Config(Random1dPhononArray.copy() * 0).id() == "G"
        assert Config(Random2dPhononArray.copy() * 0).id() == "G"
        assert Config(Random3dPhononArray.copy() * 0).id() == "G"

    @staticmethod
    def test_1d_check_config_errorstate_dim_too_few():
        with pytest.raises(SystemExit):
            Config(np.array([0, 1, 2, 3, 4]))

    @staticmethod
    def test_1d_check_config_errorstate_negative_phonon_occupation(
        Random3dPhononArray,
    ):
        arr = Random3dPhononArray.copy()
        arr[2, 0, 1, 3] = -1
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_1d_check_config_warning():

        arr = (np.random.random(size=(2, 3, 3, 3, 4)) + 1.5).astype(int)
        with _testing_mode():
            with warnings.catch_warnings(record=True) as w:
                Config(arr)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "DUMMY WARNING" in str(w[-1].message)

    @staticmethod
    @pytest.mark.parametrize(
        "n_phonons",
        [1, 3, 5],
    )
    def test_n_phonon_types(n_phonons):
        arr = (np.random.random(size=(n_phonons, 3, 2, 3)) + 1.5).astype(int)
        config = Config(arr)
        assert config.n_phonon_types == n_phonons

    @staticmethod
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_phonon_cloud_shapes(dims):
        arr = (np.random.random(size=([2] + [3] * dims)) + 1.5).astype(int)
        config = Config(arr)
        assert config.phonon_cloud_shape == tuple([3] * dims)

    @staticmethod
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_total_phonons_per_type_and_total_phonons(dims):
        size = tuple([ii + 4 for ii in range(dims)])
        ptype1 = (np.random.random(size=size) + 1.5).astype(int)
        ptype2 = (np.random.random(size=size) + 2.5).astype(int)
        ptype3 = (np.random.random(size=size) + 3.5).astype(int)
        arr = np.array([ptype1, ptype2, ptype3])
        config = Config(arr)
        sarray = np.array([ptype1.sum(), ptype2.sum(), ptype3.sum()])
        assert np.all(config.total_phonons_per_type == sarray)
        assert config.total_phonons == sarray.sum()

    @staticmethod
    def test_Greens_config(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr *= 0
        Config(arr)

    @staticmethod
    def test_validate_critical_1d(Random1dPhononArray):
        arr = Random1dPhononArray.copy()
        arr[1, 2] = -1
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_2d(Random2dPhononArray):
        arr = Random2dPhononArray.copy()
        arr[1, 0, 1] = -1
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_3d(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr[1, 2, 1, 1] = -1
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_1d_left(Random1dPhononArray):
        arr = Random1dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, 0] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_1d_right(Random1dPhononArray):
        arr = Random1dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, -1] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_2d_left(Random2dPhononArray):
        arr = Random2dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, 0, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_2d_right(Random2dPhononArray):
        arr = Random2dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, -1, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_2d_up(Random2dPhononArray):
        arr = Random2dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, 0] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_2d_down(Random2dPhononArray):
        arr = Random2dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, -1] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_left(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, 0, :, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_right(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, -1, :, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_up(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, 0, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_down(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, -1, :] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_front(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, :, 0] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_validate_critical_config_edges_3d_back(Random3dPhononArray):
        arr = Random3dPhononArray.copy()
        arr += 1
        Config(arr.copy())
        arr[:, :, :, -1] = 0
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_add_phonon_max_modifications(Random1dPhononArray):
        config = Config(Random1dPhononArray.copy())
        config.add_phonon_(0, 1)
        with pytest.raises(SystemExit):
            config.add_phonon_(1, 2)

    @staticmethod
    def test_remove_phonon_max_modifications(Random1dPhononArray):
        config = Config(Random1dPhononArray.copy() + 3)
        config.remove_phonon_(0, 1)
        with pytest.raises(SystemExit):
            config.remove_phonon_(1, 2)

    @staticmethod
    def test_add_remove_dimension_mismatch(Random3dPhononArray):
        config = Config(Random3dPhononArray.copy())
        with pytest.raises(SystemExit):
            config.add_phonon_(1, 0, 0, 0, 0)
        config = Config(Random3dPhononArray.copy())
        with pytest.raises(SystemExit):
            config.remove_phonon_(1, 0, 0, 0, 0)

    @staticmethod
    def test_remove_phonon_index_dne(Random3dPhononArray):
        config = Config(Random3dPhononArray.copy())
        with pytest.raises(SystemExit):
            config.remove_phonon_(1, 0, 0, 20)

    @staticmethod
    def test_remove_phonon_negative_occupancy_after_removal(
        Random3dPhononArray,
    ):
        arr = Random3dPhononArray.copy()
        arr[0, 2, 3, 3] = 0
        config = Config(arr)
        with pytest.raises(SystemExit):
            config.remove_phonon_(0, 2, 3, 3)

    @staticmethod
    def test_remove_phonon_shift_1d_right():
        arr = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 0]])
        config = Config(arr.copy())
        shift = config.remove_phonon_(0, 6)
        assert np.all(shift == np.array([0]))
        assert np.all(config.config == np.array([[1, 2, 3], [4, 5, 6]]))

    @staticmethod
    def test_remove_phonon_shift_1d_left():
        arr = np.array([[0, 0, 0, 0, 1, 2, 3], [1, 0, 0, 0, 4, 5, 6]])
        config = Config(arr.copy())
        shift = config.remove_phonon_(1, 0)
        assert np.all(shift == np.array([4]))
        assert np.all(config.config == np.array([[1, 2, 3], [4, 5, 6]]))

    @staticmethod
    def test_add_phonon_Shift_1d_right():
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        config = Config(arr.copy())
        shift = config.add_phonon_(0, 6)
        arr2 = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 0]])
        assert np.all(shift == np.array([0]))
        assert np.all(config.config == arr2)

    @staticmethod
    def test_add_phonon_Shift_1d_left():
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        config = Config(arr.copy())
        shift = config.add_phonon_(1, -4)
        arr2 = np.array([[0, 0, 0, 0, 1, 2, 3], [1, 0, 0, 0, 4, 5, 6]])
        assert np.all(shift == np.array([4]))
        assert np.all(config.config == arr2)

    @staticmethod
    def test_remove_phonon_shift_2d_0():
        arr = np.array([[[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 0]]])
        config = Config(arr.copy())
        shift = config.remove_phonon_(0, 0, 6)
        assert np.all(shift == np.array([0, 0]))
        assert np.all(config.config == np.array([[[1, 2, 3], [4, 5, 6]]]))

    @staticmethod
    def test_remove_phonon_shift_2d_1():
        arr = np.array([[[0, 2, 3], [1, 0, 0]]])
        config = Config(arr.copy())
        shift = config.remove_phonon_(0, 1, 0)
        assert np.all(shift == np.array([0, 1]))
        assert np.all(config.config == np.array([[[2, 3]]]))
