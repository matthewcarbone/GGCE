import pytest

import numpy as np
import warnings

from ggce import _testing_mode
from ggce.engine.terms import Config


class TestConfig:
    @staticmethod
    def test_Config_MSONable():
        config = Config(np.array([[1, 2, 3, 4]]))
        d = config.as_dict()
        config2 = Config.from_dict(d)
        assert np.all(config.config == config2.config)

    @staticmethod
    @pytest.mark.parametrize(
        "arr",
        [(np.array([0, 1, 2, 3, 4]))],
    )
    def test_Config_1d_check_config_errorstates(arr):
        with pytest.raises(SystemExit):
            Config(arr)

    @staticmethod
    def test_Config_1d_check_config_warning():

        arr = (np.random.random(size=(2, 3, 3, 3, 4)) + 1.5).astype(int)
        with _testing_mode():
            with warnings.catch_warnings(record=True) as w:
                Config(arr)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "DUMMY WARNING" in str(w[-1].message)
