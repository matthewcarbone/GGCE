import numpy as np

from ggce.engine.terms import Config


class TestConfig:
    @staticmethod
    def test_config_MSONable():
        config = Config(np.array([[1, 2, 3, 4]]))
        d = config.as_dict()
        config2 = Config.from_dict(d)
        assert np.all(config.config == config2.config)
