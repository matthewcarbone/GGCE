#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import pytest

import numpy as np

from ggce.engine.terms import BosonConfig


DUMMY_CACHE = "DUMMY_CACHE"
DUMMY_LIFO = "DUMMY_LIFO"


class TestBosonConfigs:

    # def test_add_same_shape(self):

    #     config1 = np.array([
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig1 = BosonConfig(config1)

    #     config2 = np.array([
    #         [3, 4, 5],
    #         [29, 1, 9]
    #     ])
    #     bosonConfig2 = BosonConfig(config2)

    #     bosonConfig3 = bosonConfig1 + bosonConfig2
    #     np.testing.assert_array_equal(
    #         bosonConfig3.config, config1 + config2
    #     )

    # def test_add_different_shapes(self):

    #     config1 = np.array([
    #         [1, 2, 3, 10, 11, 16],
    #         [4, 5, 6, 0, 17, 24]
    #     ])
    #     bosonConfig1 = BosonConfig(config1)

    #     config2 = np.array([
    #         [3, 4, 5],
    #         [29, 1, 9]
    #     ])
    #     bosonConfig2 = BosonConfig(config2)

    #     bosonConfig3 = bosonConfig1 + bosonConfig2

    #     new_array = np.array([
    #         [4, 6, 8, 10, 11, 16],
    #         [33, 6, 15, 0, 17, 24]
    #     ])

    #     np.testing.assert_array_equal(bosonConfig3.config, new_array)

    #     bosonConfig3 = bosonConfig2 + bosonConfig1

    #     np.testing.assert_array_equal(bosonConfig3.config, new_array)

    # def test_equal_same_size(self):

    #     config1 = np.array([
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig1 = BosonConfig(config1)
    #     config2 = np.array([
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig2 = BosonConfig(config2)
    #     assert bosonConfig1 == bosonConfig2

    # def test_neq_same_size(self):

    #     config1 = np.array([
    #         [1, 9, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig1 = BosonConfig(config1)
    #     config2 = np.array([
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig2 = BosonConfig(config2)
    #     assert bosonConfig1 != bosonConfig2

    # def test_neq_diff_size(self):

    #     config1 = np.array([
    #         [1, 2, 3, 5],
    #         [4, 5, 6, 5]
    #     ])
    #     bosonConfig1 = BosonConfig(config1)
    #     config2 = np.array([
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #     bosonConfig2 = BosonConfig(config2)
    #     assert bosonConfig1 != bosonConfig2

    def test_is_zero_one_boson_type(self):

        config = np.array([[0]])
        bosonConfig = BosonConfig(config)
        assert bosonConfig.is_zero()

    def test_is_zero_one_boson_type_1d(self):

        config = np.array([0])
        bosonConfig = BosonConfig(config)
        assert bosonConfig.is_zero()

    def test_not_zero(self):

        config = np.array([
            [1],
            [2]
        ])
        bosonConfig = BosonConfig(config)
        assert not bosonConfig.is_zero()

    def test_identifier(self):

        config = np.array([
            [1, 5],
            [2, 8]
        ])
        bosonConfig = BosonConfig(config)
        assert bosonConfig.identifier() == '[[1, 5], [2, 8]]'

    def test_is_legal_zeros(self):

        config = np.array([[0], [0]])
        bosonConfig = BosonConfig(config)
        assert bosonConfig.is_legal()

    def test_is_legal(self):

        config = np.array([
            [12, 2, 3, 9, 0, 0, 2],
            [1, 0, 0, 2, 3, 0, 8]
        ])
        bosonConfig = BosonConfig(config)
        assert bosonConfig.is_legal()

    def test_is_illegal(self):

        config = np.array([
            [0, 2, 3, 9, 0, 0, 2],
            [0, 0, 0, 2, 3, 0, 8]
        ])
        bosonConfig = BosonConfig(config)
        assert not bosonConfig.is_legal()

    def test_is_valid(self):

        config = np.array([
            [5, 5, 2],
            [1, -1, 1]
        ])
        with pytest.raises(RuntimeError):
            BosonConfig(config)
