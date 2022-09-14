import pytest

from ggce.engine.system import config_space_gen


@pytest.mark.parametrize(
    "length,total_sum,expected",
    [
        (2, 2, {(2, 0), (1, 1), (0, 2)}),
        (2, 3, {(3, 0), (0, 3), (1, 2), (2, 1)}),
        (
            3,
            2,
            {(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)},
        ),
    ],
)
def test_config_space_gen(length, total_sum, expected):
    assert set(config_space_gen(length, total_sum)) == expected
