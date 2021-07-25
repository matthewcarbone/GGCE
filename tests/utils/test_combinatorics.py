import pytest

from ggce.utils import combinatorics


@pytest.mark.parametrize("mn", [
    (1, 1), (1, 2), (1, 10), (10, 2), (5, 2), (3, 2)
])
def test_generalized_equations_combinatorics_term_is1(mn):
    assert combinatorics.generalized_equations_combinatorics_term(*mn) == 1


def test_generalized_equations_combinatorics_term():
    assert combinatorics.generalized_equations_combinatorics_term(3, 3) == 3
    assert combinatorics.generalized_equations_combinatorics_term(2, 3) == 2
