import pytest

from scipy.special import comb

from ggce.utils import combinatorics


@pytest.mark.parametrize("mn", [
    (1, 1), (1, 2), (1, 10), (10, 2), (5, 2), (3, 2)
])
def test_generalized_equations_combinatorics_term_is1(mn):
    assert combinatorics.generalized_equations_combinatorics_term(*mn) == 1


def test_generalized_equations_combinatorics_term():
    assert combinatorics.generalized_equations_combinatorics_term(3, 3) == 3
    assert combinatorics.generalized_equations_combinatorics_term(2, 3) == 2


@pytest.mark.parametrize("M,N", [(10, 4), (4, 10), (5, 5), (2, 8)])
def test_total_generalized_equations(M, N):
    s = 0
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            s += combinatorics.generalized_equations_combinatorics_term(m, n)
    assert s == combinatorics.total_generalized_equations(M, N)


@pytest.mark.parametrize("lt", [(10, 4), (4, 10), (5, 5), (2, 8)])
def test_config_space_gen(lt):
    c = comb(lt[0] + lt[1] - 1, lt[1], exact=True)
    assert len(list(combinatorics.config_space_gen(*lt))) == c
