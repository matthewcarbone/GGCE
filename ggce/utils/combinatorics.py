from numpy import prod
from scipy.special import comb


def generalized_equations_combinatorics_term(m, n):
    """The total number of generalized equations is given by the exact
    relation

    sum_{m, n = 1}^{M, N} c_{m, n}

    where c_{m, n} is given by this equation, and is equal to

    * 1 if m = 1 or n = 2
    * (m + n - 3) choose (n - 2) otherwise
    """

    if m == 1 or n == 2:
        return 1

    return comb(m + n - 3, n - 2, exact=True)


def total_generalized_equations(M, N, nbt):
    """Gets the total number of generalized equations as predicted by the
    combinatorics equation described in
    generalized_equations_combinatorics_term.
    """

    bosons = [
        sum(
            [
                sum(
                    [
                        generalized_equations_combinatorics_term(m, n)
                        for n in range(1, N[bt] + 1)
                    ]
                )
                for m in range(1, M[bt] + 1)
            ]
        )
        for bt in range(nbt)
    ]
    return int(prod(bosons))
