import numpy as np


max_bosons_per_site = 100


def far_edges(length):
    """
    Get a list of the coordinates of an LxL matrix's right column and bottom row.
    """
    indices = [*range(length)]
    edges = [length - 1 for _ in range(length)]
    right = [*zip(indices, edges)]
    bottom = [*zip(edges, indices)]
    coords = right + bottom[-2::-1]

    return coords


def overhang(pop, m):
    """
    Takes a configuration, extends it one unit right and down, and returns a list of
    configurations with 1 addition boson at each new site. Repeats until the new pops
    are of size M.
    """
    pops = []
    shape = pop.shape
    for i in range(1, m - shape[0] + 1):
        extended_pop = np.zeros([shape[0] + i, shape[1] + i], dtype=int)
        extended_pop[0:shape[0], 0:shape[1]] = pop
        for coord in far_edges(shape[0] + i):
            plus_one = np.zeros([shape[0] + i, shape[1] + i], dtype=int)
            plus_one[coord[0], coord[1]] = 1
            new_pop = extended_pop + plus_one
            pops.append(new_pop)

    return pops


def will_append(pop, comparison):
    """
    Checks to see if a pop exceeds the maximum bosons per site or if it is
    a duplicate of another pop that has already been added.
    """
    will = True
    if np.max(pop) > max_bosons_per_site:
        will = False
    for test in comparison:
        if np.array_equal(test, pop):
            will = False

    return will


def edge_cases(n, m):
    cases = []
    for j in range(2, m + 1):
        for coord in far_edges(j):  # top left occupied
            case1, case2 = np.zeros([j, j], dtype=int), np.zeros([j, j], dtype=int)
            case1[0][0] = n
            case1[coord[0]][coord[1]] = 1
            case2[0][0] = 1
            case2[coord[0]][coord[1]] = n
            cases.append(case1)
            if not n == 1 and coord != (0, j - 1):
                cases.append(case2)

        case = np.zeros([j, j], dtype=int)  # bottom left and top right occupied
        case[0, j - 1] = 1
        case[j - 1, 0] = n
        cases.append(case)
        if n != 1:
            cases.append(case.T)

        for i in range(1, j - 1):  # bottom left or top right occupied
            case = np.zeros([j, j], dtype=int)
            case[0, j - 1] = n
            case[i, 0] = 1
            cases.append(case)
            cases.append(case.T)

    return cases
