import numpy as np


def config_space_gen(length, total_sum):
    """Generator for yielding all possible combinations of integers of
    length `length` that sum to total_sum. Note that cases such as
    length = 4 and total_sum = 5 like [0, 0, 2, 3] need to be screened
    out, since these do not correspond to valid f-functions.

    Source of algorithm:
    https://stackoverflow.com/questions/7748442/
    generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """

    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in config_space_gen(length - 1, total_sum - value):
                r = (value,) + permutation
                yield r


class ConfigurationSpaceGenerator:
    """Helper class for generating configuration spaces of bosons.

    Parameters
    ----------
    system_params : SystemParams
        The system parameters containing:
        * absolute_extent : int
            The maximum extent of the cloud. Note that this is NOT trivially
            the sum(M) since there are different ways in which the clouds can
            overlap. Take for instance, two boson types, with M1 = 3 and
            M2 = 2. These clouds can overlap such that absolute extent is at
            most 5, and they can also overlap such that the maximal extent is
            3.
        * M : list
            Length of the allowed cloud extent for each boson type.
        * N : list
            Total number of maximum allowed bosons for each boson type. The
            absolute maximum number of bosons is trivially sum(N).
    """

    def __init__(self, system_params):
        self.absolute_extent = system_params.absolute_extent
        self.M = system_params.M
        assert self.absolute_extent >= max(self.M)
        self.N = system_params.N
        self.N_2d = np.atleast_2d(self.N).T
        self.n_boson_types = len(self.M)
        assert len(self.N) == self.n_boson_types
        self.max_bosons_per_site = system_params.max_bosons_per_site

    def extent_of_1d(self, config1d):
        """Gets the extent of a 1d vector."""

        where_nonzero = np.where(config1d != 0)[0]
        L = len(where_nonzero)
        if L < 2:
            return L

        minimum_index = min(where_nonzero)
        maximum_index = max(where_nonzero)
        return maximum_index - minimum_index + 1

    def is_legal(self, config):
        """Checks the condition of a config. If legal, returns True, else
        returns False."""

        # First, check that the edges of the cloud have at least one boson
        # of either type on it.
        if np.sum(config[:, 0]) < 1 or np.sum(config[:, -1]) < 1:
            return False

        # Second, check that the boson numbers for each boson type are
        # satisfied
        if not np.all(config.sum(axis=1, keepdims=True) <= self.N_2d):
            return False

        # Finally, check that each boson type satisifes its extent rule
        if any([
            self.M[ii] < self.extent_of_1d(c1d)
            for ii, c1d in enumerate(config)
        ]):
            return False

        # Constraint that we have maximum N bosons per site.
        if self.max_bosons_per_site is not None:
            if not np.all(config <= self.max_bosons_per_site):
                return False

        return True

    def __call__(self):
        """Generates all possible configurations of bosons for the composite
        system. Then, reduce that space down based on the specific rules for
        each boson type."""

        nb_max = sum(self.N)
        config_dict = {N: [] for N in range(1, nb_max + 1)}
        for nb in range(1, nb_max + 1):
            for z in range(1, self.absolute_extent + 1):
                c = list(config_space_gen(z * self.n_boson_types, nb))

                # Reshape everything properly:
                c = [np.array(z).reshape(self.n_boson_types, -1) for z in c]

                # Now, we get all of the LEGAL nvecs, which are those in
                # which at least a single boson of any type is on both
                # edges of the cloud.
                tmp_legal_configs = [nn for nn in c if self.is_legal(nn)]

                # Extend the temporary config
                config_dict[nb].extend(tmp_legal_configs)

        return config_dict
