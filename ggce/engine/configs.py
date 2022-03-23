import numpy as np


class BosonConfig:
    """A class for holding boson occupations and defining operations on the
    boson cloud.

    Attributes
    ----------
    config : np.ndarray
        An array of the shape (cloud_length_x, cloud_length_y) - will need to
        re-add multiple boson types
    max_modifications : int
        The maximum number of times one can modify the boson config before
        throwing an error. This is precisely equal to the order of the
        boson creation operators in V.
    """

    def __init__(self, config, max_modifications=1):
        self.config = np.atleast_2d(config)
        self.n_boson_types = self.config.shape[0]
        self.cloud_length = self.config.shape[-1]
        self.total_bosons_per_type = np.sum(self.config, axis=tuple(range(1, self.config.ndim)))
        self.total_bosons = np.sum(self.total_bosons_per_type)
        self.n_modified = 0
        self.max_modifications = max_modifications
        self.assert_valid()

    def is_zero(self):
        if self.cloud_length == 1:
            if np.sum(self.config) == 0:
                return True
        return False

    def identifier(self):
        if self.is_zero():
            return "G"
        return str([list(s) for s in self.config])

    def shrink(self):
        """
        Takes a boson configuration, sums over each boson type, and shifts it such that the non-zero
        elements are as far up and to the right as possible. Shaves off zeroes to the right and bottom
        if needed, and returns the legal configuration and how far up and to the right it was shifted.
        Works in 1, 2 and 3 dimensions.
        net: int. By default, this method sums over each boson type and checks for net legality.
        However, if boson removal makes the arrays illegal, each array for each boson type must
        be sanitized. The net index picks out a particular boson type to sanitize.
        """
        net_config = np.sum(self.config, axis=0)
        dimension = net_config.ndim
        shifts = [0 for _ in range(dimension)]
        for i in range(dimension):
            if dimension == 1:
                while net_config[0] == 0:
                    net_config = net_config[1:]
                    shifts[0] += 1
                while net_config[-1] == 0:
                    net_config = net_config[:-1]
            else:
                axes_to_sum = tuple([j for j in range(dimension) if j != i])
                while np.sum(net_config, axis=axes_to_sum)[0] == 0:
                    net_config = np.roll(net_config, -1, axis=i)
                    shifts[i] += 1

                # this part chops off the excess 0s of the config but there has to be a way to unify these
                if dimension == 2:
                    while np.sum(net_config[-1, :]) + np.sum(net_config[:, -1]) == 0:
                        net_config = net_config[0:-1, 0:-1]
                elif dimension == 3:
                    while np.sum(net_config[-1, :, :]) + np.sum(net_config[:, -1, :]) + np.sum(
                            net_config[:, :, -1]) == 0:
                        net_config = net_config[0:-1, 0:-1, 0:-1]

        return net_config, shifts

    def sanitize(self):
        net_config, shifts = self.shrink()
        new_shape = (self.n_boson_types, ) + tuple(net_config.shape)
        new_len = new_shape[1]
        new_arr = np.zeros(new_shape)
        dimension = self.config.ndim - 1
        if dimension == 1:
            for i in range(self.n_boson_types):
                new_arr[i, ...] = self.config[i, shifts[0]:shifts[0]+new_len]
        if dimension == 2:
            for i in range(self.n_boson_types):
                new_arr[i, ...] = self.config[i, shifts[0]:shifts[0]+new_len,
                                              shifts[1]:shifts[1]+new_len]
        if dimension == 3:
            for i in range(self.n_boson_types):
                new_arr[i, ...] = self.config[i, shifts[0]:shifts[0]+new_len,
                                              shifts[1]:shifts[1]+new_len,
                                              shifts[2]:shifts[2]+new_len]
        self.config = new_arr

    def is_legal(self):
        """Checks if the cloud is a legal configuration by comparing
        the sanitized version of the array to the sanitized version.
        If same, then is legal."""

        if self.is_zero():
            return True

        check, foo = self.shrink()
        if not np.array_equal(check, np.sum(self.config, axis=0)):
            return False

        return True

    def assert_valid(self):
        """Checks that the configuration contains entries >= 0 only. If not
        raises a RuntimeError, else silently does nothing."""

        if np.any(self.config < 0):
            raise RuntimeError(f"Invalid config: {self.config}")
        if np.sum(self.total_bosons_per_type) < 0:
            raise RuntimeError("Config has less than 0 total bosons")

    def remove_boson(self, boson_type, location):
        """Removes a boson of type boson_type from the specified cloud
        location. if removal is not possible, raises a RuntimeError. Returns
        the value to shift the exp_shift of the Terms class."""

        if self.n_modified >= self.max_modifications:
            raise RuntimeError("Max modifications exceeded")
        if boson_type > self.n_boson_types - 1:
            raise RuntimeError("Boson type remove error")
        if np.any(np.array(location) > self.cloud_length - 1) and np.any(np.array(location) < 0):
            raise RuntimeError("Mixed location remove error")
        elif np.any(np.array(location) > self.cloud_length - 1):
            raise RuntimeError("Positive location remove error")
        elif np.any(np.array(location) < 0):
            raise RuntimeError("Negative location remove error")

        # this block makes the coordinate adaptable for 1, 2 and 3 spatial dimensions
        coord = [boson_type]
        for index in location:
            coord.append(index)
        self.config[tuple(coord)] -= 1
        shift = 0

        if not self.is_zero():
            foo, shift = self.shrink()
            self.sanitize()
            print(self.config)

        self.assert_valid()
        assert self.is_legal()
        self.n_modified += 1
        self.total_bosons_per_type[boson_type] -= 1
        self.total_bosons -= 1

        return shift

    def add_boson(self, boson_type, location):
        """Adds a boson of type boson_type to the specified location."""
        location = np.array(location)
        dimension = self.config.ndim - 1
        if self.n_modified >= self.max_modifications:
            raise RuntimeError("Max modifications exceeded")
        if boson_type > self.n_boson_types - 1:
            raise RuntimeError("Boson type add error")

        coord = [boson_type]
        for index in location:
            coord.append(index)

        # Easy case: the boson type to add is in the existing cloud:
        if np.all(np.zeros(self.config.ndim - 1) <= location) \
                and np.all(location <= np.ones(self.config.ndim - 1)*self.cloud_length):
            self.config[tuple(coord)] += 1

        else:
            new_size = np.max(np.where(location < 0, self.cloud_length - location, location + 1))
            relative_corner = np.where(location < 0, -location, 0).astype(int)
            new_subshape = tuple([int(new_size) for _ in range(self.config.ndim - 1)])
            new_arr = np.zeros((self.n_boson_types, ) + new_subshape).astype(int)
            new_location = np.where(location < 0, 0, location)

            new_arr[(boson_type, ) + tuple(new_location)] += 1
            # there has to be a cleaner way to do this
            # want to embed the original config into the new one
            # the annoying part is that each slice has to be the length of the original cloud
            if dimension == 2:
                new_arr[boson_type][relative_corner[0]:relative_corner[0] + self.cloud_length,
                                    relative_corner[1]:relative_corner[1] + self.cloud_length]\
                    = self.config[boson_type]
            elif dimension == 3:
                new_arr[boson_type][relative_corner[0]:relative_corner[0] + self.cloud_length,
                                    relative_corner[1]:relative_corner[1] + self.cloud_length,
                                    relative_corner[2]:relative_corner[2] + self.cloud_length]\
                    = self.config[boson_type]

            print(new_arr)
            self.config = new_arr
            self.cloud_length = self.config.shape[1]
        shift = np.where(location < 0, -location, 0)

        self.assert_valid()
        assert self.is_legal()
        self.n_modified += 1
        self.total_bosons_per_type[boson_type] += 1
        self.total_bosons += 1

        return shift
