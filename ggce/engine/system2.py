from collections import OrderedDict
import copy
import os
from pathlib import Path
import pickle
import time

from ggce.engine.equations import Equation, GreenEquation
from ggce.utils.logger import Logger, get_GGCE_CONFIG_STORAGE
from ggce.utils.combinatorics import ConfigurationSpaceGenerator, \
    total_generalized_equations


class Metadata(dict):

    def get_name(self, model):
        ae = model.absolute_extent
        M = model.M
        N = model.N
        max_bosons_per_site = model.max_bosons_per_site
        return f"{ae}_{M}_{N}_{max_bosons_per_site}_{self._class_type}.pkl"

    def save(self):
        assert self._class_type is not None
        root = get_GGCE_CONFIG_STORAGE()
        name = root / self._name
        if not root.exists():
            root.mkdir(exist_ok=False, parents=True)

        # Convert the class into a proper dictionary before saving
        pickle.dump(dict(self), open(name, "wb"), protocol=4)

    def __init__(self, x, model, recompute_required=True):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description
        recompute_required : bool, optional
            Description
        """

        super().__init__(x)
        self._name = self.__class__._get_name(model)
        self._recompute_required = recompute_required
