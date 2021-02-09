#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os
from pathlib import Path

import numpy as np
import yaml

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ggce.utils import utils
from ggce.utils.logger import default_logger as dlog


def lorentzian(x, x0, a, gam):
    return np.abs(a) * gam**2 / (gam**2 + (x - x0)**2)


class Results:
    """Trials are single spectra A(w) for some fixed k, and all other
    parameters. This class is a helper for querying trials based on the
    parameters specified, and returning spectral functions A(w)."""

    def __init__(self, package_path, res="res.npy"):

        # Load in the initial data
        package_path = Path(package_path)

        self.paths = {
            'results': package_path / Path("results"),
            'bash_script': package_path / Path("submit.sbatch"),
            'configs': package_path / Path("configs"),
            'grids': package_path / Path("grids.yaml")
        }

        self.all_configs = {
            f: yaml.safe_load(open(f, 'r'))
            for f in self.paths['configs'].iterdir()
        }
