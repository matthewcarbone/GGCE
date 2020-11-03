#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import os
import yaml
import numpy as np
import copy
from scipy.signal import find_peaks
from scipy import interpolate


def listdir_fullpath(d):
    dirs = [os.path.join(d, f) for f in os.listdir(d)]
    return [d for d in dirs if os.path.isdir(d)]


class Results:

    @staticmethod
    def key_str(config, value):
        t = f"{config['model']}_{config['Omega']}_{config['t']}_"
        t += f"{value['k_units_pi']}_{value['lambda']}_{value['M']}" \
            + f"_{value['N']}_{value['eta']}"
        return t

    def _assign_key(self, config, value):
        """Returns the correct key value and logs the keys in their
        appropriate lists."""

        self.vals['model'].append(config['model'])
        self.vals['Omega'].append(config['Omega'])
        self.vals['t'].append(config['t'])

        self.vals['k_units_pi'].append(value['k_units_pi'])
        self.vals['lambda'].append(value['lambda'])
        self.vals['M'].append(value['M'])
        self.vals['N'].append(value['N'])
        self.vals['eta'].append(value['eta'])

        return Results.key_str(config, value)

    def __init__(self, path):

        self.path = path

        self.master = dict()

        self.vals = {
            'lambda': [], 'Omega': [], 'model': [], 't': [], 'M': [], 'N': [],
            'eta': [], 'k_units_pi': []
        }

        dirs = listdir_fullpath(self.path)

        for d in dirs:
            config = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            mapping = yaml.safe_load(open(os.path.join(d, "mapping.yaml")))

            # These lower level directories correspond to the parameter sets
            # iterated over in a single config file
            for key, value in mapping.items():
                subdir = os.path.join(d, f"{key:03}")

                # Then we iterate over the partitionings of the grid within a
                # single trial
                A = np.concatenate([
                    np.loadtxt(os.path.join(p, "res.txt"))
                    for p in listdir_fullpath(subdir)
                ], axis=0)

                # Sort by w, just in case the results are not already sorted
                A = A[A[:, 0].argsort()]

                # Assign this entry a key
                self.master[self._assign_key(config, value)] = A

        self.defaults = dict()
        for key in list(self.vals.keys()):
            self.vals[key] = np.unique(self.vals[key])
            if len(self.vals[key]) == 1:
                self.defaults[key] = self.vals[key][0]
            else:
                self.defaults[key] = None

    def info(self):
        for key, value in self.vals.items():
            if key == 'k_units_pi':
                key = 'kdpi'
            print(key, "\t", value)

    def __call__(self, **kwargs):

        keys = list(kwargs.keys())

        config = {
            'model': kwargs['model']
            if 'model' in keys else self.defaults['model'],
            't': kwargs['t'] if 't' in keys else self.defaults['t'],
            'Omega': kwargs['Omega']
            if 'Omega' in keys else self.defaults['Omega'],
        }

        value = {
            'M': kwargs['M'] if 'M' in keys else self.defaults['M'],
            'N': kwargs['N'] if 'N' in keys else self.defaults['N'],
            'eta': kwargs['eta'] if 'eta' in keys else self.defaults['eta'],
            'k_units_pi': kwargs['k']
            if 'k' in keys else self.defaults['k_units_pi'],
            'lambda': kwargs['lam']
            if 'lam' in keys else self.defaults['lambda'],
        }
        return self.master[Results.key_str(config, value)]

    def lambda_band(self, **kwargs):
        """Returns the `lambda` band structure (E0 as a function) of lambda."""

        band = []
        for lambd in self.vals['lambda']:
            kwargs['lam'] = lambd
            A = self.__call__(**kwargs)
            band.append(A[:, 0][find_peaks(A[:, 1])[0][0]])
        return self.vals['lambda'], np.array(band)

    def lambda_band_exact(self, **kwargs):
        """Uses an analytic equation to determine the true energy. Requires two
        eta values."""

        band = []
        for lambd in self.vals['lambda']:

            d1 = copy.deepcopy(kwargs)
            d1["eta"] = d1["eta1"]
            d1["lam"] = lambd
            d1.pop("eta1")
            d1.pop("eta2")
            A1 = self.__call__(**d1)

            d2 = copy.deepcopy(kwargs)
            d2["eta"] = d2["eta2"]
            d2["lam"] = lambd
            d2.pop("eta1")
            d2.pop("eta2")
            A2 = self.__call__(**d2)

            # Find the ballpark maxima
            m1 = find_peaks(A1[:, 1])[0][0]
            m2 = find_peaks(A2[:, 1])[0][0]

            w1 = A1[:, 0][m1]
            a1 = A1[:, 1][m1]

            # We use the SAME omega value here
            a2 = A2[:, 1][m2]

            epsilon = (
                a1 * (w1 + 1j * d1["eta"]) - a2 * (w1 + 1j * d2["eta"])
            ) / (a1 - a2)
            band.append(epsilon.real)

        return self.vals['lambda'], np.array(band)

    def k_v_w(self, ninterp_w=1000, ninterp_k=1000, **kwargs):
        """Returns a band structure-like plot of k vs w."""

        overall_minimum_w = 1e16
        overall_maximum_w = -1e16

        # Find the overall grid
        for k in self.vals['k_units_pi']:
            kwargs['k'] = k
            A = self.__call__(**kwargs)
            overall_maximum_w = max(overall_maximum_w, np.max(A[:, 0]))
            overall_minimum_w = min(overall_minimum_w, np.min(A[:, 0]))

        band = []
        wgrid = np.linspace(
            overall_minimum_w, overall_maximum_w, ninterp_w,
            endpoint=True
        )
        kgrid = np.linspace(
            self.vals['k_units_pi'][0], self.vals['k_units_pi'][-1], ninterp_w,
            endpoint=True
        )

        for k in self.vals['k_units_pi']:
            kwargs['k'] = k
            A = self.__call__(**kwargs)
            A_interp = np.interp(wgrid, A[:, 0], A[:, 1], right=0, left=0)
            band.append(A_interp)

        Z = np.array(band)
        f = interpolate.interp2d(
            wgrid, self.vals['k_units_pi'], Z, kind='cubic', fill_value=0,
            bounds_error=False
        )

        return wgrid, kgrid, f(wgrid, kgrid)
