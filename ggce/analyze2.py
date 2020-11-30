#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import copy
import os

from itertools import product
import numpy as np
import pickle

from scipy.signal import find_peaks
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from ggce.utils import utils


class Results:

    @staticmethod
    def key_str(config, value):
        t = f"{config['model']}_{config['Omega']}_{config['t']}_"
        t += f"{value['k_units_pi']:.08f}_{config['lambda']}_{value['M']}" \
            + f"_{value['N']}_{value['eta']}"
        return t

    def _assign_key(self, config, value):
        """Returns the correct key value and logs the keys in their
        appropriate lists."""

        self.vals['model'].append(config['model'])
        self.vals['Omega'].append(config['Omega'])
        self.vals['t'].append(config['t'])
        self.vals['lambda'].append(config['lambda'])

        self.vals['k_units_pi'].append(value['k_units_pi'])
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

        (
            master_list, config_mapping, M_N_eta_k_mapping, package_cache_path
        ) = pickle.load(open(os.path.join(self.path, 'protocol.pkl'), 'rb'))
        perms = list(product(
            M_N_eta_k_mapping['M'], M_N_eta_k_mapping['N'],
            M_N_eta_k_mapping['eta'], M_N_eta_k_mapping['k_units_pi']
        ))

        config_directories = utils.listdir_fullpath_dirs_only(self.path)

        # Nesting begins with the configs
        for c_idx, c_dir in enumerate(config_directories):
            config = config_mapping[c_idx]

            # Then M -> N -> eta -> k_units_pi
            for perm in perms:
                target = utils.N_M_eta_k_subdir(
                    *perm, M_N_eta_k_mapping, c_idx
                )
                target = os.path.join(self.path, target)

                (M, N, eta, k_units_pi) = perm

                value = {'k_units_pi': k_units_pi, 'M': M, 'N': N, 'eta': eta}

                # G contains w, G.real, G.imag, elapsed time and largest matrix
                # size.
                G = np.loadtxt(os.path.join(target, 'res.txt'))

                # Sort by w, just in case the results are not already sorted
                G = G[G[:, 0].argsort()]

                # Assign this entry a key
                self.master[self._assign_key(config, value)] = G

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
            'lambda': kwargs['lam']
            if 'lam' in keys else self.defaults['lambda']
        }

        value = {
            'M': kwargs['M'] if 'M' in keys else self.defaults['M'],
            'N': kwargs['N'] if 'N' in keys else self.defaults['N'],
            'eta': kwargs['eta'] if 'eta' in keys else self.defaults['eta'],
            'k_units_pi': kwargs['k']
            if 'k' in keys else self.defaults['k_units_pi']
        }

        G = self.master[Results.key_str(config, value)]

        # Return the frequency grid and A(k, w)
        return G[:, 0], -G[:, 2] / np.pi

    def lambda_band(
        self, interp=False, cutoff=False, cutoff_eps=1e-6,
        **kwargs
    ):
        """Returns the `lambda` band structure (E0 as a function) of lambda.
        Note for interp to work we must assume that we're centered on the peak
        of interest."""

        band = []
        for lambd in self.vals['lambda']:
            kwargs['lam'] = lambd
            A = self.__call__(**kwargs)

            if not interp:
                y = A[1]
                x = A[0]
                peaks = find_peaks(y)
                band.append(x[peaks[0][0]])

            else:
                spl = ius(A[:, 0], A[:, 1])
                x = np.linspace(A[:, 0][0], A[:, 0][-1], 1000000)
                y = spl(x, ext='zeros')
                # loc = np.argmax(y)
                peaks = find_peaks(y)
                band.append(x[peaks[0][0]])
                # band.append(x[loc])

        if cutoff:

            final_lambdas = []
            final_band = []

            # Here, we display the errors for the provided N compared with
            # N - 1, if it exists.
            try:
                if 'M' in kwargs:
                    _, band2 = self.lambda_band(
                        interp=interp,
                        N=kwargs['N'] - 1,
                        M=kwargs['M']
                    )
                else:
                    _, band2 = self.lambda_band(
                        interp=interp,
                        N=kwargs['N'] - 1
                    )
            except KeyError:
                print(f"N - 1 = {kwargs['N']-1} DNE in this dataset")
                return self.vals['lambda'], np.array(band)
            for ii in range(len(band2)):
                delta = np.abs(band[ii] - band2[ii])
                appended = False
                if delta < cutoff_eps:
                    final_lambdas.append(self.vals['lambda'][ii])
                    final_band.append(band[ii])
                    appended = True
                print(
                    f"{ii}\t{band2[ii]:.04f}"
                    f"\t{(band[ii] - band2[ii]):.06f}\tappended {appended}"
                )
            return final_lambdas, final_band

        else:
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
            m1 = find_peaks(A1[1])[0][0]
            m2 = find_peaks(A2[1])[0][0]

            w1 = A1[0][m1]
            a1 = A1[1][m1]

            # We use the SAME omega value here
            a2 = A2[1][m2]

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
            overall_maximum_w = max(overall_maximum_w, np.max(A[0]))
            overall_minimum_w = min(overall_minimum_w, np.min(A[0]))

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
            A_interp = np.interp(wgrid, A[0], A[1], right=0, left=0)
            band.append(A_interp)

        Z = np.array(band)
        f = interpolate.interp2d(
            wgrid, self.vals['k_units_pi'], Z, kind='cubic', fill_value=0,
            bounds_error=False
        )

        return wgrid, kgrid, f(wgrid, kgrid)
