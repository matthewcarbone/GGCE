#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import logging
import os
import numpy as np
import sys

# logging.getLogger("imported_module").setLevel(logging.WARNING)


K_VAL = 0.123
W_VAL = 0.456


class DisableLogger():

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
    else:
        M = 4
        N = 20

    sys.path.append("..")

    from ggce import system, structures

    nthreads = os.environ.get("OMP_NUM_THREADS")
    print(f"Benchmarking- detecting {nthreads} threads")

    np.show_config()

    with DisableLogger():
        config = structures.InputParameters(
            M=M, N=N, model='H', t=1.0, eta=0.03, lambd=1.0, Omega=1.0
        )
        sy = system.System(config)
        sy.initialize_generalized_equations()
        sy.initialize_equations()
        sy.generate_unique_terms()
        sy.prime_solver()
        G, meta = sy.solve(K_VAL, W_VAL)
        for ii, t in enumerate(meta['time']):
            print(f"{meta['As'][ii]} -> {t:.02f}")
