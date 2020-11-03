#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
import os
import shlex
import subprocess
import time


from ggce.utils.logger import default_logger as dlog


def get_cache_dir():
    cache = os.environ.get('GGCE_CACHE_DIR')
    if cache is None:
        cache = 'results'
    os.makedirs(cache, exist_ok=True)
    return cache


def get_package_dir():
    cache = os.environ.get('GGCE_PACKAGES_DIR')
    if cache is None:
        cache = 'packages'
    os.makedirs(cache, exist_ok=True)
    return cache


def listdir_fullpath(d):
    """https://stackoverflow.com/a/120948"""

    return [os.path.join(d, f) for f in os.listdir(d)]


def listdir_files_fp(d):
    x = [os.path.join(d, f) for f in os.listdir(d)]
    return [xx for xx in x if not os.path.isdir(xx)]


def time_func(arg1=None):
    """source: http://scottlobdell.me/2015/04/decorators-arguments-python/"""

    def real_decorator(function):

        def wrapper(*args, **kwargs):

            aa = arg1
            if aa is None:
                aa = function.__name__

            t1 = time.time()
            x = function(*args, **kwargs)
            t2 = time.time()
            elapsed = (t2 - t1) / 60.0
            print(f"\t{aa} done {elapsed:.02f} m")
            return x

        return wrapper

    return real_decorator


def configuration_space_generator(length, total_sum):
    """Generator for yielding all possible combinations of integers of length
    `length` that sum to total_sum. Not that cases such as length = 4 and
    total_sum = 5 like [0, 0, 2, 3] need to be screened out, since these do
    not correspond to valid f-functions.

    Source of algorithm:
    https://stackoverflow.com/questions/7748442/
    generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """

    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in configuration_space_generator(
                length - 1, total_sum - value
            ):
                r = (value,) + permutation
                yield r


def assert_n_vec_legal(n, config_filter=None):
    """Ensures that the input n object is a legal configuration.

    Returns
    -------
    bool
        True if the configuration is legal, and False if it is not.
    """

    if (n[0] <= 0 or n[-1] <= 0):
        return False

    if config_filter is None:
        return True

    if config_filter == 'gaussian':
        M = len(n)
        N = sum(n)

        if M % 2 == 0:  # even
            x = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, M)
        else:
            x = np.linspace(-N // 2, N // 2, M)
        scale_factor = M**2 / 4.0 / np.log(N)
        y = N * np.exp(-x**2 / scale_factor)

        if any([n[ii] > y[ii] for ii in range(M)]):
            return False
        return True

    else:
        raise RuntimeError(f"Unknown filter {config_filter}")


def time_remaining(time_elapsed, percentage_complete):
    """Returns the time remaining."""

    # time_elapsed / percent_elapsed = time_remaining / pc_remaining
    # time_remaining = time_elapased / percent_elapsed * pc_remaining
    if percentage_complete == 100:
        return 0.0
    return (100.0 - percentage_complete) * time_elapsed / percentage_complete


def run_command(command):
    """https://www.endpoint.com/blog/2015/01/28/
    getting-realtime-output-using-python"""

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            dlog.info(output.strip().decode())

    rc = process.poll()
    return rc
