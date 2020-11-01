#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
import math
import os
import shlex
import subprocess
import time


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


def mgf_sum_rule(w, s, order):
    return np.sum(s[1:] * w[1:]**order * np.diff(w))


def holstein_sum_rule_check(w, s, config):
    """Imports the wgrid (w), spectrum (s) and config and produces a summary
    of the sum rules."""

    ek = -2.0 * config.t * np.cos(config.k * config.a)
    g = config.g

    print("Sum rules ratios: (computed / analytic)")

    # First sum rule, area under curve is 1:
    s0 = mgf_sum_rule(w, s, 0)
    print(f"\t#0: {s0:.04f}")

    s1 = mgf_sum_rule(w, s, 1) / ek
    print(f"\t#1: {s1:.04f}")

    s2_ana = ek**2 + g**2
    s2 = mgf_sum_rule(w, s, 2) / s2_ana
    print(f"\t#2: {s2:.04f}")

    s3_ana = ek**3 + 2.0 * g**2 * ek + g**2 * config.Omega
    s3 = mgf_sum_rule(w, s, 3) / s3_ana
    print(f"\t#3: {s3:.04f}")


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
    `length` that sum to tota_sum. Not that cases such as length = 4 and
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


def assert_n_vec_legal(n_vec):
    """Ensures that the input n_vec object is legal, or that it corresponds to
    a Green's function."""

    if n_vec == [[0] for __ in range(len(n_vec))]:
        # Case where we have the Green's function
        return

    for yy in n_vec:
        assert all(isinstance(xx, int) for xx in yy)
        assert all(xx >= 0 for xx in yy)

        if len(yy) > 1:
            assert yy[0] > 0
            assert yy[-1] > 0


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
            print(output.strip().decode())

    rc = process.poll()
    return rc
