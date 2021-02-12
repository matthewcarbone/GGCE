#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import logging
import os
import shlex
import subprocess
import time


from ggce.utils.logger import default_logger as dlog


LIFO_QUEUE_PATH = '.LIFO_queue.yaml'
JOB_DATA_PATH = 'job_data'


def lorentzian(x, x0, a, gam):
    return abs(a) * gam**2 / (gam**2 + (x - x0)**2)


def flatten(t):
    return [item for sublist in t for item in sublist]


class DisableLogger:

    def __init__(self, disable=True):
        self.disable = disable

    def __enter__(self):
        if self.disable:
            logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self.disable:
            logging.disable(logging.NOTSET)


# https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def bold(s):
    """Makes a string bold for console output."""

    return Color.BOLD + s + Color.END


def get_cache_dir():
    cache = os.environ.get('GGCE_CACHE_DIR')
    if cache is None:
        dlog.warning(
            "Environment variable GGCE_CACHE_DIR not found. "
            "Cache directory set to 'results'"
        )
        cache = 'results'
    # os.makedirs(cache, exist_ok=True)
    dlog.debug(f"Cache directory set to {cache}")
    return cache


def get_package_dir():
    cache = os.environ.get('GGCE_PACKAGES_DIR')
    if cache is None:
        cache = 'packages'
    # os.makedirs(cache, exist_ok=True)
    dlog.debug(f"Package directory set to {cache}")
    return cache


def listdir_fullpath(d):
    """https://stackoverflow.com/a/120948"""

    return [os.path.join(d, f) for f in os.listdir(d)]


def listdir_files_fp(d):
    x = [os.path.join(d, f) for f in os.listdir(d)]
    return [xx for xx in x if not os.path.isdir(xx)]


def listdir_fullpath_dirs_only(d):
    dirs = [os.path.join(d, f) for f in os.listdir(d)]
    return [d for d in dirs if os.path.isdir(d)]


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


def time_remaining(time_elapsed, percentage_complete):
    """Returns the time remaining."""

    # time_elapsed / percent_elapsed = time_remaining / pc_remaining
    # time_remaining = time_elapased / percent_elapsed * pc_remaining
    if percentage_complete == 100:
        return 0.0
    return (100.0 - percentage_complete) * time_elapsed / percentage_complete


def run_command(command, silent=True):
    """https://www.endpoint.com/blog/2015/01/28/
    getting-realtime-output-using-python"""

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output and not silent:
            dlog.debug(output.strip().decode())

    rc = process.poll()
    return rc
