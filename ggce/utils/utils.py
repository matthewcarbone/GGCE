#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from pathlib import Path
import pickle
import shlex
import subprocess
import uuid
import time


LIFO_QUEUE_PATH = '.LIFO_queue.yaml'
JOB_DATA_PATH = 'job_data'


class LoggerOnRank:

    def __init__(self, rank, logger, debug_flag=False):
        self.rank = rank
        self.logger = logger
        self.debug_flag = debug_flag

    def debug(self, msg):
        if self.debug_flag:
            self.logger.debug(f"({self.rank}) {msg}")

    def info(self, msg):
        self.logger.info(f"({self.rank}) {msg}")

    def warning(self, msg):
        self.logger.warning(f"({self.rank}) {msg}")

    def error(self, msg):
        self.logger.error(f"({self.rank}) {msg}")

    def critical(self, msg):
        self.logger.critical(f"({self.rank}) {msg}")


class RankTools:
    """A helper class containing information about the current MPI communicator
    as well as the rank, and logger."""

    def __init__(self, communicator, logger, debug):
        self.size = communicator.size
        self.rank = communicator.rank
        self.logger = \
            LoggerOnRank(rank=self.rank, logger=logger, debug_flag=bool(debug))

    def chunk_jobs(self, jobs):
        """Returns self.SIZE chunks, each of which is a list which is a
        reasonably equally distributed representation of jobs."""

        return [jobs[ii::self.size] for ii in range(self.size)]


class Buffer:

    def __init__(self, nbuff, target_directory):
        self.nbuff = nbuff
        self.counter = 0
        self.queue = []
        self.target_directory = Path(target_directory)

    def flush(self):
        if self.counter > 0:
            path = self.target_directory / Path(f"{uuid.uuid4().hex}.pkl")
            pickle.dump(self.queue, open(path, 'wb'), protocol=4)
            self.counter = 0
            self.queue = []

    def __call__(self, val):
        self.queue.append(val)
        self.counter += 1
        if self.counter >= self.nbuff:
            self.flush()


def lorentzian(x, x0, a, gam):
    return abs(a) * gam**2 / (gam**2 + (x - x0)**2)


def flatten(t):
    return [item for sublist in t for item in sublist]


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


def listdir_fullpath(d):
    return [p for p in d.iterdir()]


def listdir_files_fp(d):
    x = listdir_fullpath(d)
    return [xx for xx in x if not xx.is_dir()]


def listdir_fullpath_dirs_only(d):
    dirs = listdir_fullpath(d)
    return [d for d in dirs if d.is_dir()]


def elapsed_time_str(dt):
    """Returns the elapsed time in variable format depending on how long
    the calculation took.

    Parameters
    ----------
    dt : {float}
        The elapsed time in seconds.

    Returns
    -------
    float, str
        The elapsed time in the format given by the second returned value.
        Either seconds, minutes, hours or days.
    """

    if dt < 10.0:
        return dt, "s"
    elif 10.0 <= dt < 600.0:  # 10 s <= dt < 10 m
        return dt / 60.0, "m"
    elif 600.0 <= dt < 36000.0:  # 10 m <= dt < 10 h
        return dt / 3600.0, "h"
    else:
        return dt / 86400.0, "d"


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
            print(output.strip().decode())

    rc = process.poll()
    return rc
