import numpy as np
import os
from pathlib import Path
import pickle
import uuid
import time


def get_GGCE_CONFIG_STORAGE(default_name=".GGCE/GGCE_config_storage"):
    """Returns the user-set value for the location to store the basis
    functions. If  is set as an environment variable,
    that value is returned. Else, the default of /default_name is
    returned.

    Parameters
    ----------
    default_name : str, optional
        The name of the directory [structure], relative to $HOME, where the
        basis functions should be stored.

    Returns
    -------
    Posix.Path
    """

    path = os.environ.get("GGCE_CONFIG_STORAGE", None)
    if path is None:
        path = Path.home() / Path(default_name)
    return path


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


class Metadata(dict):

    @classmethod
    def load(cls, path=None):
        """Loads in a metadata json object from disk. If the path is None,
        simply returns a standard Metadata object.
        
        Parameters
        ----------
        path : str, optional
            Path to the json file in question.
        
        Returns
        -------
        Metadata
            The metadata class.
        """

        if path is None:
            return cls(dict(), path=path)

        path = Path(path)

        if not path.exists():
            return cls(dict(), path=path)

        # with open(path, "r") as infile:
        #     d = json.load(infile)
        d = pickle.load(open(path, "rb"))
        return cls(d, path=path)

    def save(self):
        """Saves the state of the dictionary to the user-provided path.
        If the path is None, does nothing."""

        # with open(self._path, "w") as outfile:
        #     json.dump(dict(self), outfile, indent=4, sort_keys=True)

        if self._path is None:
            return

        pickle.dump(dict(self), open(self._path, "wb"), protocol=4)

    def __init__(self, d=dict(), path=None):
        super().__init__(d)
        self._path = path


def chunk_jobs(jobs, world_size, rank):
    return np.array_split(jobs, world_size)[rank].tolist()


def float_to_list(val):
    if isinstance(val, float):
        val = [val]
    return val


def flatten(t):
    return [item for sublist in t for item in sublist]


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


def adjust_log_msg_for_time(msg, _elapsed):
    if _elapsed is None:
        return msg
    (elapsed, units) = elapsed_time_str(_elapsed)
    return f"({elapsed:.02f} {units}) {msg}"


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


def mpi_required(func):
    """Decorator that ensures a class has an MPI communicator defined.

    [description]

    Parameters
    ----------
    func : {[type]}
        [description]

    """

    def wrapper(self, *args, **kwargs):
        assert self.mpi_comm is not None
        return func(self, *args, **kwargs)
    return wrapper


def peak_location_and_weight(w, A, Aprime, eta, eta_prime):
    """Assumes that the polaron peak is a Lorentzian has the same weight
    no matter the eta. With these assumptions, we can determine the
    location and weight exactly using two points, each from a different
    eta calculation."""

    numerator1 = np.sqrt(eta * eta_prime)
    numerator2 = (A * eta - Aprime * eta_prime)
    den1 = Aprime * eta - A * eta_prime
    den2 = A * eta - Aprime * eta_prime
    loc = w - np.abs(numerator1 * numerator2 / np.sqrt(den1 * den2))
    area = np.pi * A * ((w - loc)**2 + eta**2) / eta
    return loc, area
