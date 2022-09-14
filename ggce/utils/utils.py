from contextlib import contextmanager
from pathlib import Path
import time
import uuid

import numpy as np
import pickle
from scipy.optimize import curve_fit


class Buffer:
    def __init__(self, nbuff, target_directory):
        self.nbuff = nbuff
        self.counter = 0
        self.queue = []
        self.target_directory = Path(target_directory)

    def flush(self):
        if self.counter > 0:
            path = self.target_directory / Path(f"{uuid.uuid4().hex}.pkl")
            pickle.dump(self.queue, open(path, "wb"), protocol=4)
            self.counter = 0
            self.queue = []

    def __call__(self, val):
        self.queue.append(val)
        self.counter += 1
        if self.counter >= self.nbuff:
            self.flush()


def chunk_jobs(jobs, world_size, rank):
    return np.array_split(jobs, world_size)[rank].tolist()


def float_to_list(val):
    if isinstance(val, float):
        val = [val]
    return val


def flatten(t):
    return [item for sublist in t for item in sublist]


def time_remaining(time_elapsed, percentage_complete):
    """Returns the time remaining."""

    # time_elapsed / percent_elapsed = time_remaining / pc_remaining
    # time_remaining = time_elapased / percent_elapsed * pc_remaining
    if percentage_complete == 100:
        return 0.0
    return (100.0 - percentage_complete) * time_elapsed / percentage_complete


def _elapsed_time_str(dt):
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


def _adjust_log_msg_for_time(msg, elapsed):
    if elapsed is None:
        return msg
    (elapsed, units) = _elapsed_time_str(elapsed)
    return f"[{elapsed:.02f} {units}] {msg}"


@contextmanager
def timeit(stream, msg):
    """A simple utility for timing how long a certain block of code will take.

    .. hint::

        Here's an example::

            with timeit(logger.info, "Test foo"):
                foo()

    Parameters
    ----------
    stream : callable
        The callable function/method which must take only a string as an
        argument. Usually something like ``logger.info``.
    msg : str
        The message to pass to the ``stream``.
    """

    t0 = time.time()
    try:
        yield None
    finally:
        dt = time.time() - t0
        stream(_adjust_log_msg_for_time(msg, dt))


def peak_location_and_weight(w, A, Aprime, eta, eta_prime):
    """Assumes that the polaron peak is a Lorentzian has the same weight
    no matter the eta. With these assumptions, we can determine the
    location and weight exactly using two points, each from a different
    eta calculation."""

    numerator1 = np.sqrt(eta * eta_prime)
    numerator2 = A * eta - Aprime * eta_prime
    den1 = Aprime * eta - A * eta_prime
    den2 = A * eta - Aprime * eta_prime
    loc = w - np.abs(numerator1 * numerator2 / np.sqrt(den1 * den2))
    area = np.pi * A * ((w - loc) ** 2 + eta**2) / eta
    return loc, area


def peak_location_and_weight_wstep(w, wprime, A, Aprime, eta):
    """Takes two points (w, A) and (wprime, Aprime) from the same eta
    calculation and assumes they lie on the same Lorentzian of the form
    f(x) = C (eta/pi) / ( eta**2 + (x-loc)**2 ). Solves the equation to fit
    a Lorentzian through those two points by determining C, x0."""

    firstterm = A * Aprime * (w - wprime) ** 2 / (A - Aprime) ** 2
    secondterm = eta**2
    loc1 = (w * A - wprime * Aprime) / (A - Aprime) + np.sqrt(
        firstterm - secondterm
    )
    loc2 = (w * A - wprime * Aprime) / (A - Aprime) - np.sqrt(
        firstterm - secondterm
    )
    area1 = np.pi * A * ((w - loc1) ** 2 + eta**2) / eta
    area2 = np.pi * A * ((w - loc2) ** 2 + eta**2) / eta
    if area1 > 1.0:
        return loc2, area2
    else:
        return loc1, area1


def peak_location_and_weight_scipy(wrange, Arange, eta):
    """Takes a bunch of points lying on the Lorentzian and does scipy.minimize
    fit of a Lorentzian function to it. Outputs fit parameters and error."""

    fitparams, error = curve_fit(
        lorentzian, wrange, Arange, p0=[wrange[-1], 1, eta]
    )

    return fitparams, error


def lorentzian(w, loc, scale, eta):
    return scale * (eta / np.pi) / ((w - loc) ** 2 + eta**2)
