import numpy as np
from pathlib import Path
import pickle
import uuid
import time
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
