#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import numpy as np

np.random.seed(123)


@profile
def invert100():
    N = 100
    X = np.random.random((N, N)) + 1j * np.random.random((N, N))
    _ = np.linalg.inv(X)


@profile
def invert1000():
    N = 1000
    X = np.random.random((N, N)) + 1j * np.random.random((N, N))
    _ = np.linalg.inv(X)


@profile
def invert5000():
    N = 5000
    X = np.random.random((N, N)) + 1j * np.random.random((N, N))
    _ = np.linalg.inv(X)


if __name__ == '__main__':
    invert100()
    invert1000()
    invert5000()
