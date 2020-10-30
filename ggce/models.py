#!/usr/bin/env python3


__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


"""General information about the parameters

k : float, List[float]
    The momentum. Either a single value or list of values.
tf : float
    The kinetic hopping term.
tb, lambda_SSH : List[float], optional
    A list of the fermion-boson couplings. The length of the list is equal to
    the number of different types of bosons in the model. Usually, this is just
    1. In some cases, this can be optional, like in the case of the SSH model
    where lambda_SSH can be provided instead. The alpha_0 term, which is our
    tb, is related to lambda_SSH by the function alpha_from_lambda_SSH.
Omega : List[float]
    A list of the boson Einstein frequencies. Generally just of length 1.

Notes
-----
* tf is the prefactor that multiplies the kinetic energy term in the
  Hamiltonian.
* Omega is a list of prefactors that multiply the boson energy term in the
  Hamiltonian.
* tb is a list of prefactors that multiply each coupling term in the
  Hamiltonian.

"""

import numpy as np


ETA = 0.002
LATTICE_PARAMETER = 1.0


class SingleTerm:

    def __init__(self, *, alpha_idx, d, x, y, sign):
        """Initializer

        Parameters
        ----------
        alpha_idx : int
            Indexes the "model", which is considered to be a coupling + boson
            frequency pair.
        d : {'+', '-'}
            Creation or annihilation term.
        x, y : int
            References the fundamental atomic term of the model,
            c_i^dagger c_{i+x} b_{i+y}^d.
        sign : {1, -1}
            The sign of the term itself. Note that the tb inputs should
            generally be positive, although empirically for most models it
            probably won't matter.
        """

        assert isinstance(alpha_idx, int)
        self.alpha_idx = alpha_idx

        assert d in ['+', '-']
        self.d = d

        assert isinstance(x, int)
        self.x = x

        assert isinstance(y, int)
        self.y = y

        assert np.abs(sign) == 1.0
        self.sign = sign


class Config:

    def __init__(
        self, *, name=None, k=None, eta=None, tf=None, tb=None, Omega=None,
        lattice_parameter=None
    ):
        self._MODEL_NAME = None
        self.name = name
        self.k = k
        self.eta = eta

        self.tf = tf
        assert isinstance(self.tf, float) or isinstance(self.tf, int)

        if isinstance(Omega, list):
            self.Omega = Omega
        else:
            self.Omega = [Omega]

        self.tb = tb

        self.a = lattice_parameter

        self.terms = None

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, kval):
        if isinstance(kval, float):
            self._k = kval
        else:
            raise NotImplementedError("Invalid k")

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, etaval):
        if etaval is None:
            self._eta = ETA
        else:
            self._eta = etaval

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, aval):
        if aval is None:
            self._a = LATTICE_PARAMETER
        else:
            self._a = aval

    def parse_model_parameters(self):
        """Quantum mechanically, it is imperative to treat bosons from
        different models (couplings) that have the same frequency as the same
        boson. this method accounts for these cases."""

        self.alpha_to_Omega_map = None
        self.alpha_to_tb_map = None
        self.n_unique_bosons = None

        n_unique_alpha_idx = len(np.unique([t.alpha_idx for t in self.terms]))
        if n_unique_alpha_idx != len(self.tb):
            raise RuntimeError(
                "len(tb) must be equal to the number of unique alpha_idx in "
                "the model"
            )

        if len(self.Omega) == 1:

            # The user can specify a case where Omega is like [Omega_1] and
            # tb is like [tb1, tb2, ...] indicating a single boson frequency
            # with different coupling-model-dependent fermion-boson coupling
            # strengths. Then, the number of unique bosons is simply 1, and
            # each alpha index maps onto the same boson, and each alpha index
            # maps onto the proper coupling strength depending on the model.
            self.alpha_to_Omega_map = {ii: 0 for ii in range(len(self.tb))}
            self.alpha_to_tb_map = {ii: ii for ii in range(len(self.tb))}
            self.n_unique_bosons = 1
            # print("Initializing config with 1 unique boson frequency")

        elif len(self.Omega) == len(self.tb):

            # This is the much more complicated case where there are multiple
            # values for Omega, some of which could be equal.
            unique_omega = np.unique([omega for omega in self.Omega])

            if len(unique_omega) != len(self.Omega):
                raise NotImplementedError(
                    "Method not implemented to handle edge case of > 2 boson "
                    "types in which some may be the same"
                )

            self.alpha_to_Omega_map = {ii: ii for ii in range(len(self.tb))}
            self.alpha_to_tb_map = {ii: ii for ii in range(len(self.tb))}
            self.n_unique_bosons = len(self.tb)
            # print(
            #     f"Initializing config with {self.n_unique_bosons} unique "
            #     "boson frequencies"
            # )

        else:
            raise RuntimeError("Omega must be of length 1 or len(tb)")


class HolsteinConfig(Config):

    def __init__(self, **kwargs):
        self._MODEL_NAME = 'Holstein'

        super().__init__(**kwargs)

        if self.tb is None:
            raise RuntimeError("tb cannot be None for Holstein")

        self.terms = [
            SingleTerm(alpha_idx=0, d='+', x=0, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=0, y=0, sign=1.0)
        ]

        self.parse_model_parameters()


class EFBConfig(Config):

    def __init__(self, **kwargs):
        self._MODEL_NAME = 'EFB'

        super().__init__(**kwargs)

        if self.tb is None:
            raise RuntimeError("tb cannot be None for EFB")

        self.terms = [
            SingleTerm(alpha_idx=0, d='+', x=1, y=1, sign=1.0),
            SingleTerm(alpha_idx=0, d='+', x=-1, y=-1, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=1, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=-1, y=0, sign=1.0)
        ]

        self.parse_model_parameters()


class SSHConfig(Config):

    def __init__(self, **kwargs):
        self._MODEL_NAME = 'SSH'

        super().__init__(**kwargs)

        self.terms = [
            SingleTerm(alpha_idx=0, d='+', x=1, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=1, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='+', x=1, y=1, sign=-1.0),
            SingleTerm(alpha_idx=0, d='-', x=1, y=1, sign=-1.0),
            SingleTerm(alpha_idx=0, d='+', x=-1, y=-1, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=-1, y=-1, sign=1.0),
            SingleTerm(alpha_idx=0, d='+', x=-1, y=0, sign=-1.0),
            SingleTerm(alpha_idx=0, d='-', x=-1, y=0, sign=-1.0)
        ]

        self.parse_model_parameters()


class SSHHConfig(Config):

    def __init__(self, **kwargs):
        self._MODEL_NAME = 'SSH+H'

        super().__init__(**kwargs)

        self.terms = [
            SingleTerm(alpha_idx=0, d='+', x=1, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=1, y=0, sign=1.0),
            SingleTerm(alpha_idx=0, d='+', x=1, y=1, sign=-1.0),
            SingleTerm(alpha_idx=0, d='-', x=1, y=1, sign=-1.0),
            SingleTerm(alpha_idx=0, d='+', x=-1, y=-1, sign=1.0),
            SingleTerm(alpha_idx=0, d='-', x=-1, y=-1, sign=1.0),
            SingleTerm(alpha_idx=0, d='+', x=-1, y=0, sign=-1.0),
            SingleTerm(alpha_idx=0, d='-', x=-1, y=0, sign=-1.0),
            SingleTerm(alpha_idx=1, d='+', x=0, y=0, sign=1.0),
            SingleTerm(alpha_idx=1, d='-', x=0, y=0, sign=1.0)
        ]

        self.parse_model_parameters()


MODEL_MAP = {
    'SSH': SSHConfig,      # Peierls/SSH model
    'EFB': EFBConfig,      # Edwards Fermion Boson model
    'H': HolsteinConfig,   # Holstein model
    'SSH+H': SSHHConfig    # SSH + Holstein model
}
