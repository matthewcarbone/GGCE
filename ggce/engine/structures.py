#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

from collections import namedtuple
import copy
import itertools
import numpy as np
import math

import yaml


def model_coupling_map(coupling_type, t, Omega, lam, ignore):
    """Returns the value for g, the scalar that multiplies the coupling in the
    Hamiltonian. Converts the user-input lambda value to this g.

    Parameters
    ----------
    coupling_type : {'H', 'SSH', 'bondSSH', 'EFB'}
        The desired coupling type. Can be Holstein, SSH (Peierls), bond SSH
        or Edwards Fermion Boson (EFB).
    t : float
        The hopping strength.
    Omega : float
        The (Einsten) boson frequency.
    lam : float
        The dimensionless coupling.
    ignore : bool
        If True, simply returns the value for lambda as g. Useful in some
        situations where the user wants to tune g directly.

    Returns
    -------
    float
        The value for the coupling (g).

    Raises
    ------
    RuntimeError
        If an unknown coupling type is provided.
    """

    if ignore:
        return lam

    if coupling_type == 'H':  # Holstein
        return math.sqrt(2.0 * t * Omega * lam)
    elif coupling_type == 'EFB':  # EFB convention lam = g for convenience
        return lam
    elif coupling_type == 'SSH':  # SSH
        return math.sqrt(t * Omega * lam / 2.0)
    elif coupling_type == 'bondSSH':  # bond SSH (note this is a guess)
        return math.sqrt(t * Omega * lam)
    else:
        raise RuntimeError(f"Unknown coupling_type type {coupling_type}")


class GridParams:

    def __init__(self, path):
        self.grid_info = yaml.safe_load(open(path, 'rb'))

    @staticmethod
    def _get_grid_helper(linspace, vals, round_values):
        """TODO

        [description]

        Parameters
        ----------
        linspace : {[type]}
            [description]
        vals : {[type]}
            [description]
        round_values : {[type]}
            [description]

        Returns
        -------
        np.array
            The grid.
        """

        if linspace:
            assert all([isinstance(xx, list) for xx in vals])
            assert all([len(xx) == 3 for xx in vals])
            assert all(xx[0] < xx[1] for xx in vals)

            return np.round(np.sort(np.concatenate([
                np.linspace(*c, endpoint=True) for c in vals
            ])), round_values)

        else:
            assert isinstance(vals, list)
            return np.round(vals, round_values)

    def _get_grid_zero_temperature_ground_state(
        self, grid_type, round_values=8
    ):
        """Returns the desired grid, modified for the ground state calculation.

        Parameters
        ----------
        grid_type : {'k', 'w'}
        round_values : int, optional
            Number of values to round in the final grids. (The default is 8).

        Returns
        -------
        array_like
            The numpy array grid, or a tuple containing the ground state
            calculation values.
        """

        if grid_type == 'k':
            vals = self.grid_info[grid_type]['vals']
            linspace = self.grid_info[grid_type]['linspace']
            assert isinstance(linspace, bool)
            return GridParams._get_grid_helper(linspace, vals, round_values)
        return (
            self.grid_info['w']['w0'],
            self.grid_info['w']['w_N_max'],
            self.grid_info['w']['eta_div'],
            self.grid_info['w']['eta_step_div'],
            self.grid_info['w']['next_k_offset_factor']
        )

    def get_grid(self, grid_type, round_values=8):
        """Returns the desired grid.

        Parameters
        ----------
        grid_type : {'k', 'w'}
        round_values : int, optional
            Number of values to round in the final grids. (The default is 8).

        Returns
        -------
        np.array
            The grid.
        """

        assert grid_type in ['k', 'w']

        if self.grid_info["protocol"] == "zero temperature ground state":
            return self._get_grid_zero_temperature_ground_state(
                grid_type, round_values=round_values
            )

        vals = self.grid_info[grid_type]['vals']
        linspace = self.grid_info[grid_type]['linspace']

        assert isinstance(linspace, bool)

        return GridParams._get_grid_helper(linspace, vals, round_values)


class _ProtocolBase:

    @staticmethod
    def _assert_parameters(param_name, data, model_len):
        """Sanity checks the input parameters for any issues.

        Specifically, this method doesn't check against the protocol, but it
        does ensure that all specified parameters have the correct 'format'.

        Parameters
        ----------
        param_name : str
            The key.
        data : dict
        model_len : int
            The 'length' of the provided model.

        Raises
        ------
        RuntimeError
            Raised every time there is a parameter issue. For example, an
            unknown parameter type, or a parameter contains an illegal
            configuration.
        """

        # These parameters require a list of lists (for prod or zip), or
        # simply a list (for solo)
        if param_name in [
            'M_extent', 'N_bosons', 'Omega', 'lam', 'g', 'M_tfd', 'N_tfd',
            'M_trace', 'N_trace'
        ]:
            assert isinstance(data['vals'], list)

            if data['cycle'] == 'solo':
                assert len(data['vals']) == model_len

            elif data['cycle'] in ['zip', 'prod']:
                assert all(isinstance(xx, list) for xx in data['vals'])
                assert all(len(xx) == model_len for xx in data['vals'])

            elif data['cycle'] == 'prod-linspace':
                assert param_name in ['Omega', 'lam', 'g']
                assert len(data['vals']) == model_len
                assert all(len(xx) == 3 for xx in data['vals'])
                assert all(
                    data['vals'][ii][2] == data['vals'][ii + 1][2]
                    for ii in range(model_len - 1)
                )

            else:
                raise RuntimeError(f"Unknown cycle: {data['cycle']}")

        elif param_name in [
            'hopping', 'broadening', 'absolute_extent', 'max_bosons_per_site',
            'temperature'
        ]:
            if data['cycle'] == 'solo':
                cond1 = isinstance(data['vals'], float)
                cond2 = isinstance(data['vals'], int)
                assert cond1 or cond2

            elif data['cycle'] in ['zip', 'prod']:
                assert isinstance(data['vals'], list)

            else:
                raise RuntimeError(f"Unknown cycle: {data['cycle']}")

        else:
            raise RuntimeError(f"Unknown parameter name {param_name}")

    def _init_params(self):
        """Initializes the solo, prod and zip dictionaries.

        Raises
        ------
        RuntimeError
            If the cycle value is invalid.
        """

        self.solo = dict()
        self.prod = dict()
        self.zip = dict()
        zip_lens = []
        model = self.input_params['model']

        for param_name, data in self.input_params['model_parameters'].items():
            _ProtocolBase._assert_parameters(param_name, data, len(model))

            if data['cycle'] == 'solo':
                self.solo[param_name] = data['vals']
            elif data['cycle'] == 'zip':
                zip_lens.append(len(data['vals']))
                self.zip[param_name] = data['vals']
            elif data['cycle'] == 'prod':
                self.prod[param_name] = data['vals']
            elif data['cycle'] == 'prod-linspace':
                dat = np.array([np.linspace(*xx) for xx in data['vals']]).T
                dat = np.round(dat, 3)
                self.prod[param_name] = [
                    [float(yy) for yy in xx] for xx in dat
                ]
            else:
                raise RuntimeError(f"Unknown cycle {data['cycle']}")

        # Assert that all lists in zip have the same length
        assert all([
            zip_lens[ii] == zip_lens[ii + 1]
            for ii in range(len(zip_lens) - 1)
        ])

        try:
            zip_indexes = [ii for ii in range(zip_lens[0])]
        except IndexError:
            zip_indexes = [None]

        # Define a product over the zip_indexes product and single terms
        self.master = list(itertools.product(
            zip_indexes, *list(self.prod.values())
        ))
        self._counter = 0
        self._counter_max = len(self.master)

    def save_grid(self, path):
        """Saves the grid information to disk as path. Also saves the desired
        protocol, which will be used for parsing the grid information later.

        Parameters
        ----------
        path : str
            The path to the location at which to save the yaml file containing
            the grid information.
        """

        d = {
            "k": self.input_params['grid_parameters']['k'],
            "w": self.input_params['grid_parameters']['w'],
            "protocol": self.input_params["protocol"]
        }

        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False)

    def __init__(self, input_params):
        self.input_params = input_params
        self._init_params()

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._counter_max:
            raise StopIteration

        current_parameters = self.master[self._counter]

        # Deal with the solo parameters
        d = copy.deepcopy(self.solo)
        d['model'] = self.input_params['model']
        d['info'] = self.input_params['info']
        d['protocol'] = self.input_params['protocol']

        # The zipped parameters
        for key, value in self.zip.items():
            if current_parameters[0] is not None:
                d[key] = value[current_parameters[0]]

        # And the product parameters
        ii = 1
        for key in list(self.prod.keys()):
            d[key] = current_parameters[ii]
            ii += 1

        self._counter += 1
        return d


def _assert_common(input_params):
    """Runs the assertions common to every trial.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    """

    model_parameter_keys = list(input_params['model_parameters'].keys())

    assert 'M_extent' in model_parameter_keys
    assert 'N_bosons' in model_parameter_keys
    assert 'Omega' in model_parameter_keys
    assert ('lam' in model_parameter_keys) or ('g' in model_parameter_keys)
    assert 'hopping' in model_parameter_keys
    assert 'broadening' in model_parameter_keys

    # Require the absolute extent no matter what if the model length is
    # longer than one.
    model = input_params['model']
    assert isinstance(model, list)
    assert len(model) > 0
    if len(model) > 1:
        assert 'absolute_extent' in model_parameter_keys


def _assert_zero_temperature(input_params):
    """Runs assertions on the zero-temperature calculations.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    """

    model_parameter_keys = list(input_params['model_parameters'].keys())

    # This is a zero-temperature calculation, temperature should be left
    # undefined
    assert 'temperature' not in model_parameter_keys

    # There also shouldn't be any other M_* or N_* keys defined since they
    # are not necessary for zero temperature. (Could use set and
    # isdisjoint here but using 'not in' for readability).
    assert 'M_tfd' not in model_parameter_keys
    assert 'N_tfd' not in model_parameter_keys
    assert 'M_trace' not in model_parameter_keys
    assert 'N_trace' not in model_parameter_keys


def _assert_TFD(input_params):
    """Runs assertions for the thermofield dynamics calculations.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    """

    model_parameter_keys = list(input_params['model_parameters'].keys())

    assert 'temperature' in model_parameter_keys
    assert 'M_tfd' in model_parameter_keys
    assert 'N_tfd' in model_parameter_keys
    assert 'M_trace' not in model_parameter_keys
    assert 'N_trace' not in model_parameter_keys
    assert 'absolute_extent' in model_parameter_keys


def _assert_trace(input_params):
    """Runs assertions for the finite temperature trace calculations.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    """

    model_parameter_keys = list(input_params['model_parameters'].keys())

    assert 'temperature' in model_parameter_keys
    assert 'M_tfd' not in model_parameter_keys
    assert 'N_tfd' not in model_parameter_keys
    assert 'M_trace' in model_parameter_keys
    assert 'N_trace' in model_parameter_keys
    assert 'absolute_extent' in model_parameter_keys


def _assert_standard_grid(input_params, which):
    """Runs assertions to check that the grid_params are correct.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    which : {'k', 'w'}
        Run assertions for either the k or omega grids.
    """

    grid_parameter_keys = list(input_params['grid_parameters'][which].keys())

    assert 'vals' in grid_parameter_keys
    assert 'linspace' in grid_parameter_keys


def _assert_gs_grid(input_params):
    """Asserts that the correct parameters are contained for the ground state
    calculations.

    Parameters
    ----------
    input_params : dict
        The loaded input parameters.
    """

    grid_parameter_keys = list(input_params['grid_parameters']['w'].keys())

    assert 'w0' in grid_parameter_keys
    assert 'w_N_max' in grid_parameter_keys
    assert 'eta_div' in grid_parameter_keys
    assert 'eta_step_div' in grid_parameter_keys
    assert 'next_k_offset_factor' in grid_parameter_keys


class ProtocolZeroTemperature(_ProtocolBase):

    def __init__(self, input_params):
        _assert_common(input_params)
        _assert_zero_temperature(input_params)
        _assert_standard_grid(input_params, 'k')
        _assert_standard_grid(input_params, 'w')
        super().__init__(input_params)


class ProtocolZeroTemperatureGroundState(_ProtocolBase):

    def __init__(self, input_params):
        _assert_common(input_params)
        _assert_zero_temperature(input_params)
        _assert_standard_grid(input_params, 'k')
        _assert_gs_grid(input_params)
        super().__init__(input_params)


class ProtocolTFD(_ProtocolBase):

    def __init__(self, input_params):
        _assert_common(input_params)
        _assert_TFD(input_params)
        _assert_standard_grid(input_params, 'k')
        _assert_standard_grid(input_params, 'w')
        super().__init__(input_params)


class ProtocolTrace(_ProtocolBase):

    def __init__(self, input_params):
        _assert_common(input_params)
        _assert_trace(input_params)
        _assert_standard_grid(input_params, 'k')
        _assert_standard_grid(input_params, 'w')
        super().__init__(input_params)


# Define a protocol mapping, which maps the user-defined protocol in the input
# files to classes defined above.
protocol_mapping = {
    "zero temperature": ProtocolZeroTemperature,
    "tfd": ProtocolTFD,
    "trace": ProtocolTrace,
    "zero temperature ground state": ProtocolZeroTemperatureGroundState
}


def parse_inp(inp_path):
    """Parses the user-generated input yaml file and returns the LoadedParams
    and GridParams classes.

    Parameters
    ----------
    inp_path : str
        The path to the yaml file to load.

    Returns
    -------
    _ProtocolBase
        A derived class of _ProtocolBase.

    Raises
    ------
    KeyError
        If any of the user-defined keys are incorrect.
    """

    loaded_params = yaml.safe_load(open(inp_path, 'r'))

    try:
        key = loaded_params['protocol']
    except KeyError:
        raise KeyError("Key 'protocol' must be defined in the input file")

    # After loading, check the protocol that the user defined.
    try:
        protocol = protocol_mapping[key](loaded_params)
    except KeyError as err:
        print(f"Caught error {err}; likely:")
        raise KeyError(f"Unknown user-defined protocol '{key}' in input file")

    return protocol


# Define a namedtuple which contains the shift indexes, x and y, the dagger
# status, d, the coupling term, g, and the boson frequency and type (index)
SingleTerm = namedtuple("SingleTerm", ["x", "y", "d", "g", "bt"])

# Define another namedtuple which contains only the terms that the f-functions
# need to calculate their prefactors, thus saving space.
fFunctionInfo = namedtuple("fFunctionInfo", ["a", "t", "Omega"])


class SystemParams:

    def _set_coupling(self, d):
        """Handle the coupling, which can be defined as either lambda or g
        itself. In the latter case, use_g will be set to True so as to ignore
        any dimensionless coupling conversion.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        self.lambdas = d.get('lam')
        self.use_g = False
        if self.lambdas is None:
            self.lambdas = d['g']  # If not defined, will throw a KeyError
            self.use_g = True

    def _set_temperature(self, d):
        """Handle setting the temperature. The temperature in the original
        input file must not be defined for zero-temperature calculations, and
        must be a float > 0.0 for the finite-temperature calculations.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol in ["tfd", "trace"]:
            self.temperature = d['temperature']
            assert self.temperature is not None
            assert self.temperature > 0.0
        else:
            assert d.get("temperature") is None
            self.temperature = 0.0

    def _set_extra_boson_clouds(self, d):
        """Handles setting extra boson cloud information depending on the
        protocol.

        Notes
        -----
        The number of bosons is actually an optional quantity in the parameter
        file due to the option of setting a hard boson constraint
        (max_bosons_per_site). These possibilities are handled in a later
        assertion.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol not in ["tfd", "trace"]:
            assert d.get("M_tfd") is None
            assert d.get("N_tfd") is None
            assert d.get("M_trace") is None
            assert d.get("N_trace") is None
        elif self.protocol == "tfd":
            self.M_tfd = d["M_tfd"]
            self.N_tfd = d.get("N_tfd")
            assert d.get("M_trace") is None
            assert d.get("N_trace") is None
        elif self.protocol == "trace":
            assert d.get("M_tfd") is None
            assert d.get("N_tfd") is None
            self.M_trace = d["M_trace"]
            self.N_trace = d.get("N_trace")

    def _set_absolute_extent_information(self, d):
        """Sets the absolute extent and runs assertions on it.

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        if self.protocol not in ["tfd", "trace"]:
            if self.n_boson_types > 1:
                self.absolute_extent = d["absolute_extent"]
                assert self.absolute_extent >= np.max(self.M)
            else:
                assert d.get("absolute_extent") is None
                self.absolute_extent = self.M[0]
        else:
            self.absolute_extent = d["absolute_extent"]
            assert self.absolute_extent >= np.max(self.M)
            if self.protocol == "tfd":
                assert self.absolute_extent >= np.max(self.M_tfd)
            elif self.protocol == "trace":
                assert self.absolute_extent >= np.max(self.M_trace)
        assert self.absolute_extent > 0

    def _set_max_bosons_per_site(self, d):
        """Handles the maximum bosons per site assertions (hard bosons).

        Parameters
        ----------
        d : dict
            Input dictionary.
        """

        self.max_bosons_per_site = d.get('max_bosons_per_site')
        if self.max_bosons_per_site is not None:
            assert self.max_bosons_per_site > 0
            assert isinstance(self.max_bosons_per_site, int)
            assert self.N is None
            self.N = [
                self.max_bosons_per_site * self.n_boson_types * m
                for m in self.M
            ]

            if self.protocol == "tfd":
                assert self.N_tfd is None
                self.N_tfd = [
                    self.max_bosons_per_site * self.n_boson_types * m
                    for m in self.M_tfd
                ]
            elif self.protocol == "trace":
                assert self.N_trace is None
                self.N_trace = [
                    self.max_bosons_per_site * self.n_boson_types * m
                    for m in self.M_trace
                ]
        else:
            assert self.N is not None
            if self.protocol == "tfd":
                assert self.N_tfd is not None
            elif self.protocol == "trace":
                assert self.N_trace is not None

    def __init__(self, d):

        # Start with parameters that are required for all trials
        self.M = d['M_extent']
        self.N = d.get('N_bosons')
        self.t = d['hopping']
        self.eta = d['broadening']
        self.a = 1.0  # Hard code lattice constant
        self.Omega = d['Omega']
        self.protocol = d['protocol']

        # Handle the coupling
        self._set_coupling(d)

        # Handle temperature
        self._set_temperature(d)

        # Handle extra boson clouds due to TFD or other finite-T methods
        self._set_extra_boson_clouds(d)

        # Set the model
        self.models = d['model']
        self.n_boson_types = len(self.models)
        assert self.n_boson_types == len(self.M)

        # Handle the absolute extent information
        self._set_absolute_extent_information(d)

        # Handle the hard boson constraints
        self._set_max_bosons_per_site(d)

    def get_fFunctionInfo(self):
        return fFunctionInfo(a=self.a, t=self.t, Omega=self.Omega)

    def _extend_terms(self, m, g, bt):
        """Helper method to extent the self.terms list.

        This method contains the 'programmed' notation of the coupling terms.
        Every model must have a corresponding string matching the cases below.

        Parameters
        ----------
        m : {'H', 'EFB', 'bondSSH', 'SSH'}
            The model type.
        g : float
            The coupling term (multiplying V in the Hamiltonian).
        bt : int
            The boson type index. Indexes the place in the model list. For
            example, if the current boson type is 1 and the model is
            ['H', 'SSH'], then the boson corresponds to an SSH phonon.

        Raises
        ------
        RuntimeError
            If the model type is unknown.
        """

        if m == 'H':
            self.terms.extend([
                SingleTerm(x=0, y=0, d='+', g=-g, bt=bt),
                SingleTerm(x=0, y=0, d='-', g=-g, bt=bt)
            ])
        elif m == 'EFB':
            self.terms.extend([
                SingleTerm(x=1, y=1, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=0, d='-', g=g, bt=bt)
            ])
        elif m == 'bondSSH':
            self.terms.extend([
                SingleTerm(x=1, y=0.5, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0.5, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-0.5, d='-', g=g, bt=bt)
            ])
        elif m == 'SSH':
            self.terms.extend([
                SingleTerm(x=1, y=0, d='+', g=g, bt=bt),
                SingleTerm(x=1, y=0, d='-', g=g, bt=bt),
                SingleTerm(x=1, y=1, d='+', g=-g, bt=bt),
                SingleTerm(x=1, y=1, d='-', g=-g, bt=bt),
                SingleTerm(x=-1, y=-1, d='+', g=g, bt=bt),
                SingleTerm(x=-1, y=-1, d='-', g=g, bt=bt),
                SingleTerm(x=-1, y=0, d='+', g=-g, bt=bt),
                SingleTerm(x=-1, y=0, d='-', g=-g, bt=bt)
            ])
        else:
            raise RuntimeError("Unknown model type when setting terms")

    def _get_coupling_prefactors(self, Omega):
        """Get's the TFD coupling prefactors.

        The TFD prefactors are defined clearly in e.g. JCP 145, 224101 (2016).

        Parameters
        ----------
        Omega : float
            The (Einstein) phonon frequency.

        Returns
        -------
        float, float
            The modifying prefactor to the real and fictitious couplings.
        """

        if self.protocol == "tfd":
            assert self.temperature > 0.0
            beta = 1.0 / self.temperature
            theta_beta = np.arctanh(np.exp(-beta * Omega / 2.0))
            V_prefactor = np.cosh(theta_beta)
            V_tilde_prefactor = np.sinh(theta_beta)
            return V_prefactor, V_tilde_prefactor
        else:
            return 1.0, None

    def _adjust_bosons_if_necessary(self):
        """Adjusts all attributes according to e.g. TFD.

        Note that this method essentially does nothing if the protocol is not
        TFD.
        """

        # Adjust the number of boson types according to thermofield
        if self.protocol == "tfd":
            self.n_boson_types *= 2  # Thermo field "double"
            assert isinstance(self.M, list)
            assert isinstance(self.N, list)
            assert isinstance(self.Omega, list)
            assert isinstance(self.lambdas, list)
            assert isinstance(self.models, list)

            new_M = []
            new_N = []
            new_Omega = []
            new_lambdas = []
            new_models = []

            for ii in range(len(self.models)):
                new_M.extend([
                    self.M[ii], self.M[ii]
                    if self.M_tfd is None else self.M_tfd[ii]
                ])
                new_N.extend([
                    self.N[ii], self.N[ii]
                    if self.N_tfd is None else self.N_tfd[ii]
                ])

                # Need the negative Omega here to account for the TFD truly.
                # the term's value for Omega is never actually called. Here, we
                # note that the boson frequency is NEGATIVE, indicative of the
                # fictitious space!
                new_Omega.extend([self.Omega[ii], -self.Omega[ii]])
                new_lambdas.extend([self.lambdas[ii], self.lambdas[ii]])
                new_models.extend([self.models[ii], self.models[ii]])

            self.M = new_M
            self.N = new_N
            self.Omega = new_Omega

            # Some of these parameters aren't used but we'll redfine them
            # anyway for consistency. Some of this is actually used in logging
            # so it's still useful.
            self.lambdas = new_lambdas
            self.models = new_models
            self.models_vis = []
            for ii, m in enumerate(self.models):
                if ii % 2 == 0:  # Even
                    self.models_vis.append(m)
                else:
                    self.models_vis.append(f"fict({m})")
        else:
            self.models_vis = self.models

    def prime(self):
        """Initializes the terms object, which contains the critical
        information about the Hamiltonian necessary for running the
        computation. Note that the sign is *relative*, so as long as
        every term in V is multipled by an overall factor, and each term has
        the correct sign relative to the others, the result will be the
        same."""

        self.terms = []

        bt = 0

        for (m, bigOmega, lam) in zip(self.models, self.Omega, self.lambdas):
            g = model_coupling_map(m, self.t, bigOmega, lam, self.use_g)

            # Handle the TFD stuff if necessary
            V_prefactor, V_tilde_prefactor = \
                self._get_coupling_prefactors(bigOmega)

            self._extend_terms(m, g*V_prefactor, bt)
            bt += 1

            # Now we implement the thermo field double changes to the
            # coupling prefactor, if necessary.
            if self.protocol == "tfd":
                self._extend_terms(m, g*V_tilde_prefactor, bt)
                bt += 1

        self._adjust_bosons_if_necessary()
