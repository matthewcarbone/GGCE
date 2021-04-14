#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"


import argparse
from pathlib import Path
import shutil
import yaml

from ggce.overlord import Prime
from ggce.engine.structures import SystemParams


DUMMY_CACHE = "DUMMY_CACHE"
DUMMY_LIFO = "DUMMY_LIFO"


def test_zero_temperature_prime():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', dest='inp')
    a = args.parse_args([
        '-i', 'tests/example_input_files/TEST_inp_ZeroTemperature.yaml'
    ])
    primer = Prime(a)
    primer.cache_dir = Path(DUMMY_CACHE)
    primer.queue_path = Path(DUMMY_LIFO)
    run_name = primer.scaffold()
    config_path = Path(DUMMY_CACHE) / Path(run_name) / "configs/00000000.yaml"
    sy = SystemParams(yaml.safe_load(open(config_path)))
    sy.prime()
    shutil.rmtree(DUMMY_CACHE)
    Path.unlink(Path(DUMMY_LIFO))


def test_TFD():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', dest='inp')
    a = args.parse_args([
        '-i', 'tests/example_input_files/TEST_inp_TFD.yaml'
    ])
    primer = Prime(a)
    primer.cache_dir = Path(DUMMY_CACHE)
    primer.queue_path = Path(DUMMY_LIFO)
    run_name = primer.scaffold()
    config_path = Path(DUMMY_CACHE) / Path(run_name) / "configs/00000000.yaml"
    sy = SystemParams(yaml.safe_load(open(config_path)))
    sy.prime()
    shutil.rmtree(DUMMY_CACHE)
    Path.unlink(Path(DUMMY_LIFO))


def test_zero_temperature_ground_state_prime():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', dest='inp')
    a = args.parse_args([
        '-i', 'tests/example_input_files/TEST_inp_ZeroTemperatureGS.yaml'
    ])
    primer = Prime(a)
    primer.cache_dir = Path(DUMMY_CACHE)
    primer.queue_path = Path(DUMMY_LIFO)
    run_name = primer.scaffold()
    config_path = Path(DUMMY_CACHE) / Path(run_name) / "configs/00000000.yaml"
    sy = SystemParams(yaml.safe_load(open(config_path)))
    sy.prime()
    shutil.rmtree(DUMMY_CACHE)
    Path.unlink(Path(DUMMY_LIFO))
