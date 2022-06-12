from ._version import get_versions

from .logger import logger  # noqa
from .model import Model  # noqa
from .engine.system import System  # noqa
from .executors.solvers import SparseSolver  # noqa

__version__ = get_versions()["version"]
del get_versions
