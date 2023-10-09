from .logger import logger  # noqa
from .model import Model  # noqa
from .engine.system import System  # noqa
from .executors.solvers import SparseSolver, DenseSolver  # noqa

try:
    from .executors.petsc4py.solvers import MassSolverMUMPS  # noqa
except ImportError:
    pass

# DO NOT CHANGE ANYTHING BELOW THIS
__version__ = '0.dev' # version placeholder for editable installs
# DO NOT CHANGE ANYTHING ABOVE THIS
