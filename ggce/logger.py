from contextlib import contextmanager
import sys
from warnings import warn

from loguru import logger


try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None


def generic_filter(names):
    if names == "all":
        return None

    def f(record):
        return record["level"].name in names

    return f


if COMM is None:
    RANK = " "
else:
    RANK = str(COMM.Get_rank()).zfill(3)
    RANK = f" ({RANK}) "

DEBUG_FMT_WITH_MPI_RANK = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "{name}:{function}:{line}</> "
    "|<lvl>{level: <10}</>|%s<lvl>{message}</>" % RANK
)

# DEBUG_FMT_WITHOUT_MPI_RANK = (
#     "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
#     "{name}:{function}:{line}</> "
#     "|<lvl>{level: <10}</>| <lvl>{message}</>"
# )

DEBUG_FMT_WITHOUT_MPI_RANK = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss}</> <lvl>{message}</>"
)

WARN_FMT_WITHOUT_MPI_RANK = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "{name}:{function}:{line}</> "
    "|<lvl>{level: <10}</>| <lvl>{message}</>"
)


def configure_loggers(
    stdout_filter=["INFO", "SUCCESS"],
    stdout_debug_fmt=DEBUG_FMT_WITH_MPI_RANK,
    stdout_fmt=DEBUG_FMT_WITHOUT_MPI_RANK,
    stderr_filter=["WARNING", "ERROR", "CRITICAL"],
    stderr_fmt=WARN_FMT_WITHOUT_MPI_RANK,
    run_as_mpi=False,
    enable_python_standard_warnings=False,
):
    """Configures the ``loguru`` loggers. Note that the loggers are initialized
    using the default values by default.

    .. important::

        ``logger.critical`` `always` terminates the program, either through
        ``COMM.MPI_Abort()`` if ``run_as_mpi`` is True, or ``sys.exit(1)``
        otherwise.

    Parameters
    ----------
    stdout_filter : list of str, optional
        List of logging levels to include in the standard output stream.
    stdout_debug_fmt : str, optional
        Loguru format for the special debug stream.
    stdout_fmt : str, optional
        Loguru format for the rest of the standard output stream.
    stderr_filter : list, optional
        List of logging levels to include in the standard error stream.
    stderr_fmt : str, optional
        Loguru format for the rest of the standard error stream.
    run_as_mpi : bool, optional
        If True, critical errors will run ``COMM.MPI_Abort()``. Otherwise,
        ``sys.exit(1)`` is called.
    enable_python_standard_warnings : bool, optional
        Raises dummy warnings on ``logger.warning`` and ``logger.error``.
    """

    logger.remove(None)  # Remove ALL handlers

    # # Use the MPI ranks if available
    if "DEBUG" in stdout_filter:
        stdout_filter = [xx for xx in stdout_filter if xx != "DEBUG"]
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["DEBUG"]),
            format=stdout_debug_fmt,
        )

    logger.add(
        sys.stdout,
        colorize=True,
        filter=generic_filter(stdout_filter),
        format=stdout_fmt,
    )

    logger.add(
        sys.stderr,
        colorize=True,
        filter=generic_filter(stderr_filter),
        format=stderr_fmt,
    )

    # We always exit on critical
    if not run_as_mpi:
        logger.add(lambda _: sys.exit(1), level="CRITICAL")
    else:
        logger.add(lambda _: COMM.MPI_Abort(), level="CRITICAL")

    if enable_python_standard_warnings:
        logger.add(lambda _: warn("DUMMY WARNING"), level="WARNING")
        logger.add(lambda _: warn("DUMMY ERROR"), level="ERROR")


def DEBUG():
    """Quick helper to enable DEBUG mode."""

    configure_loggers(stdout_filter=["DEBUG", "INFO", "SUCCESS"])


def _TESTING_MODE():
    """Enables a testing mode where loggers are configured as usual but where
    the logger.warning and logger.error calls actually also raise a dummy
    warning with the text "DUMMY WARNING" and "DUMMY ERROR", respectively.
    Used for unit tests."""

    configure_loggers(
        stdout_filter=["DEBUG", "INFO", "SUCCESS"],
        enable_python_standard_warnings=True,
    )


def DISABLE_DEBUG():
    """Quick helper to disable DEBUG mode."""

    configure_loggers(stdout_filter=["INFO", "SUCCESS"])


@contextmanager
def disable_logger():
    """Context manager for disabling the logger."""

    logger.disable("")
    try:
        yield None
    finally:
        logger.enable("")


@contextmanager
def _testing_mode():
    _TESTING_MODE()
    try:
        yield None
    finally:
        DEBUG()


@contextmanager
def debug():
    DEBUG()
    try:
        yield None
    finally:
        DISABLE_DEBUG()


DISABLE_DEBUG()
