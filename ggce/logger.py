from contextlib import contextmanager
import sys
from warnings import warn

from loguru import logger


try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    MPI_WORLD_SIZE = COMM.Get_size()
    MPI_RANK = COMM.Get_rank()

except ImportError:
    COMM = None
    MPI_WORLD_SIZE = 1
    MPI_RANK = 0


def generic_filter(names):
    if names == "all":
        return None

    def f(record):
        return record["level"].name in names

    return f


rank_string = str(MPI_RANK).zfill(3)
rank_string = f" ({rank_string}) "

DEBUG_FMT_WITH_MPI_RANK = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "{name}:{function}:{line}</> "
    "|<lvl>{level: <10}</>|%s<lvl>{message}</>" % rank_string
)

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
    stderr_filter_warnings=["WARNING"],
    stderr_filter=["ERROR", "CRITICAL"],
    stdout_fmt=DEBUG_FMT_WITHOUT_MPI_RANK,
    stderr_fmt=WARN_FMT_WITHOUT_MPI_RANK,
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

    # Only log info, success and warnings on RANK == 0
    if MPI_RANK == 0:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(stdout_filter),
            format=stdout_fmt,
        )
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(stderr_filter_warnings),
            format=stderr_fmt,
        )

    # Log errors and criticals on every rank
    logger.add(
        sys.stderr,
        colorize=True,
        filter=generic_filter(stderr_filter),
        format=stderr_fmt,
    )

    # We always exit on critical
    if MPI_WORLD_SIZE == 1:
        logger.add(lambda _: sys.exit(1), level="CRITICAL")
    else:
        logger.add(lambda _: COMM.Abort(), level="CRITICAL")

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
