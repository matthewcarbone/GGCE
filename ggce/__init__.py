"""Basic logging module."""

import sys

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None

from loguru import logger as loguru_logger

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


FILE_LOGGING_FMT = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "{name}:{function}:{line}</> "
    "[<lvl>{level}</>] <lvl>{message}</>"
)


class Logger:
    @staticmethod
    def generic_filter(names):
        if names == "all":
            return None

        def f(record):
            return record["level"].name in names

        return f

    @staticmethod
    def elapsed_time_str(dt):
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

    @staticmethod
    def adjust_log_msg_for_time(msg, _elapsed):
        if _elapsed is None:
            return msg
        (elapsed, units) = Logger.elapsed_time_str(_elapsed)
        return f"({elapsed:.02f} {units}) {msg}"

    def __init__(
        self,
        dummy=False,
        mpi_rank=0,
        logger=loguru_logger,
        stdout_filter=["DEBUG", "INFO", "SUCCESS"],
        stdout_FMT=FILE_LOGGING_FMT,
        stderr_filter=["WARNING", "ERROR", "CRITICAL"],
        stderr_FMT=FILE_LOGGING_FMT,
        log_file=None,
        log_file_FMT=FILE_LOGGING_FMT,
        log_file_rotation="2.0 GB",
    ):
        """Initializes the Logger object, which is a lightweight wrapper for
        the loguru library. Provides support for both console and file logging.

        Parameters
        ----------
        dummy : bool, optional
            If True, simply does nothing whenenver the logging commands
            are called. Used primarily for debugging (the default is False).
        mpi_rank : int
            The mpi_rank of the MPI process. Default is 0. Note logging other
            than debug is only output for mpi_rank 0.
        logger : TYPE, optional
            Description
        stdout_filter : list, optional
            Description
        stdout_FMT : TYPE, optional
            Description
        stderr_filter : list, optional
            Description
        stderr_FMT : TYPE, optional
            Description
        log_file : str, optional
            The location and extension of the logging file output (the default
            is None, which means no log file will be produced).
        log_file_FMT : TYPE, optional
            Description
        log_file_rotation : str, optional
            Description
        """

        self._dummy = dummy
        self._mpi_rank = mpi_rank
        self._stdout_logger_id = None
        self._stderr_logger_id = None
        self._file_logger_id = None
        self._logger = logger

        self._logger.remove(0)  # Remove the default logger

        if not self._dummy:

            # Setup the loggers
            self._stdout_logger_id = self._logger.add(
                sys.stdout,
                colorize=True,
                filter=Logger.generic_filter(stdout_filter),
                format=stdout_FMT,
            )
            self._stderr_logger_id = self._logger.add(
                sys.stderr,
                colorize=True,
                filter=Logger.generic_filter(stderr_filter),
                format=stderr_FMT,
            )

            if log_file is not None:
                self._file_logger_id = self._logger.add(
                    log_file,
                    filter=Logger.generic_filter("all"),
                    format=log_file_FMT,
                    rotation=log_file_rotation,
                )

            self._log(f"GGCE {__version__} initialized", "debug", None)

    def _log(self, msg, level, elapsed):
        if self._dummy:
            return
        msg = Logger.adjust_log_msg_for_time(msg, elapsed)
        if level == "debug":
            msg = f"RANK {self._mpi_rank:03}: {msg}"
        eval(f"self._logger.{level}('{msg}')")

    def debug(self, msg, elapsed=None):
        self._log(msg, "debug", elapsed)

    def info(self, msg, elapsed=None):
        self._log(msg, "info", elapsed)

    def success(self, msg, elapsed=None):
        self._log(msg, "success", elapsed)

    def warning(self, msg, elapsed=None):
        self._log(msg, "warning", elapsed)

    def error(self, msg, elapsed=None):
        self._log(msg, "error", elapsed)

    def critical(self, msg):
        self._log(msg, "critical", None)

        if COMM is not None:
            self._log("Critical error -> COMM.Abort()", "critical", None)
            COMM.Abort()
        else:
            self._log("Critical error -> sys.exit(1)", "critical", None)
            sys.exit(1)


def _setup_logger():
    """Exposes a way for the user to manually set the Logger object in the
    global scope."""

    global logger

    if COMM is None:
        mpi_rank = 0
    else:
        mpi_rank = COMM.Get_rank()
    logger = Logger(mpi_rank=mpi_rank)


_setup_logger()
