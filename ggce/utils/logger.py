#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

"""Basic logging module."""

import logging
import sys

from ggce.utils.utils import adjust_log_msg_for_time

# logger_string_format = \
#     '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logger_string_format = \
    '%(asctime)s %(levelname)-8s %(message)s'

# https://stackoverflow.com/questions/384076/
# how-can-i-color-python-logging-output
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the
# foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = \
            message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': BLUE,
    'DEBUG': WHITE,
    'CRITICAL': MAGENTA,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg, "%y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) \
                + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


COLOR_FORMAT = formatter_message(logger_string_format, True)


# https://stackoverflow.com/
# questions/36337244/logging-how-to-set-a-maximum-log-level-for-a-handler
class LevelFilter(logging.Filter):

    def __init__(self, low, high):
        self._low = low
        self._high = high
        logging.Filter.__init__(self)

    def filter(self, record):
        if self._low <= record.levelno <= self._high:
            return True
        return False


def setup_logger(name="ggce-logger"):

    color_formatter = ColoredFormatter(COLOR_FORMAT)

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(color_formatter)
        handler.addFilter(LevelFilter(0, 20))
        logger.addHandler(handler)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARN)
        handler.setFormatter(color_formatter)
        handler.addFilter(LevelFilter(30, 50))
        logger.addHandler(handler)

    return logger


def setup_console_logger(log_file=None, name="ggce-console-logger"):

    if log_file is None:
        return None

    color_formatter = ColoredFormatter(COLOR_FORMAT)

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(color_formatter)
        logger.addHandler(handler)

    return logger


class Logger:

    def __init__(self, log_file=None, dummy=False, mpi_comm=None):
        """Initializes the Logger object, which is a lightweight wrapper for
        the python logging library.

        Provides support for both console and file logging. Also (TODO) will
        indlude support for logging MPI-enabled code.

        Parameters
        ----------
        log_file : str, optional
            The location and extension of the logging file output (the default
            is None, which means no log file will be produced).
        dummy : bool, optional
            If True, simply does nothing whenenver the logging commands
            are called. Used primarily for debugging (the default is False).
        mpi_comm: mpi4py.MPI.Intracomm, optional
            The MPI communicator accessed via MPI.COMM_WORLD. (The default is
            None, which is taken to imply a single MPI process).
        """

        self.dummy = dummy
        if mpi_comm is None:
            self.mpi_comm = None
            self.mpi_rank = 0
        else:
            self.mpi_comm = mpi_comm
            self.mpi_rank = self.mpi_comm.Get_rank()

        if not self.dummy:

            self._console_logger = setup_logger()
            self._file_logger = setup_console_logger(log_file)

            # The current logging level corresponds to the different loggers in
            # logging. For example 0 will log everything, 1 will log info and
            # above, 2 will log warnings and above, etc. Note this only
            # corresponds to the console output. If the log_file is defined,
            # everything is always logged to the log file.
            self._current_logging_level = 0

    def adjust_logging_level(self, level="debug"):
        """Adjusts the logging level for the console output.

        Enables loggers for the specified level and above. E.g. level='info'
        will suppress logging.DEBUG only, and leave the remainder (INFO,
        WARNING, ERROR and CRITICAL) enabled.

        Parameters
        ----------
        level : {debug, info, warning, error}, optional
            Set the minimum level of logging to be output to the outstream (the
            default is 'debug', which enables all logging).
        """

        if self.dummy:
            return

        level = level.lower()
        assert level in ['debug', 'info', 'warning', 'error']

        if level == 'debug':
            self._current_logging_level = 0
        elif level == 'info':
            self._current_logging_level = 1
        elif level == 'warning':
            self._current_logging_level = 2
        elif level == 'error':
            self._current_logging_level = 3

    def debug(self, msg, elapsed=None):
        if self.dummy:
            return
        if self.mpi_rank > 0:
            return
        msg = adjust_log_msg_for_time(msg, elapsed)
        if self._file_logger is not None:
            self._file_logger.debug(msg)
        if self._current_logging_level <= 0:
            self._console_logger.debug(msg)

    def info(self, msg, elapsed=None):
        if self.dummy:
            return
        if self.mpi_rank > 0:
            return
        msg = adjust_log_msg_for_time(msg, elapsed)
        if self._file_logger is not None:
            self._file_logger.info(msg)
        if self._current_logging_level <= 1:
            self._console_logger.info(msg)

    def warning(self, msg, elapsed=None):
        if self.dummy:
            return
        if self.mpi_rank > 0:
            return
        msg = adjust_log_msg_for_time(msg, elapsed)
        if self._file_logger is not None:
            self._file_logger.warning(msg)
        if self._current_logging_level <= 2:
            self._console_logger.warning(msg)

    def error(self, msg, elapsed=None):
        if self.dummy:
            return
        if self.mpi_rank > 0:
            return
        msg = adjust_log_msg_for_time(msg, elapsed)
        if self._file_logger is not None:
            self._file_logger.error(msg)
        if self._current_logging_level <= 3:
            self._console_logger.error(msg)

    def critical(self, msg, error_type=RuntimeError):
        if self.dummy:
            return

        if self.mpi_rank > 0 and self.mpi_comm is not None:
            self.mpi_comm.Abort()
            return

        if self._file_logger is not None:
            self._file_logger.critical(msg)

        # Always log a critical error
        self._console_logger.critical(msg)
        if error_type is not None:
            raise error_type(msg)
