#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

"""Basic logging module."""

import logging
import sys

logger_string_format = \
    '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'

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


def setup_logger(name, log_file):

    color_formatter = ColoredFormatter(COLOR_FORMAT)

    logger = logging.getLogger(name)
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

    if log_file is not None:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.WARN)
        handler.setFormatter(color_formatter)
        logger.addHandler(handler)

    return logger


default_logger = setup_logger('info', log_file=None)
