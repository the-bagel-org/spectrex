"""Provide a logger that uses rich for colorized output or falls back to the standard logging module and context managers.
"""

import inspect
import logging
from contextlib import contextmanager
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, NOTSET, WARN, WARNING
from typing import Any, cast

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None


__ALL__ = [
    "temporarily_set_log_level",
    CRITICAL,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    DEBUG,
    WARNING,
    WARN,
]


class RichLogger(logging.Logger):
    """
    This class is used to set up the logging.

    The main functionality added by this class over the built-in
    logging.Logger class is the ability to keep track of the origin of the
    messages, the ability to enable logging of warnings.warn calls and
    exceptions, and the addition of colorized output and context managers to
    easily capture messages to a file or list.
    """

    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):

        if extra is None:
            extra = {}
        extra = cast(dict[str, Any], extra)
        if "origin" not in extra:
            current_module = inspect.getmodule(inspect.currentframe())
            if current_module is not None:
                extra["origin"] = str(current_module.__name__)
            else:
                extra["origin"] = "unknown"
        return logging.Logger.makeRecord(
            self,
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func=func,
            extra=extra,
            sinfo=sinfo,
        )

    def set_defaults(self):
        # Set levels
        self.setLevel("INFO")
        # set formatter
        if RichHandler is not None:
            FORMAT = "%(message)s  [grey50](%(origin)s.%(funcName)s)[/]"
            ch = RichHandler(markup=True)
            formatter = logging.Formatter(FORMAT, datefmt="[%X]")
            ch.setFormatter(formatter)
        else:
            FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ch = logging.StreamHandler()
            formatter = logging.Formatter(FORMAT, datefmt="[%X]")
            ch.setFormatter(formatter)
        # Set up the stdout handler
        self.addHandler(ch)
        return self


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return RichLogger(name).set_defaults()


@contextmanager
def temporarily_set_log_level(logger: logging.Logger, level: int):
    """Temporarily set logger level, restoring it on exit.

    Parameters
    ----------
    logger : logging.Logger
        Logger to modify.
    level : int
        Logging level to set.
    """
    original = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original)


def testing_logger():
    log = RichLogger("rich").set_defaults()
    log.info("Testing logger info")
