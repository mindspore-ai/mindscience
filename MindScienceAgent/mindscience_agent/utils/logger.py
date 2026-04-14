# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Logging utilities for MindScienceAgent."""
import os
import sys
import time
import logging
import traceback
import threading

__all__ = ['init_logger']

_setup_logger_lock = threading.Lock()
GLOBAL_LOGGER = None
LOG_LEVEL = "INFO"  # will be replaced with log_config.level

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}


def _get_logger():
    """Get logger instance."""
    if GLOBAL_LOGGER:
        return GLOBAL_LOGGER

    raise ValueError("logger is not initialized, please call init_logger(logging_config) first to initialize it.")


def _get_stack_info(frame):
    """Get stack information."""
    stack_prefix = 'Stack (most recent call last):\n'
    sinfo = stack_prefix + "".join(traceback.format_stack(frame))
    return sinfo


def _find_caller(stack_info=False, stacklevel=1):   # pylint: disable=W0613
    """Find caller information from stack frames."""
    f = sys._getframe(3)    # pylint: disable=W0212
    sinfo = None
    # log_file is used to check caller stack frame
    log_file = os.path.normcase(f.f_code.co_filename)
    f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)", None
    while f:
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if log_file == filename:
            f = f.f_back
            continue
        if stack_info:
            sinfo = _get_stack_info(f)
        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
        break
    return rv


def _get_formatter():
    """Get string of log formatter."""
    formatter = '[%(levelname)s] %(asctime)s ' \
                '[%(filepath)s:%(lineno)d] %(message)s'
    return formatter


def _clear_handler(logger):
    """Clear the handlers that has been set, avoid repeated loading"""
    for handler in logger.handlers:
        logger.removeHandler(handler)


class _DataFormatter(logging.Formatter):
    """Log formatter with custom timestamp format.

    Args:
        fmt (str, optional): Specified format pattern. Defaults to None.
        **kwargs: Additional formatter parameters.
    """
    def __init__(self, fmt=None, **kwargs):
        """Initialize log formatter."""
        super().__init__(fmt=fmt, **kwargs)

    def formatTime(self, record, datefmt=None):
        """Override formatTime for uniform timestamp format."""
        created_time = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, created_time)

        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', created_time)
        msecs = str(round(record.msecs * 1000))
        # Format the time stamp
        return f'{timestamp}.{msecs[:3]}.{msecs[3:]}'

    def format(self, record):
        """Apply log format with specified pattern."""
        # NOTICE: when the Installation directory of mindspore changed,
        # ms_home_path must be changed
        va_install_home_path = 'mindscience_agent'
        idx = record.pathname.rfind(va_install_home_path)
        if idx >= 0:
            # Get the relative path of the file
            record.filepath = record.pathname[idx:]
        else:
            record.filepath = record.pathname
        return super().format(record)


def init_logger(config):
    """Init logger"""
    # The name of Submodule
    sub_module = 'MINDSCIENCEAGENT'
    # The name of Base log file
    pid = str(os.getpid())
    log_name = 'mindscience_agent.log.' + pid

    global GLOBAL_LOGGER    # pylint: disable=W0603

    _setup_logger_lock.acquire()    # pylint: disable=R1732
    try:
        if GLOBAL_LOGGER:
            return GLOBAL_LOGGER

        logger = logging.getLogger(name=f'{sub_module}.{log_name}')
        # Override findCaller on the logger, Support for getting log record
        logger.findCaller = _find_caller
        # Set log level
        global LOG_LEVEL    # pylint: disable=W0603
        LOG_LEVEL = LOG_LEVEL_MAP[config.level]
        logger.setLevel(LOG_LEVEL)
        # Set "propagate" attribute to False, stop searching up the hierarchy,
        # avoid to load the handler of the root logger
        logger.propagate = False
        # Get the formatter for handler
        formatter = _get_formatter()

        # Clean up handle to avoid repeated loading
        _clear_handler(logger)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.name = 'StreamHandler'
        console_handler.formatter = _DataFormatter(formatter)
        logger.addHandler(console_handler)

        GLOBAL_LOGGER = logger

    finally:
        _setup_logger_lock.release()
    return GLOBAL_LOGGER


def debug(msg, *args, **kwargs):
    """Debug level log."""
    _get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Info level log."""
    _get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Warning level log."""
    _get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Error level log."""
    _get_logger().error(msg, *args, **kwargs)
