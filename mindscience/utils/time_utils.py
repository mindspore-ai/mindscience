"""Time Utilities

This module provides utility functions for time-related operations in the MindScience toolkit.
It includes a decorator for measuring execution time of functions.
"""
import time
from .log_utils import print_log

def log_timer(func):
    r"""
    A decorator that calculates the end-to-end total time of the training step.

    Args:
        func (callable): The function to decorate. Should be a callable object.

    Returns:
        callable: The decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print_log(f"End-to-End total time: {end_time - start_time:.2f}s")
    return wrapper
