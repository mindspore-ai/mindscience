"""time utils"""
import time
from .log_utils import print_log

def log_timer(func):
    """
    A decorator calculates End to End total time in the training step.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print_log(f"End-to-End total time: {end_time - start_time:.2f}s")
    return wrapper
