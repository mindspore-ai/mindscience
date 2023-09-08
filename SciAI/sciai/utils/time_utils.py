"""time utils"""
import time
from datetime import datetime, timezone


def time_second():
    """
    Get time in milliseconds number, e.g., 1678243339.780746.

    Returns:
        float, time in millisecond.
    """
    return time.time()


def time_str():
    """
    Get time in string representation, e.g., "2000-12-31-23-59-59".

    Returns:
        str, time in string representation.
    """
    return f"{datetime.now(tz=timezone.utc):%Y-%m-%d-%H-%M-%S}"
