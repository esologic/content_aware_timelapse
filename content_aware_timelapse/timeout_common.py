"""
For tracking how long things take.
"""

import time
from typing import Callable, ParamSpec, Tuple, TypeVar

T = TypeVar("T")


P = ParamSpec("P")


def measure_execution_time(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Tuple[T, float]:
    """
    Measures the execution time of a function and returns both the time taken and the result.
    :param func: A callable function to measure.
    :param args: Positional arguments to forward to the function.
    :param kwargs: Keyword arguments to forward to the function.
    :return: A tuple containing the execution time in seconds and the result of the function.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)  # Forward args and kwargs to the original function
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return result, time_taken


def measure_execution_time_decorator(func: Callable[P, T]) -> Callable[P, Tuple[T, float]]:
    """
    A decorator that measures the execution time of a function and returns
    both the time taken and the result of the function.
    :param func: A callable function to measure.
    :return: A function that returns a tuple of execution time and result.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Tuple[T, float]:
        return measure_execution_time(func, *args, **kwargs)  # Call the helper function

    return wrapper
