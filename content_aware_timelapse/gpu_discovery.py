"""
Utilities for finding GPUs attached to the system
"""

from typing import Tuple, TypeAlias

import nvsmi

GPUDescription: TypeAlias = int


def discover_gpus() -> Tuple[GPUDescription, ...]:
    """
    Returns the currently visible GPUs.
    :return: A tuple of GPUs.
    """

    return tuple(gpu.id for gpu in nvsmi.get_available_gpus())
