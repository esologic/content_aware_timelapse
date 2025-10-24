"""Common GPU managment and usage code."""

import gc
from contextlib import contextmanager
from typing import Generator, Iterable, Optional

import torch


@contextmanager
def gpu_cleanup(models: Optional[Iterable[torch.nn.Module]] = None) -> Generator[None, None, None]:
    """
    Context manager that ensures GPU resources are released after use.

    Deletes provided models (if any) and clears the PyTorch CUDA cache,
    even if an exception occurs within the context.

    :param models: Iterable of torch.nn.Module objects to delete after use.
    :return: None
    """
    try:
        yield
    finally:
        if models is not None:
            # Explicitly free the models
            for m in models:
                del m
            gc.collect()
        # Clear any cached CUDA memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
