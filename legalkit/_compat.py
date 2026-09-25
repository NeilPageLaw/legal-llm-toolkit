"""
Compatibility helpers for the optional machine-learning dependencies.

The training and evaluation code supports a range of transformers and TRL
releases whose keyword arguments were renamed over time.
"""

import inspect
from collections.abc import Callable
from typing import Any


def dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """The from_pretrained keyword for the model dtype ("dtype" from transformers 4.56)."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    return {key: dtype}


def accepted_parameters(fn: Callable) -> set[str]:
    """Names of the keyword arguments a callable accepts."""
    return set(inspect.signature(fn).parameters)


def first_supported(parameters: set[str], *names: str) -> str | None:
    """The first of names that is an accepted parameter."""
    return next((name for name in names if name in parameters), None)
