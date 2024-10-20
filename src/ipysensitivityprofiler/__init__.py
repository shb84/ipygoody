"""Module entry point."""

# Copyright (c) 2018 Steven H. Berguin
# Distributed under the terms of the MIT License.

from ._controller import Controller
from ._model import openmdao_profiler, profiler
from ._view import View

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "profiler",
    "openmdao_profiler",
    "View",
    "Controller",
]
