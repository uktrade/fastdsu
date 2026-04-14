"""Public Python API for fastdsu."""

from importlib.metadata import version

from fastdsu._core import DSU

__version__ = version("fastdsu")

__all__ = ["DSU"]
