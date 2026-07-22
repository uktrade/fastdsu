"""Public Python API for fastdsu."""

from importlib.metadata import version

from fastdsu._core import DSU, connected_components

__version__ = version("fastdsu")

__all__ = ["DSU", "connected_components"]
